"""Per-user AI limits: requests per minute and dollars per day.

WAF limits per IP and LiteLLM budgets per service; neither stops one signed-in user from spending the
whole service budget. Counters live in Redis so every ai-service replica sees the same numbers.

Only JWT users are limited (internal-token callers such as eval scripts are not). Only endpoints
that start new AI work check the limits; continuing an existing workflow (approving a plan,
answering a clarification) is never blocked, so a user who just hit the budget can still finish
what they started. Its spend still counts.

The daily check runs before the request, and cost is only known after it, so in-flight requests
can overshoot the budget by about one request each.

Redis unreachable -> each process counts on its own until Redis is back. Limits still apply, just
per replica and starting from zero, so in the worst case a user gets one extra day's budget per
replica. Refusing all AI work instead would take the feature down over a bookkeeping outage.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import HTTPException, Request

from app.observability import USER_QUOTA_LOCAL_FALLBACK_TOTAL, USER_QUOTA_REJECTIONS_TOTAL

logger = logging.getLogger(__name__)

# Set by RequestAuthMiddleware only for JWT users; internal-token requests never have it.
AUTHENTICATED_USER_SCOPE_KEY = "ai.authenticated_user_id"

_DAY_SECONDS = 86_400


@dataclass(frozen=True)
class QuotaStatus:
    requests_per_minute: int
    daily_budget_usd: float
    spent_today_usd: float
    resets_at: str  # ISO-8601, next UTC midnight


class _LocalCounters:
    """This process's own counts, used only while Redis is unreachable."""

    def __init__(self):
        self._minute: dict[str, tuple[int, int]] = {}
        self._day: dict[str, tuple[int, float]] = {}

    def count_request(self, user_id: str, now: float) -> int:
        window = int(now // 60)
        current, count = self._minute.get(user_id, (window, 0))
        count = count + 1 if current == window else 1
        self._minute[user_id] = (window, count)
        return count

    def spent(self, user_id: str, now: float) -> float:
        day, spent = self._day.get(user_id, (int(now // _DAY_SECONDS), 0.0))
        return spent if day == int(now // _DAY_SECONDS) else 0.0

    def add_spend(self, user_id: str, now: float, cost_usd: float) -> None:
        self._day[user_id] = (int(now // _DAY_SECONDS), self.spent(user_id, now) + cost_usd)


class UserQuota:
    def __init__(
        self, redis_client, requests_per_minute: int, daily_budget_usd: float,
        clock: Callable[[], float] = time.time,
    ):
        self._redis = redis_client
        self._rpm = requests_per_minute
        self._daily_budget = daily_budget_usd
        self._clock = clock
        self._local = _LocalCounters()

    async def admit(self, user_id: str) -> None:
        """Raise 429 if the user is over either limit; otherwise count this request."""
        now = self._clock()
        try:
            spent = float(await self._redis.get(self._spend_key(user_id, now)) or 0)
            redis_up = True
        except Exception:  # noqa: BLE001 - see module docstring
            _redis_down("check")
            spent, redis_up = self._local.spent(user_id, now), False
        if spent >= self._daily_budget:
            USER_QUOTA_REJECTIONS_TOTAL.labels("daily_budget").inc()
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI budget reached (${self._daily_budget:.2f}). It resets at 00:00 UTC.",
                headers={"Retry-After": str(_seconds_to_midnight(now))},
            )
        count = await self._count_request(user_id, now) if redis_up else self._local.count_request(user_id, now)
        if count > self._rpm:
            USER_QUOTA_REJECTIONS_TOTAL.labels("rate").inc()
            raise HTTPException(
                status_code=429,
                detail=f"Too many AI requests: limit is {self._rpm} per minute. Try again shortly.",
                headers={"Retry-After": str(60 - int(now) % 60)},
            )

    async def _count_request(self, user_id: str, now: float) -> int:
        # Fixed one-minute window: simple, and the worst case (2x at a window edge) is fine here.
        minute_key = f"ai:quota:rpm:{user_id}:{int(now // 60)}"
        try:
            pipe = self._redis.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 120)
            count, _ = await pipe.execute()
            return count
        except Exception:  # noqa: BLE001 - see module docstring
            _redis_down("count")
            return self._local.count_request(user_id, now)

    async def add_spend(self, user_id: str, cost_usd: float) -> None:
        now = self._clock()
        key = self._spend_key(user_id, now)
        try:
            pipe = self._redis.pipeline()
            pipe.incrbyfloat(key, cost_usd)
            pipe.expire(key, 2 * _DAY_SECONDS)
            await pipe.execute()
        except Exception:  # noqa: BLE001 - see module docstring
            _redis_down("spend")
            self._local.add_spend(user_id, now, cost_usd)

    async def status(self, user_id: str) -> QuotaStatus:
        now = self._clock()
        try:
            spent = float(await self._redis.get(self._spend_key(user_id, now)) or 0)
        except Exception:  # noqa: BLE001 - see module docstring
            _redis_down("status")
            spent = self._local.spent(user_id, now)
        midnight = datetime.fromtimestamp(int(now) + _seconds_to_midnight(now), tz=timezone.utc)
        return QuotaStatus(
            requests_per_minute=self._rpm,
            daily_budget_usd=self._daily_budget,
            spent_today_usd=round(spent, 6),
            resets_at=midnight.isoformat(),
        )

    @staticmethod
    def _spend_key(user_id: str, now: float) -> str:
        day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        return f"ai:quota:spend:{user_id}:{day}"


def _redis_down(step: str) -> None:
    USER_QUOTA_LOCAL_FALLBACK_TOTAL.labels(step).inc()
    logger.warning("user quota: Redis unreachable, counting in this process only (%s)", step)


def _seconds_to_midnight(now: float) -> int:
    return _DAY_SECONDS - int(now) % _DAY_SECONDS


# The ledger reports spend from deep inside features (and from graph tasks that can outlive the
# request), so it reaches the quota through this module rather than through the request.
_active: Optional[UserQuota] = None
_pending: set = set()


def install(quota: Optional[UserQuota]) -> None:
    global _active
    _active = quota


def record_spend(user_id: str, cost_usd: float) -> None:
    """Add cost to the user's daily total without blocking the caller."""
    quota = _active
    if quota is None or not user_id or cost_usd <= 0:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(quota.add_spend(user_id, cost_usd))
    # Keep a reference until done, or the task can be garbage-collected mid-flight.
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def enforce_user_quota(request: Request) -> None:
    """FastAPI dependency for endpoints that start new AI work."""
    quota: Optional[UserQuota] = getattr(request.app.state, "user_quota", None)
    user_id = request.scope.get(AUTHENTICATED_USER_SCOPE_KEY)
    if quota is None or not user_id:
        return
    await quota.admit(user_id)
