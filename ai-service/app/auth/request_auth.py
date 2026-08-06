"""Authenticate every request before any route runs.

A caller is either a user with a JWT (verified against jira-backend's public keys) or an internal
service holding the shared X-Internal-Token. For users, X-User-Id/X-Username are rewritten from the
token, so the rest of the service can keep reading those headers and a browser can't forge them.
Internal callers keep the identity headers they send (or none: eval scripts).
"""
from __future__ import annotations

import asyncio
import hmac
import logging

import jwt
from starlette.responses import JSONResponse

from app.auth.user_quota import AUTHENTICATED_USER_SCOPE_KEY

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({"/healthz", "/healthz/cache", "/metrics"})
# Ops data (token spend, model names), never meant for browsers.
INTERNAL_ONLY_PATHS = frozenset({"/stats"})

_INTERNAL_TOKEN = b"x-internal-token"
_AUTHORIZATION = b"authorization"
_IDENTITY_HEADERS = frozenset({b"x-user-id", b"x-username"})


class JwtVerifier:
    def __init__(self, jwks_url: str, issuer: str):
        # Caches keys and refetches on an unknown kid, so a key rotation needs no restart.
        self._jwks = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=300, timeout=5)
        self._issuer = issuer

    async def verify(self, token: str) -> dict:
        # PyJWKClient fetches with blocking urllib; keep that off the event loop.
        signing_key = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=self._issuer,
            options={"require": ["exp", "iss", "sub"]},
        )


class RequestAuthMiddleware:
    def __init__(self, app, verifier: JwtVerifier, internal_token: str):
        self._app = app
        self._verifier = verifier
        self._internal_token = internal_token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] == "OPTIONS" or scope["path"] in PUBLIC_PATHS:
            await self._app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        internal = _first(headers, _INTERNAL_TOKEN)
        if internal is not None:
            if not hmac.compare_digest(internal, self._internal_token):
                await _reject(scope, receive, send, 401, "Invalid internal token")
                return
            await self._app(scope, receive, send)
            return

        if scope["path"] in INTERNAL_ONLY_PATHS:
            await _reject(scope, receive, send, 403, "Internal endpoint")
            return

        authorization = (_first(headers, _AUTHORIZATION) or b"").decode("latin-1")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            await _reject(scope, receive, send, 401, "Sign in required")
            return
        try:
            claims = await self._verifier.verify(token.strip())
        except jwt.PyJWKClientConnectionError:
            logger.error("could not fetch JWT signing keys", exc_info=True)
            await _reject(scope, receive, send, 503, "Cannot verify sign-in right now")
            return
        except jwt.PyJWTError as exc:
            await _reject(scope, receive, send, 401, f"Invalid token: {exc}")
            return

        uid = claims.get("uid")
        if isinstance(uid, bool) or not (isinstance(uid, int) or (isinstance(uid, str) and uid.isdigit())):
            await _reject(scope, receive, send, 401, "Token has no numeric uid claim")
            return

        rewritten = [(k, v) for k, v in headers if k not in _IDENTITY_HEADERS]
        rewritten.append((b"x-user-id", str(uid).encode()))
        rewritten.append((b"x-username", str(claims["sub"]).encode()))
        await self._app(
            {**scope, "headers": rewritten, AUTHENTICATED_USER_SCOPE_KEY: str(uid)}, receive, send
        )


def _first(headers, name: bytes) -> bytes | None:
    for key, value in headers:
        if key == name:
            return value
    return None


async def _reject(scope, receive, send, status: int, detail: str) -> None:
    extra = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    await JSONResponse({"detail": detail}, status_code=status, headers=extra)(scope, receive, send)
