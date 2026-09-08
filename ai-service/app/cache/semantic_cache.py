"""Answer cache for POST /ask in Redis Stack: exact match first, then semantic (vector KNN) match.

Repeated or near-duplicate questions otherwise pay for retrieval, the corrective loop and generation
every time. Hand-rolled rather than GPTCache (see README "Semantic query cache").

Redis Stack (RediSearch) does the nearest-neighbour search inside Redis; the first version scanned
embeddings in Python, O(n) per lookup. Verified on a real redis-stack-server: COSINE returns a
distance (similarity = 1 - distance), and a TAG filter combined with KNN really returns zero hits for
another space, so the permission boundary is enforced by Redis itself.

Two tiers:
  1. Exact: normalized question + space_ids, one GET.
  2. Semantic: KNN over question embeddings (via vectorization-service /embed), filtered to this
     space_key and data_version. A semantic hit is also saved under the exact wording asked, so a
     re-ask (including a follow-up, which only uses tier 1) is a single GET.

Strict on the two ways to serve a wrong answer (worse than a miss):
  - Cross-space leaks: the exact key is namespaced by space_key, and KNN is TAG-filtered to it.
  - False semantic matches: threshold 0.97 by default; a miss only costs an LLM call. On top of
    that, a semantic hit must name the same identifiers (words with a digit: part numbers, issue
    keys, versions, dates). Questions that differ only in one of those embed above 0.97, e.g. the
    price of part CM4108064 vs CM4104032, yet have different answers.

Invalidation is event-driven: keys carry a `data_version` derived from vectorization-service's
GET /index/watermark (index progress plus authoritative business-data fingerprints) and from
`code_version()`, a fingerprint of this service's code and prompts. Any indexed change, and any
deploy that changes how answers are made, makes older entries unreachable immediately. The TTL
(600s) only bounds memory.

The caller (routes.py) decides what may be read/written: follow-ups get exact reads only,
attachment-scoped questions skip the cache, and only non-abstaining semantic-search and
multi-agent answers are written.

Testing: fakeredis has no FT.* commands, so the exact tier and eviction are tested with fakeredis,
and the semantic tier with a real redis-stack-server in CI (`TEST_REDIS_STACK_URL`), including
cross-space isolation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.cache.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

SpaceKey = str

# Bumped when the grounding contract changed, again (v4 -> v5) when data_version was added, and
# again (v5 -> v6) when execution_mode (single vs multi) was added to the key/TAG so adaptive Ask
# AI cannot serve a single-CRAG answer when the router chose multi (or the reverse).
# Being in both the index and the keys stops an old, now-unsupported answer from being served after
# a deploy. Old keys just expire. Ordinary code and prompt changes no longer need a bump (see
# code_version); bump this only when the key or index layout changes.
_CACHE_VERSION = "v6"
_INDEX_NAME = f"aicache_vec_idx_{_CACHE_VERSION}"
_VECTOR_KEY_PREFIX = f"aicache:{_CACHE_VERSION}:vec:"
_ALLOWED_EXECUTION_MODES = frozenset({"single", "multi"})


def _execution_mode(mode: str | None) -> str:
    value = (mode or "single").strip().lower()
    if value not in _ALLOWED_EXECUTION_MODES:
        raise ValueError(f"execution_mode must be single|multi, got {mode!r}")
    return value


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


_WORD = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def _identifiers(question: str) -> frozenset[str]:
    """Words with a digit in them, e.g. {"cm4108064", "scrum-12", "2026"}. Latin letters and digits
    only, so in "CM4104032的價格" the part number is still its own word."""
    found = set()
    for word in _WORD.findall(question.lower()):
        if any(ch.isdigit() for ch in word):
            found.add(word)
    return frozenset(found)


def _space_key(space_ids: Sequence[int]) -> SpaceKey:
    # "_" not "," — RediSearch TAG fields treat "," as the default multi-value separator, which would
    # silently turn "space_key:1,3" into two separate tag values instead of one compound key.
    return "_".join(str(s) for s in sorted(space_ids))


def _pack_vector(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


_APP_DIR = Path(__file__).resolve().parent.parent


def code_version(app_dir: Path = _APP_DIR) -> str:
    """Fingerprint of every .py file and prompt template under app/.

    Without it, a fix to the agent or a prompt kept serving the old (wrong) answer until the TTL ran
    out, because only the data was in the key. The same image gives the same value, so replicas still
    share entries.
    """
    digest = hashlib.sha256()
    for path in sorted(app_dir.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".jinja"):
            digest.update(str(path.relative_to(app_dir)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _data_version(watermark: str, code_version: str = "") -> str:
    """Hash the freshness token (and the code version) into a TAG-safe value. The raw token is full of
    `-`, `:` and `+`, which RediSearch's TAG tokenizer splits on, so a literal filter would match the
    wrong thing. 16 hex chars is plenty for a cache discriminator."""
    token = f"{watermark}|{code_version}" if code_version else watermark
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


class RedisSemanticCache:
    def __init__(
        self,
        redis_client,
        embed_client: EmbeddingClient,
        ttl_seconds: float,
        similarity_threshold: float,
        max_entries: int,
        embedding_dim: int,
        code_version: str = "",
    ):
        self._redis = redis_client
        self._embed = embed_client
        self._code_version = code_version
        self._ttl = int(ttl_seconds)
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._embedding_dim = embedding_dim

    async def ensure_index(self) -> None:
        """Create the vector index once, idempotently. Call during app startup (see main.py) — not
        lazily on first request, so index creation isn't a race between concurrent first callers."""
        try:
            await self._redis.ft(_INDEX_NAME).create_index(
                [
                    TagField("space_key"),
                    # Freshness as its own TAG, separate from space_key: "may this caller see it"
                    # and "is it still true" are separate facts, each testable on its own.
                    TagField("data_version"),
                    # single vs multi path — same question can have two valid answers.
                    TagField("execution_mode"),
                    VectorField(
                        "embedding", "HNSW",
                        {"TYPE": "FLOAT32", "DIM": self._embedding_dim, "DISTANCE_METRIC": "COSINE"},
                    ),
                ],
                definition=IndexDefinition(prefix=[_VECTOR_KEY_PREFIX], index_type=IndexType.HASH),
            )
            logger.info("created semantic cache vector index", extra={"index": _INDEX_NAME})
        except Exception as exc:  # noqa: BLE001
            if "already exists" not in str(exc).lower():
                raise

    def _exact_key(
        self, space_key: SpaceKey, norm_q: str, data_version: str, execution_mode: str,
    ) -> str:
        digest = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()[:16]
        return (
            f"aicache:{_CACHE_VERSION}:exact:{space_key}:{data_version}:"
            f"{execution_mode}:{digest}"
        )

    def _vector_key(self, space_key: SpaceKey, entry_id: str) -> str:
        return f"{_VECTOR_KEY_PREFIX}{space_key}:{entry_id}"

    def _index_key(self, space_key: SpaceKey) -> str:
        # FIFO eviction queue (Redis List) bounding how many entries per space the cache keeps —
        # unrelated to the FT.SEARCH vector index despite the similar name.
        return f"aicache:{_CACHE_VERSION}:fifo:{space_key}"

    async def get(
        self,
        question: str,
        space_ids: Sequence[int],
        watermark: str,
        exact_only: bool = False,
        execution_mode: str = "single",
    ) -> Optional[dict]:
        space_key = _space_key(space_ids)
        norm_q = _normalize(question)
        data_version = _data_version(watermark, self._code_version)
        mode = _execution_mode(execution_mode)

        exact_raw = await self._redis.get(self._exact_key(space_key, norm_q, data_version, mode))
        if exact_raw is not None:
            return json.loads(exact_raw)["response"]

        if exact_only:
            # Follow-up question: exact re-asks are fine, but no semantic match — a paraphrase
            # mid-conversation depends on context the cached entry didn't have.
            return None

        embedding = await self._embed.embed(question)
        query = (
            # Filters must be parenthesised, or the `=>` is a syntax error
            # (only caught against a real redis-stack-server).
            Query(
                f"(@space_key:{{{space_key}}} @data_version:{{{data_version}}} "
                f"@execution_mode:{{{mode}}})"
                "=>[KNN 1 @embedding $vec AS score]"
            )
            .sort_by("score")
            .return_fields("response", "score", "question")
            .dialect(2)
        )
        try:
            result = await self._redis.ft(_INDEX_NAME).search(
                query, query_params={"vec": _pack_vector(embedding)}
            )
        except Exception as exc:  # noqa: BLE001
            # FLUSHDB, a restart without persistence, or eviction can drop the index while the service
            # runs, making every semantic lookup miss. Recreate it and retry once.
            if "no such index" not in str(exc).lower() and "unknown index" not in str(exc).lower():
                raise
            logger.warning("semantic cache index missing; recreating", extra={"index": _INDEX_NAME})
            await self.ensure_index()
            result = await self._redis.ft(_INDEX_NAME).search(
                query, query_params={"vec": _pack_vector(embedding)}
            )
        if not result.docs:
            return None

        best = result.docs[0]
        similarity = 1.0 - float(best.score)  # RediSearch COSINE returns distance, not similarity
        if similarity < self._threshold:
            return None
        cached_question = getattr(best, "question", None)
        if cached_question is None or _identifiers(cached_question) != _identifiers(question):
            logger.info(
                "semantic cache hit rejected: identifiers differ",
                extra={"similarity": round(similarity, 4)},
            )
            return None

        response = json.loads(best.response)
        # Save the hit under the wording actually asked, so the next identical re-ask (and any
        # follow-up, which only uses the exact tier) is a plain GET. Same data_version, so it
        # invalidates together with everything else.
        await self._redis.set(
            self._exact_key(space_key, norm_q, data_version, mode),
            json.dumps({"response": response}),
            ex=self._ttl,
        )
        return response

    async def put(
        self,
        question: str,
        space_ids: Sequence[int],
        response: dict,
        watermark: str,
        execution_mode: str = "single",
    ) -> None:
        space_key = _space_key(space_ids)
        norm_q = _normalize(question)
        data_version = _data_version(watermark, self._code_version)
        mode = _execution_mode(execution_mode)
        embedding = await self._embed.embed(question)
        entry_id = uuid.uuid4().hex
        vector_key = self._vector_key(space_key, entry_id)
        index_key = self._index_key(space_key)

        pipe = self._redis.pipeline()
        pipe.set(
            self._exact_key(space_key, norm_q, data_version, mode),
            json.dumps({"response": response}),
            ex=self._ttl,
        )
        pipe.hset(
            vector_key,
            mapping={
                "space_key": space_key,
                "data_version": data_version,
                "execution_mode": mode,
                # Not indexed; read back on a semantic hit to compare identifiers.
                "question": norm_q,
                "embedding": _pack_vector(embedding),
                "response": json.dumps(response),
            },
        )
        pipe.expire(vector_key, self._ttl)
        pipe.rpush(index_key, entry_id)
        pipe.expire(index_key, self._ttl)
        await pipe.execute()

        await self._evict_overflow(space_key, index_key)

    async def _evict_overflow(self, space_key: SpaceKey, index_key: str) -> None:
        length = await self._redis.llen(index_key)
        overflow = length - self._max_entries
        if overflow <= 0:
            return
        # FIFO: the oldest entries are at the head (put() only ever RPUSHes).
        old_ids = await self._redis.lpop(index_key, overflow)
        if not old_ids:
            return
        for raw_id in old_ids:
            entry_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            # Deleting the HASH key also removes it from the FT index — RediSearch keeps hash-backed
            # indexes in sync with the keyspace automatically, no separate "remove from index" call.
            await self._redis.delete(self._vector_key(space_key, entry_id))


class NoopCache:
    """Used when AI_CACHE_ENABLED=false, and as the hermetic default in tests."""

    async def ensure_index(self) -> None:
        return None

    async def get(
        self,
        question: str,
        space_ids: Sequence[int],
        watermark: str,
        exact_only: bool = False,
        execution_mode: str = "single",
    ) -> Optional[dict]:
        return None

    async def put(
        self,
        question: str,
        space_ids: Sequence[int],
        response: dict,
        watermark: str,
        execution_mode: str = "single",
    ) -> None:
        return None


def build_cache(settings, redis_client, embed_client: EmbeddingClient):
    if not settings.cache_enabled:
        return NoopCache()
    return RedisSemanticCache(
        redis_client=redis_client,
        embed_client=embed_client,
        ttl_seconds=settings.cache_ttl_seconds,
        similarity_threshold=settings.cache_similarity_threshold,
        max_entries=settings.cache_max_entries,
        embedding_dim=settings.cache_embedding_dim,
        code_version=code_version(),
    )
