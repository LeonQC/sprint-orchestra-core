"""Semantic cache tests — fakeredis plus an optional real Redis Stack integration test.

fakeredis does not implement RediSearch's `FT.*` commands (confirmed directly — `FT.CREATE` raises
"unknown command"), so the semantic-tier KNN search itself (`RedisSemanticCache.get()`'s vector
search path) cannot have hermetic coverage here. CI sets ``TEST_REDIS_STACK_URL`` and executes the
real integration test below against `redis-stack-server` (including COSINE
distance-vs-similarity semantics, and that a `space_key` TAG filter genuinely returns zero cross-space
hits). What is hermetically tested with fakeredis: the exact-match tier (pure `GET`/`SET`, no `FT.*` calls)
and the FIFO eviction bookkeeping (`RPUSH`/`LPOP`/`DEL`, also no `FT.*` calls) — both real code paths
`put()` and `get()` exercise on every call regardless of whether the semantic tier ever fires.
"""
import asyncio
import os
import uuid

import fakeredis
import pytest
import redis.asyncio as redis

from app.cache.semantic_cache import (
    NoopCache,
    RedisSemanticCache,
    _data_version,
    _identifiers,
    build_cache,
    code_version,
)

# A freshness token shaped like a real one (vectorization-service returns
# "<max(updated_at) ISO>:<row count>"), and a second one standing in for "something in this
# space was indexed since". The cache only ever compares them, never parses them.
WATERMARK = "2026-09-03T10:00:00+00:00:100"
NEWER_WATERMARK = "2026-09-03T10:05:00+00:00:101"


class FakeEmbedClient:
    """Maps exact question text to a scripted vector — makes cosine similarity fully controllable."""

    def __init__(self, vectors: dict):
        self._vectors = vectors
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return self._vectors[text]


@pytest.fixture
def redis_client():
    client = fakeredis.FakeAsyncRedis()
    yield client


def _cache(redis_client, vectors, ttl=600.0, threshold=0.97, max_entries=500, embedding_dim=2):
    return RedisSemanticCache(
        redis_client=redis_client,
        embed_client=FakeEmbedClient(vectors),
        ttl_seconds=ttl,
        similarity_threshold=threshold,
        max_entries=max_entries,
        embedding_dim=embedding_dim,
    )


async def test_exact_match_hit_skips_embedding_call(redis_client):
    embed = FakeEmbedClient({"How do I reset my password?": [1.0, 0.0]})
    cache = RedisSemanticCache(redis_client, embed, ttl_seconds=600, similarity_threshold=0.97, max_entries=10, embedding_dim=2)

    await cache.put("How do I reset my password?", [1], {"answer": "click forgot password"}, WATERMARK)
    calls_after_put = embed.calls

    # "how do i reset my password?" normalizes (lowercase + whitespace-collapse) to the same exact
    # key the put() call stored, without ever calling embed() again.
    result = await cache.get("how do i reset my password?", [1], WATERMARK)
    assert result == {"answer": "click forgot password"}
    assert embed.calls == calls_after_put  # exact hit never needed an embedding call


async def test_exact_only_returns_an_exact_hit_but_never_reaches_the_semantic_tier(redis_client):
    """A follow-up question (`exact_only=True`, see app/api/routes.py) may be served from the exact
    tier but must not trigger the embedding call / KNN search the semantic tier needs."""
    embed = FakeEmbedClient({"How many bugs are open?": [1.0, 0.0]})
    cache = RedisSemanticCache(redis_client, embed, ttl_seconds=600, similarity_threshold=0.97, max_entries=10, embedding_dim=2)

    await cache.put("How many bugs are open?", [1], {"answer": "7"}, WATERMARK)
    calls_after_put = embed.calls

    # Exact re-ask (post-normalization): served from the exact tier, no embedding call.
    assert await cache.get("how many bugs are open?", [1], WATERMARK, exact_only=True) == {"answer": "7"}
    assert embed.calls == calls_after_put

    # A different question with exact_only set: the semantic tier would embed and KNN-search, but
    # exact_only short-circuits before that, so it is a clean miss with no embedding call.
    assert await cache.get("what is blocking the release?", [1], WATERMARK, exact_only=True) is None
    assert embed.calls == calls_after_put


async def test_exact_match_isolates_execution_mode(redis_client):
    """Adaptive Ask AI: a multi answer must not be served when the router chose single."""
    cache = _cache(redis_client, {"compare checkout risks": [1.0, 0.0]})
    await cache.put(
        "compare checkout risks", [1], {"answer": "multi answer"}, WATERMARK,
        execution_mode="multi",
    )
    # exact_only: fakeredis has no FT.SEARCH; this asserts the exact-key namespace alone.
    assert await cache.get(
        "compare checkout risks", [1], WATERMARK, exact_only=True, execution_mode="single",
    ) is None
    assert await cache.get(
        "compare checkout risks", [1], WATERMARK, exact_only=True, execution_mode="multi",
    ) == {"answer": "multi answer"}


async def test_exact_match_respects_space_ids_isolation(redis_client):
    """Security-critical: identical question text/embedding must not cross a space_ids boundary."""
    cache = _cache(redis_client, {"is checkout broken": [1.0, 0.0]})
    await cache.put("is checkout broken", [1], {"answer": "yes, ATLAS-3"}, WATERMARK)

    # Inspect the exact tier directly instead of falling through to FT.SEARCH, which fakeredis does
    # not implement. The real-Redis integration test below owns semantic-tier isolation coverage.
    normalized = "is checkout broken"
    version = _data_version(WATERMARK)
    assert await redis_client.get(cache._exact_key("1", normalized, version, "single")) is not None
    assert await redis_client.get(cache._exact_key("2", normalized, version, "single")) is None


async def test_exact_tier_entry_is_unreachable_after_the_index_changes(redis_client):
    """The freshness half of the key, hermetically — the reason /ask may consult this cache at all.

    Same question, same space, same stored answer; only the watermark moved, which is what an ingest
    of *anything* in this space does. The entry the earlier watermark stored is no longer addressable
    under the new one.

    Asserted against the exact tier directly rather than through `get()`, for the same reason
    `test_exact_match_respects_space_ids_isolation` does: a `get()` that misses the exact tier falls
    through to `FT.SEARCH`, which fakeredis does not implement. The real-Redis integration test below
    covers the identical property on the semantic tier.
    """
    cache = _cache(redis_client, {"is checkout broken": [1.0, 0.0]})
    await cache.put("is checkout broken", [1], {"answer": "yes, ATLAS-3"}, WATERMARK)
    normalized = "is checkout broken"

    assert await redis_client.get(cache._exact_key("1", normalized, _data_version(WATERMARK), "single")) is not None
    assert await redis_client.get(cache._exact_key("1", normalized, _data_version(NEWER_WATERMARK), "single")) is None


async def test_freshness_and_permission_are_independent_dimensions_of_the_key(redis_client):
    """Neither TAG subsumes the other: changing either one alone must miss.

    Worth pinning separately because a cheaper implementation — fusing space and version into one
    compound key — would pass every other test here while making it impossible to assert that the
    permission boundary still holds when only the version matches, and vice versa.
    """
    cache = _cache(redis_client, {"is checkout broken": [1.0, 0.0]})
    await cache.put("is checkout broken", [1], {"answer": "yes, ATLAS-3"}, WATERMARK)
    normalized = "is checkout broken"

    same_space_new_version = cache._exact_key("1", normalized, _data_version(NEWER_WATERMARK), "single")
    other_space_same_version = cache._exact_key("2", normalized, _data_version(WATERMARK), "single")
    assert await redis_client.get(same_space_new_version) is None
    assert await redis_client.get(other_space_same_version) is None


async def test_real_redis_stack_semantic_hit_and_cross_space_isolation():
    """Exercises the native FT.SEARCH path CI's Redis Stack service is specifically for."""
    redis_url = os.getenv("TEST_REDIS_STACK_URL")
    if not redis_url:
        pytest.skip("TEST_REDIS_STACK_URL is not configured")

    client = redis.from_url(redis_url)
    # One run id for both: different ids would be different identifiers, and never hit.
    run_id = uuid.uuid4().hex
    question = f"checkout failure {run_id}"
    paraphrase = f"why checkout failed {run_id}"
    vector = [1.0, *([0.0] * 1023)]
    cache = _cache(
        client,
        {question: vector, paraphrase: vector},
        ttl=60,
        threshold=0.97,
        max_entries=10,
        embedding_dim=1024,
    )
    space_ids = [987654321]
    other_space_ids = [987654322]
    await cache.ensure_index()
    try:
        await cache.put(question, space_ids, {"answer": "connection pool exhaustion"}, WATERMARK)
        # RediSearch indexing can be asynchronous; poll briefly instead of making the test timing-
        # sensitive on a newly started CI container.
        hit = None
        for _ in range(20):
            hit = await cache.get(paraphrase, space_ids, WATERMARK)
            if hit is not None:
                break
            await asyncio.sleep(0.05)
        assert hit == {"answer": "connection pool exhaustion"}
        assert await cache.get(paraphrase, other_space_ids, WATERMARK) is None

        # The semantic hit was promoted: an exact-tier entry now exists for the phrasing that was
        # actually asked (the paraphrase), so an identical re-ask — including the exact-only path a
        # follow-up question takes — is served without touching the embedder or FT.SEARCH again.
        promoted_key = cache._exact_key(
            "987654321", " ".join(paraphrase.lower().split()), _data_version(WATERMARK), "single",
        )
        assert await client.get(promoted_key) is not None
        embed_calls_before = cache._embed.calls
        assert await cache.get(paraphrase, space_ids, WATERMARK, exact_only=True) == {
            "answer": "connection pool exhaustion"
        }
        assert cache._embed.calls == embed_calls_before

        # The freshness half of the same key. Nothing about the corpus or the question changed here;
        # only the token did, exactly as an ingest of any document in this space would change it —
        # and the previously-hitting paraphrase stops being reachable.
        assert await cache.get(paraphrase, space_ids, NEWER_WATERMARK) is None

        # A dropped index (FLUSHDB / Redis restart / eviction) is recreated in place on the next
        # lookup instead of silently degrading every semantic hit to a miss until the next deploy.
        await client.execute_command("FT.DROPINDEX", "aicache_vec_idx_v6")
        await cache.put(question, space_ids, {"answer": "connection pool exhaustion"}, WATERMARK)
        healed = None
        for _ in range(20):
            healed = await cache.get(paraphrase, space_ids, WATERMARK)
            if healed is not None:
                break
            await asyncio.sleep(0.05)
        assert healed == {"answer": "connection pool exhaustion"}
    finally:
        space_key = "987654321"
        entry_ids = await client.lrange(cache._index_key(space_key), 0, -1)
        keys = [
            cache._exact_key(space_key, " ".join(question.lower().split()), _data_version(WATERMARK), "single"),
            cache._exact_key(space_key, " ".join(paraphrase.lower().split()), _data_version(WATERMARK), "single"),
            cache._index_key(space_key),
            *(cache._vector_key(space_key, raw.decode() if isinstance(raw, bytes) else raw) for raw in entry_ids),
        ]
        if keys:
            await client.delete(*keys)
        await client.aclose()


def test_identifiers_are_the_words_with_a_digit():
    assert _identifiers("What is the price of Compute Module 4 part number CM4108064?") == {
        "4", "cm4108064",
    }
    assert _identifiers("Who owns SCRUM-12 in sprint 3.1?") == {"scrum-12", "3.1"}
    assert _identifiers("CM4104032的價格是多少") == {"cm4104032"}
    assert _identifiers("why did checkout fail?") == frozenset()


async def test_real_redis_stack_semantic_hit_needs_the_same_identifiers():
    """Two questions that differ only in a part number embed above 0.97 (seen live with
    CM4108064 vs CM4104032) but have different answers."""
    redis_url = os.getenv("TEST_REDIS_STACK_URL")
    if not redis_url:
        pytest.skip("TEST_REDIS_STACK_URL is not configured")

    client = redis.from_url(redis_url)
    asked = "What is the price of part CM4108064?"
    other_part = "What is the price of part CM4104032?"
    same_part = "How much is part CM4108064?"
    vector = [1.0, *([0.0] * 1023)]
    cache = _cache(
        client,
        {asked: vector, other_part: vector, same_part: vector},
        ttl=60,
        threshold=0.97,
        max_entries=10,
        embedding_dim=1024,
    )
    space_ids = [987654323]
    space_key = "987654323"
    await cache.ensure_index()
    try:
        await cache.put(asked, space_ids, {"answer": "$195"}, WATERMARK)
        hit = None
        for _ in range(20):
            hit = await cache.get(same_part, space_ids, WATERMARK)
            if hit is not None:
                break
            await asyncio.sleep(0.05)
        assert hit == {"answer": "$195"}

        assert await cache.get(other_part, space_ids, WATERMARK) is None
    finally:
        entry_ids = await client.lrange(cache._index_key(space_key), 0, -1)
        version = _data_version(WATERMARK)
        keys = [
            cache._exact_key(space_key, " ".join(q.lower().split()), version, "single")
            for q in (asked, other_part, same_part)
        ]
        keys.append(cache._index_key(space_key))
        for raw in entry_ids:
            keys.append(cache._vector_key(space_key, raw.decode() if isinstance(raw, bytes) else raw))
        await client.delete(*keys)
        await client.aclose()


async def test_disabled_cache_never_stores_or_returns_anything():
    cache = NoopCache()
    await cache.put("q", [1], {"answer": "x"}, WATERMARK)  # must not raise with nothing configured
    assert await cache.get("q", [1], WATERMARK) is None


async def test_max_entries_evicts_oldest(redis_client):
    # Exercises put()'s FIFO bookkeeping only (RPUSH/LPOP/DEL) — no FT.SEARCH involved, so this is
    # fully hermetic despite living in the same class as the semantic-search code.
    vectors = {f"q{i}": [float(i), 0.0] for i in range(5)}
    cache = _cache(redis_client, vectors, max_entries=2)
    for i in range(3):
        await cache.put(f"q{i}", [1], {"answer": str(i)}, WATERMARK)

    surviving_ids = await redis_client.lrange(cache._index_key("1"), 0, -1)
    assert len(surviving_ids) == 2  # q0 was evicted (FIFO) when q2 pushed past max_entries=2


class _Settings:
    def __init__(self, cache_enabled):
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = 600.0
        self.cache_similarity_threshold = 0.97
        self.cache_max_entries = 500
        self.cache_embedding_dim = 2
        self.cache_watermark_ttl_seconds = 0.0


def test_build_cache_returns_noop_when_disabled(redis_client):
    cache = build_cache(_Settings(cache_enabled=False), redis_client, FakeEmbedClient({}))
    assert isinstance(cache, NoopCache)


def test_build_cache_returns_redis_backed_when_enabled(redis_client):
    cache = build_cache(_Settings(cache_enabled=True), redis_client, FakeEmbedClient({}))
    assert isinstance(cache, RedisSemanticCache)


async def test_a_new_code_version_cannot_read_answers_written_by_the_old_code(redis_client):
    """A fix to the agent or a prompt must take effect at once, not after the TTL (seen live)."""
    embed = FakeEmbedClient({"how many bugs are open?": [1.0, 0.0]})
    old = RedisSemanticCache(redis_client, embed, ttl_seconds=600, similarity_threshold=0.97,
                             max_entries=10, embedding_dim=2, code_version="before-fix")
    new = RedisSemanticCache(redis_client, embed, ttl_seconds=600, similarity_threshold=0.97,
                             max_entries=10, embedding_dim=2, code_version="after-fix")

    await old.put("how many bugs are open?", [1], {"answer": "wrong"}, WATERMARK)

    assert await old.get("how many bugs are open?", [1], WATERMARK, exact_only=True) == {"answer": "wrong"}
    assert await new.get("how many bugs are open?", [1], WATERMARK, exact_only=True) is None


def test_code_version_changes_with_any_code_or_prompt_file(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "loop.py").write_text("x = 1\n")
    (tmp_path / "agent" / "answer.jinja").write_text("Answer briefly.")
    (tmp_path / "notes.txt").write_text("ignored")
    first = code_version(tmp_path)

    assert code_version(tmp_path) == first, "same files, same version (replicas share entries)"
    (tmp_path / "notes.txt").write_text("still ignored")
    assert code_version(tmp_path) == first
    (tmp_path / "agent" / "answer.jinja").write_text("Answer in detail.")
    assert code_version(tmp_path) != first
