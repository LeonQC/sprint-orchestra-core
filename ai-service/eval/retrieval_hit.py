"""Retrieval-only check on the real AtlasCart corpus: is the issue the answer needs in the top k?

The 50-question RAGAS set carries `required_issue_keys` for its 32 semantic questions. This script
sends each question straight to vectorization-service `/search` (the same hybrid + rerank call the
agent makes, with the agent's query rewriting left out), collapses the ranked chunks to issue keys,
and scores them like a classic retrieval benchmark:

  * hit@k     -- any required key among the first k distinct issues (k = 1, 3, 5)
  * recall@k  -- share of the required keys among the first k distinct issues (k = 5, 10)
  * mrr       -- 1 / rank of the first required key

No LLM is involved: the labels are issue keys and the hits carry issue keys, so a question either
puts its issue in the top 5 or it does not. The whole set runs in seconds and costs only the query
embeddings, which is what makes it fit to run on every change, not nightly.

hit@5 is the number that matters: production `retrieval_top_k` is 5, so a miss at 5 means the agent
never saw the right issue, whatever the generation step did afterwards.

    python -m eval.retrieval_hit --out /tmp/retrieval_hit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx

from eval.atlascart_ids import atlascart_space_id
from eval.ragas_eval import CASES, RagasCase

KS_HIT = (1, 3, 5)
KS_RECALL = (5, 10)


@dataclass
class CaseScore:
    question: str
    category: str
    required_issue_keys: List[str]
    ranked_issue_keys: List[str]
    hit: Dict[str, bool]
    recall: Dict[str, float]
    reciprocal_rank: float
    error: str = ""


def distinct_issue_keys(hits: Sequence[dict]) -> List[str]:
    """Chunks in rank order collapsed to issue keys, first appearance wins."""
    seen: List[str] = []
    for hit in hits:
        key = hit.get("issue_key")
        if key and key not in seen:
            seen.append(key)
    return seen


def score(case: RagasCase, ranked_issue_keys: List[str]) -> CaseScore:
    required = set(case.required_issue_keys)
    hit = {f"hit@{k}": bool(required & set(ranked_issue_keys[:k])) for k in KS_HIT}
    recall = {f"recall@{k}": len(required & set(ranked_issue_keys[:k])) / len(required) for k in KS_RECALL}
    rr = 0.0
    for rank, key in enumerate(ranked_issue_keys, start=1):
        if key in required:
            rr = 1.0 / rank
            break
    return CaseScore(case.question, case.category, sorted(required), ranked_issue_keys, hit, recall, rr)


def search(client: httpx.Client, url: str, question: str, space_id: int, limit: int, mode: str) -> dict:
    response = client.post(
        f"{url}/search", json={"query": question, "space_ids": [space_id], "limit": limit, "mode": mode}
    )
    response.raise_for_status()
    return response.json()


def summarize(scores: List[CaseScore], meta: dict) -> dict:
    def mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def means(subset: List[CaseScore]) -> dict:
        out = {f"hit@{k}": mean([1.0 if s.hit[f"hit@{k}"] else 0.0 for s in subset]) for k in KS_HIT}
        out.update({f"recall@{k}": mean([s.recall[f"recall@{k}"] for s in subset]) for k in KS_RECALL})
        out["mrr"] = mean([s.reciprocal_rank for s in subset])
        return out

    scored = [s for s in scores if not s.error]
    by_category: Dict[str, dict] = {}
    for category in sorted({s.category for s in scored}):
        subset = [s for s in scored if s.category == category]
        by_category[category] = {"n": len(subset), **means(subset)}
    misses = [
        {"question": s.question, "required": s.required_issue_keys, "top5": s.ranked_issue_keys[:5]}
        for s in scored
        if not s.hit["hit@5"]
    ]
    return {
        **meta,
        "n": len(scored),
        "errors": [s.error for s in scores if s.error],
        "means": means(scored),
        "by_category": by_category,
        "missed_at_5": misses,
        "cases": [asdict(s) for s in scores],
    }


def run(url: str, space_id: int, limit: int, mode: str, cases: Sequence[RagasCase] = CASES) -> dict:
    labelled = [c for c in cases if c.required_issue_keys]
    scores: List[CaseScore] = []
    served_mode, reranked = None, None
    with httpx.Client(timeout=60.0) as client:
        for case in labelled:
            try:
                body = search(client, url, case.question, space_id, limit, mode)
            except Exception as exc:  # noqa: BLE001 - one failed search must not hide the others
                scores.append(CaseScore(case.question, case.category, list(case.required_issue_keys), [], {}, {}, 0.0, str(exc)))
                continue
            served_mode, reranked = body.get("mode"), body.get("reranked")
            scores.append(score(case, distinct_issue_keys(body.get("hits", []))))
    meta = {"space_id": space_id, "limit": limit, "requested_mode": mode, "served_mode": served_mode, "reranked": reranked}
    return summarize(scores, meta)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100", help="vectorization-service")
    parser.add_argument("--space-id", type=int, default=None, help="defaults to the current AtlasCart space")
    parser.add_argument("--limit", type=int, default=10, help="chunks to fetch; recall@10 needs 10")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "retrieval_hit.json")
    args = parser.parse_args(argv)

    space_id = args.space_id if args.space_id is not None else atlascart_space_id()
    summary = run(args.url, space_id, args.limit, args.mode)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    m = summary["means"]
    print(f"{summary['n']} labelled questions, space {space_id}, {summary['served_mode']} reranked={summary['reranked']}")
    print(f"hit@1 {m['hit@1']:.3f}  hit@3 {m['hit@3']:.3f}  hit@5 {m['hit@5']:.3f}  recall@5 {m['recall@5']:.3f}  recall@10 {m['recall@10']:.3f}  mrr {m['mrr']:.3f}")
    for category, row in summary["by_category"].items():
        print(f"  {category:28s} n={row['n']:2d}  hit@5 {row['hit@5']:.3f}  recall@10 {row['recall@10']:.3f}  mrr {row['mrr']:.3f}")
    for miss in summary["missed_at_5"]:
        print(f"  MISS@5 {miss['required']} not in {miss['top5']}  <- {miss['question'][:70]}")
    if summary["errors"]:
        print(f"  {len(summary['errors'])} search error(s): {summary['errors'][0]}")
    print(f"-> {args.out}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
