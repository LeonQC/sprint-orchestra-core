"""A scope-aware catalog of actual worker evidence and validated claim-to-source links."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from app.agent.crag_loop import _ISSUE_KEY_RE
from app.agent.retrieval_tool import RetrievedChunk


@dataclass
class Evidence:
    id: str
    space_ids: list[int]
    content: str
    source_keys: set[str]
    chunk: RetrievedChunk | None = None
    workers: list[str] = field(default_factory=list)
    # Display name of the scope ("PRM"). The model sees this instead of the numeric space ids.
    workspace: str = ""

    def payload(self):
        if self.chunk:
            source = asdict(self.chunk)
            for field_name in ("content", "score", "retrievers"):
                source.pop(field_name, None)
        else:
            source = {"kind": "structured_tool_result"}

        payload = {
            "id": self.id,
            "content": self.content,
            "source": source,
            "workers": self.workers,
        }
        # Only set when several workspaces are compared; see MultiAgentRag._plan.
        if self.workspace:
            payload["workspace"] = self.workspace
        return payload


def catalog(reports: list[dict]) -> list[Evidence]:
    """Merge exact duplicates, while retaining distinct pages, positions, text and scopes."""
    entries: dict[tuple, Evidence] = {}
    for report in reports:
        answer = report["result"]
        if answer is None or answer.needs_clarification:
            continue

        scope = tuple(report["space_ids"])
        workspace = report.get("workspace", "")

        # A model's refusal is not evidence, but successful structured tool receipts still are.
        # In particular, zero matches can answer a count even if a worker mistakenly abstained.
        if answer.abstained:
            citation_chunks = ()
        else:
            citation_chunks = answer.citations

        for chunk in citation_chunks:
            identity = (
                "chunk",
                scope,
                chunk.issue_key.upper(),
                chunk.chunk_type,
                chunk.source_id,
                chunk.page_number,
                json.dumps(chunk.provenance, sort_keys=True),
                " ".join(chunk.content.split()),
            )
            if identity not in entries:
                entries[identity] = Evidence(
                    id=f"E{len(entries) + 1}",
                    space_ids=list(scope),
                    content=chunk.content,
                    source_keys={chunk.issue_key.upper()},
                    chunk=chunk,
                    workspace=workspace,
                )

            entry = entries[identity]
            worker_id = report["id"]
            if worker_id not in entry.workers:
                entry.workers.append(worker_id)

        for text in answer.structured_evidence:
            identity = ("structured", scope, " ".join(text.split()))
            if identity not in entries:
                # CRAG formats this text from tool results; it is not model-written prose.
                mentioned_keys = set()
                for key in _ISSUE_KEY_RE.findall(text):
                    mentioned_keys.add(key.upper())
                actual_keys = set(report["source_keys"])

                entries[identity] = Evidence(
                    id=f"E{len(entries) + 1}",
                    space_ids=list(scope),
                    content=text,
                    source_keys=mentioned_keys & actual_keys,
                    workspace=workspace,
                )

            entry = entries[identity]
            worker_id = report["id"]
            if worker_id not in entry.workers:
                entry.workers.append(worker_id)

    return list(entries.values())


def _validate_claim(claim: dict, evidence_by_id: dict[str, Evidence]) -> tuple[str, list[str]]:
    if not isinstance(claim, dict) or set(claim) != {"text", "evidence_ids"}:
        raise ValueError("invalid claim shape")

    text = claim["text"]
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError("invalid claim text")

    evidence_ids = claim["evidence_ids"]
    if not isinstance(evidence_ids, list) or not evidence_ids or len(evidence_ids) > 12:
        raise ValueError("claim cites unknown or missing evidence")

    unique_ids = []
    for evidence_id in evidence_ids:
        if not isinstance(evidence_id, str) or evidence_id not in evidence_by_id:
            raise ValueError("claim cites unknown or missing evidence")
        if evidence_id not in unique_ids:
            unique_ids.append(evidence_id)

    return text.strip(), unique_ids


def _validate_source_citations(text: str, evidence_ids: list[str], evidence_by_id: dict[str, Evidence]):
    allowed_keys = set()
    for evidence_id in evidence_ids:
        allowed_keys.update(evidence_by_id[evidence_id].source_keys)

    for group in re.findall(r"\(([^()]*)\)", text):
        mentioned_keys = set()
        for key in _ISSUE_KEY_RE.findall(group):
            mentioned_keys.add(key.upper())
        remainder = _ISSUE_KEY_RE.sub("", group).strip(" ,;\t\n")

        if mentioned_keys and not remainder and not mentioned_keys <= allowed_keys:
            raise ValueError("model supplied an unsupported source citation")


def render_claims(payload: dict, evidence: list[Evidence]) -> tuple[str, list[str], list[dict]]:
    """No model-controlled evidence IDs, scopes or source identities can escape this validation."""
    if set(payload) != {"claims", "limitations"}:
        raise ValueError("synthesis must supply claims and limitations only")

    claims = payload["claims"]
    limitations = payload["limitations"]
    if not isinstance(claims, list) or len(claims) > 40 or not isinstance(limitations, list):
        raise ValueError("invalid synthesis shape")
    for limitation in limitations:
        if not isinstance(limitation, str) or not limitation.strip() or len(limitation) > 2000:
            raise ValueError("invalid coverage limitation")

    evidence_by_id = {}
    for entry in evidence:
        evidence_by_id[entry.id] = entry

    selected_ids = []
    claim_links = []
    paragraphs = []
    for claim in claims:
        text, evidence_ids = _validate_claim(claim, evidence_by_id)
        _validate_source_citations(text, evidence_ids, evidence_by_id)

        citation_keys = []
        for evidence_id in evidence_ids:
            chunk = evidence_by_id[evidence_id].chunk
            if chunk and chunk.issue_key not in citation_keys:
                citation_keys.append(chunk.issue_key)

        paragraph = text
        citation = "(" + ", ".join(citation_keys) + ")"
        # The model sometimes writes the same citation itself; don't print "(ATC-44). (ATC-44)".
        if citation_keys and not text.rstrip(" .").endswith(citation):
            paragraph += " " + citation
        paragraphs.append(paragraph)
        claim_links.append({"text": text, "evidence_ids": evidence_ids})

        for evidence_id in evidence_ids:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)

    formatted_paragraphs = []
    for paragraph in paragraphs:
        if len(paragraphs) > 1:
            paragraph = "- " + paragraph
        formatted_paragraphs.append(paragraph)

    answer_text = "\n\n".join(formatted_paragraphs)
    if answer_text and limitations:
        answer_text += "\n\n" + "\n".join(limitations)
    return answer_text, selected_ids, claim_links
