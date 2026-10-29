"""Signed receipts bind submitted feedback to a real answer without trusting browser snapshots.

Only submitted feedback is persisted by jira-backend. A receipt belongs to its caller and
expires in seven days. It must never be written into the shared answer cache.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from pathlib import Path

DOMAIN = b"ask-ai-feedback-v1:"
MAX_RECEIPT_BYTES = 1_000_000


def attach_feedback_receipt(request, ask_request, response, watermark):
    secret = getattr(getattr(request.app.state, "settings", None), "internal_service_token", "")
    user_id = request.headers.get("x-user-id", "")
    if not secret or not user_id.isdigit() or int(user_id) <= 0:
        return response
    now = int(time.time())
    answer_id = str(uuid.uuid4())
    templates = Path(__file__).parent / "agent" / "templates"
    snapshot = {
        "schemaVersion": 1, "answerId": answer_id, "userId": int(user_id),
        "issuedAt": now, "expiresAt": now + 7 * 86400,
        "request": ask_request.model_dump(by_alias=True),
        "response": response.model_dump(by_alias=True, exclude={"answer_id", "feedback_receipt"}),
        "dataVersion": watermark,
        # A cache hit's serving deployment may differ from its original generation.
        "servingPromptVersions": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            for p in sorted(templates.glob("*.jinja"))
        },
    }
    payload = base64.urlsafe_b64encode(
        json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode()
    ).rstrip(b"=")
    signature = hmac.new(secret.encode(), DOMAIN + payload, hashlib.sha256).hexdigest()
    receipt = payload.decode() + "." + signature
    if len(receipt) > MAX_RECEIPT_BYTES:
        return response
    return response.model_copy(update={"answer_id": answer_id, "feedback_receipt": receipt})
