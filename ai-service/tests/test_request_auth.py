"""RequestAuthMiddleware: every request needs a valid user JWT or the internal service token."""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.auth.request_auth import JwtVerifier, RequestAuthMiddleware

INTERNAL = "test-internal-token"
ISSUER = "jira-agentic-ai"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _FakeJwks:
    def __init__(self, error=None):
        self._error = error

    def get_signing_key_from_jwt(self, token):
        if self._error:
            raise self._error
        return jwt.PyJWK.from_dict(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key(), as_dict=True))


def _client(jwks_error=None) -> TestClient:
    async def echo(request):
        return JSONResponse({
            "user_id": request.headers.get("x-user-id"),
            "username": request.headers.get("x-username"),
        })

    app = Starlette(routes=[Route(p, echo, methods=["GET", "POST", "OPTIONS"]) for p in ("/ask", "/healthz", "/stats")])
    verifier = JwtVerifier("http://jwks.invalid/jwks.json", ISSUER)
    verifier._jwks = _FakeJwks(jwks_error)
    return TestClient(RequestAuthMiddleware(app, verifier=verifier, internal_token=INTERNAL))


def _token(key=KEY, **overrides) -> str:
    claims = {"iss": ISSUER, "sub": "atlascart.maya", "uid": 42, "exp": int(time.time()) + 60}
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256")


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_valid_token_sets_identity_from_claims():
    resp = _client().post("/ask", headers=_bearer(_token()))
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "42", "username": "atlascart.maya"}


def test_forged_identity_headers_are_overwritten():
    headers = {**_bearer(_token()), "X-User-Id": "1", "X-Username": "admin"}
    assert _client().post("/ask", headers=headers).json() == {"user_id": "42", "username": "atlascart.maya"}


def test_forged_identity_without_credentials_is_rejected():
    resp = _client().post("/ask", headers={"X-User-Id": "1", "X-Username": "admin"})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("token", [
    _token(exp=int(time.time()) - 60),
    _token(iss="someone-else"),
    _token(key=OTHER_KEY),
    _token(uid=None),
    _token(uid="alice"),
    "not-a-jwt",
])
def test_bad_tokens_are_rejected(token):
    assert _client().post("/ask", headers=_bearer(token)).status_code == 401


def test_hs256_token_is_rejected():
    # Only RS256 tokens signed by the issuer's key; a shared-secret HS256 token proves nothing.
    token = jwt.encode({"iss": ISSUER, "sub": "x", "uid": 1, "exp": int(time.time()) + 60}, "secret" * 8, algorithm="HS256")
    assert _client().post("/ask", headers=_bearer(token)).status_code == 401


def test_internal_token_keeps_forwarded_identity():
    headers = {"X-Internal-Token": INTERNAL, "X-User-Id": "7", "X-Username": "svc-user"}
    assert _client().post("/ask", headers=headers).json() == {"user_id": "7", "username": "svc-user"}


def test_internal_token_without_user_is_a_trusted_internal_caller():
    assert _client().post("/ask", headers={"X-Internal-Token": INTERNAL}).json() == {"user_id": None, "username": None}


def test_wrong_internal_token_is_rejected_even_with_a_valid_jwt():
    headers = {**_bearer(_token()), "X-Internal-Token": "wrong"}
    assert _client().post("/ask", headers=headers).status_code == 401


def test_stats_is_internal_only():
    assert _client().get("/stats", headers=_bearer(_token())).status_code == 403
    assert _client().get("/stats", headers={"X-Internal-Token": INTERNAL}).status_code == 200


def test_health_and_preflight_need_no_credentials():
    assert _client().get("/healthz").status_code == 200
    assert _client().options("/ask").status_code == 200


def test_unreachable_key_server_is_a_503_not_a_401():
    # The token may be fine; a 401 would blame the user for a jira-backend outage.
    resp = _client(jwks_error=jwt.PyJWKClientConnectionError("down")).post("/ask", headers=_bearer(_token()))
    assert resp.status_code == 503
