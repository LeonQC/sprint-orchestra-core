from app.ingest.attachment_fetcher import _s3_client


def test_local_settings_use_minio_with_its_keys():
    _s3_client.cache_clear()
    client = _s3_client("http://localhost:9000", "minioadmin", "minioadmin", "us-east-1")
    assert client.meta.endpoint_url == "http://localhost:9000"
    assert client._request_signer._credentials.access_key == "minioadmin"


def test_blank_endpoint_and_keys_use_real_s3_and_the_default_credential_chain(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "from-the-task-role")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    _s3_client.cache_clear()
    client = _s3_client("", "", "", "us-east-1")
    assert client.meta.endpoint_url.startswith("https://s3.")
    assert client._request_signer._credentials.access_key == "from-the-task-role"
    _s3_client.cache_clear()
