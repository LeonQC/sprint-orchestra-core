"""Real asyncpg boundary regressions for JSONB provenance.

These use small protocol fakes rather than PostgreSQL, but exercise the exact pool initializer and
VectorStore argument mapping that the previous in-memory pipeline tests skipped.
"""
from __future__ import annotations

from dataclasses import asdict

from app.api.routes import SearchHitOut
from app.db import pool as pool_module
from app.db.vector_store import VectorStore, _coerce_provenance
from app.models import Chunk


class FakeConnection:
    def __init__(self):
        self.codecs = []
        self.executemany_args = None

    async def set_type_codec(self, type_name, **kwargs):
        self.codecs.append((type_name, kwargs))

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, *_args):
        return "DELETE 0"

    async def executemany(self, _sql, args):
        self.executemany_args = args


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return self.connection


async def test_pool_registers_native_json_and_jsonb_codecs(monkeypatch):
    connection = FakeConnection()
    vector_registered = []

    async def fake_register_vector(conn):
        vector_registered.append(conn)

    monkeypatch.setattr(pool_module, "register_vector", fake_register_vector)

    await pool_module._init_connection(connection)

    assert [name for name, _ in connection.codecs] == ["json", "jsonb"]
    for _, options in connection.codecs:
        assert options["schema"] == "pg_catalog"
        assert options["format"] == "text"
        assert options["decoder"]('{"source_type":"pdf","page_number":2}') == {
            "source_type": "pdf",
            "page_number": 2,
        }
        assert options["encoder"]({"source_type": "image"}) == '{"source_type": "image"}'
    assert vector_registered == [connection]


async def test_vector_store_writes_provenance_as_native_dict_for_registered_codec():
    connection = FakeConnection()
    store = VectorStore(FakePool(connection))
    provenance = {"source_type": "image", "bbox": [0, 0, 640, 360]}
    chunk = Chunk(
        id="attachment:1#0",
        chunk_type="attachment",
        issue_id=1,
        issue_key="ATC-1",
        space_id=7,
        source_id=1,
        chunk_index=0,
        content="Screenshot text",
        provenance=provenance,
    )

    await store.replace_source("attachment", 1, [chunk], [[0.1, 0.2]])

    assert connection.executemany_args[0][10] == provenance
    assert isinstance(connection.executemany_args[0][10], dict)


def test_provenance_boundary_decodes_legacy_strings_and_rejects_non_objects():
    assert _coerce_provenance('{"source_type":"xlsx","sheet_name":"Q2"}') == {
        "source_type": "xlsx",
        "sheet_name": "Q2",
    }
    assert _coerce_provenance("not json") == {}
    assert _coerce_provenance(["not", "an", "object"]) == {}


class FakeSearchPool:
    """Stands in for a pool whose connections never got the json/jsonb codec — ``fetch`` hands back
    ``provenance`` as the raw string asyncpg would actually return in that case, not a dict."""

    def __init__(self, rows):
        self._rows = rows

    async def fetch(self, _sql, *_args):
        return self._rows


async def test_search_vector_survives_legacy_string_provenance_through_response_serialization():
    """Reproduces the live ATC-432 bug end to end: an attachment (image) row whose ``provenance``
    comes back as a JSON string — the one case text-source chunks never hit, since their provenance
    is always ``{}`` — crashed exactly here. ``search_vector`` -> ``_row_to_hit`` -> the /search
    route's ``SearchHitOut(**asdict(hit))`` is the real call chain that raised
    ``pydantic_core.ValidationError: Input should be a valid dictionary`` and turned into a 502 that
    ai-service reported as "nothing in the knowledge base," even though the chunk was fully ingested
    and embedded. Exercising ``_coerce_provenance`` in isolation (above) would not have caught a
    regression where a future caller reads ``row["provenance"]`` directly instead of through it.
    """
    row = {
        "id": "attachment:5001517#0", "chunk_type": "attachment", "issue_id": 5001731,
        "issue_key": "ATC-432", "space_id": 5000018, "source_id": 5001517,
        "content": "[Attachment: flight-schedule.png — issue ATC-432]\nDEPARTURE: SUNDAY 16 AUG ...",
        "page_number": None,
        "provenance": '{"bbox": [0, 0, 1516, 712], "source_type": "image", "coordinate_system": "pixels"}',
        "score": 0.87,
    }
    store = VectorStore(FakeSearchPool([row]))

    hits = await store.search_vector([0.1, 0.2], [5000018], limit=5)

    assert len(hits) == 1
    assert hits[0].provenance == {
        "bbox": [0, 0, 1516, 712], "source_type": "image", "coordinate_system": "pixels",
    }
    # The literal statement that raised ValidationError in production.
    out = SearchHitOut(**asdict(hits[0]))
    assert out.provenance == hits[0].provenance


async def test_get_issue_attachments_survives_legacy_string_provenance_row():
    """Same regression, other call site: ``get_issue_attachments`` reads ``provenance`` off the
    same ``chunks`` table and hits the identical asyncpg string-vs-dict gap on an un-codec'd pool."""
    from tests.test_issue_attachments import FakePool as AttachmentsFakePool

    rows = [
        {"id": "attachment:5001517#0", "issue_id": 5001731, "issue_key": "ATC-432", "source_id": 5001517,
         "content": "flight schedule screenshot text", "chunk_index": 0, "page_number": None,
         "provenance": '{"bbox": [0, 0, 1516, 712], "source_type": "image"}'},
    ]
    store = VectorStore(AttachmentsFakePool(rows=rows))

    result = await store.get_issue_attachments([5000018], ["ATC-432"])

    assert result.attachments[0].provenance == {"bbox": [0, 0, 1516, 712], "source_type": "image"}
