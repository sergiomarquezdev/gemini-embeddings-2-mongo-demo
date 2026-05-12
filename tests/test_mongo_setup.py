from pymongo.errors import DuplicateKeyError
import pytest
from mongo_setup import init_indexes, healthcheck


def test_init_indexes_creates_dedup_unique(test_db):
    init_indexes(test_db, vector_dim=1536, collection_name="documents")
    idx = list(test_db["documents"].list_indexes())
    by_name = {i["name"]: i for i in idx}
    assert "dedup_idx" in by_name
    assert by_name["dedup_idx"]["unique"] is True


def test_unique_idx_blocks_concurrent_duplicate(test_db):
    init_indexes(test_db, vector_dim=1536, collection_name="documents")
    coll = test_db["documents"]
    base = {
        "content_hash": "sha256:abc",
        "embedding_model": "gemini-embedding-2",
        "embedding_dim": 1536,
        "chunk_index": 0,
    }
    coll.insert_one(base)
    with pytest.raises(DuplicateKeyError):
        coll.insert_one(dict(base))


def test_healthcheck_returns_ok_when_mongo_up(test_db, mongo_client):
    init_indexes(test_db, vector_dim=1536, collection_name="documents")
    h = healthcheck(test_db, collection_name="documents")
    assert h["mongo"] == "ok"
    assert h["dedup_index"] == "ready"
