"""Shared fixtures for the test suite.

Mongo: real container at localhost:27017 (started via `docker compose up -d mongo`).
Vertex: mocked.
"""
import uuid
from unittest.mock import MagicMock
import pytest
from pymongo import MongoClient


MONGO_URI = "mongodb://localhost:27017/?directConnection=true"


@pytest.fixture
def test_db_name():
    return f"test_emb_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mongo_client():
    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    c.admin.command("ping")  # fail fast if mongo down
    yield c
    c.close()


@pytest.fixture
def test_db(mongo_client, test_db_name):
    db = mongo_client[test_db_name]
    yield db
    mongo_client.drop_database(test_db_name)


@pytest.fixture
def fake_vector():
    """1536-dim fake embedding (random-ish but deterministic per call)."""
    import random
    rng = random.Random(42)
    return [rng.uniform(-1, 1) for _ in range(1536)]


@pytest.fixture
def mock_vertex_client(fake_vector):
    """Mocked VertexClient that returns the fake_vector for embed_doc/embed_query."""
    from vertex_client import EmbeddingResult
    vc = MagicMock()
    vc.embed_doc.return_value = EmbeddingResult(vector=fake_vector, task_type_used="RETRIEVAL_DOCUMENT", flags={})
    vc.embed_query.return_value = EmbeddingResult(vector=fake_vector, task_type_used="RETRIEVAL_QUERY", flags={})
    vc.count_tokens.return_value = 100
    return vc
