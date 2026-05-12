"""MongoDB setup: connection helpers, index creation (vector + dedup), healthcheck."""
from __future__ import annotations

from pymongo.database import Database
from pymongo.operations import SearchIndexModel


def init_indexes(db: Database, *, vector_dim: int, collection_name: str = "documents") -> None:
    coll = db[collection_name]

    # Dedup B-tree unique index — race protection for concurrent uploads
    coll.create_index(
        [("content_hash", 1), ("embedding_model", 1), ("embedding_dim", 1), ("chunk_index", 1)],
        name="dedup_idx",
        unique=True,
    )

    # Atlas Search vector index
    try:
        existing = {ix.get("name") for ix in coll.list_search_indexes()}
        if "vector_index" not in existing:
            model = SearchIndexModel(
                definition={
                    "fields": [
                        {"type": "vector", "path": "vector",
                         "numDimensions": vector_dim, "similarity": "cosine"},
                        {"type": "filter", "path": "modality"},
                        {"type": "filter", "path": "parent_doc_id"},
                        {"type": "filter", "path": "status"},
                    ]
                },
                name="vector_index",
                type="vectorSearch",
            )
            coll.create_search_index(model=model)
    except Exception:
        pass


def healthcheck(db: Database, *, collection_name: str = "documents") -> dict:
    out = {"mongo": "down", "dedup_index": "missing", "vector_index": "missing"}
    try:
        db.client.admin.command("ping")
        out["mongo"] = "ok"
    except Exception:
        return out
    coll = db[collection_name]
    if any(i["name"] == "dedup_idx" for i in coll.list_indexes()):
        out["dedup_index"] = "ready"
    try:
        if any(i.get("name") == "vector_index" for i in coll.list_search_indexes()):
            out["vector_index"] = "ready"
    except Exception:
        out["vector_index"] = "unsupported"
    return out
