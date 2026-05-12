"""Regression: BulkWriteError with code 11000 must be rewritten to DuplicateKeyError
so multi-chunk uploads correctly resolve to already_indexed instead of crashing 500."""
from pymongo.errors import BulkWriteError, DuplicateKeyError
from app import _insert_chunks, _is_dup_key_bulk


class _FakeColl:
    def __init__(self, exc):
        self._exc = exc
    def insert_many(self, docs, ordered=False):
        raise self._exc


def test_bulkwriteerror_with_only_dup_keys_rewrites_to_duplicatekeyerror():
    bwe = BulkWriteError({"writeErrors": [
        {"code": 11000, "errmsg": "dup"},
        {"code": 11000, "errmsg": "dup"},
    ]})
    db = {"documents": _FakeColl(bwe)}
    try:
        _insert_chunks(db, [{"x": 1}, {"x": 2}])
    except DuplicateKeyError:
        pass
    else:
        raise AssertionError("expected DuplicateKeyError")


def test_bulkwriteerror_with_non_dup_propagates_as_bulk():
    bwe = BulkWriteError({"writeErrors": [
        {"code": 11000, "errmsg": "dup"},
        {"code": 121, "errmsg": "validation"},
    ]})
    db = {"documents": _FakeColl(bwe)}
    try:
        _insert_chunks(db, [{"x": 1}, {"x": 2}])
    except BulkWriteError:
        pass
    except DuplicateKeyError:
        raise AssertionError("non-dup writeError must NOT collapse to DuplicateKeyError")


def test_is_dup_key_bulk_helper():
    assert _is_dup_key_bulk(BulkWriteError({"writeErrors": [{"code": 11000}]}))
    assert not _is_dup_key_bulk(BulkWriteError({"writeErrors": [{"code": 121}]}))
    assert not _is_dup_key_bulk(BulkWriteError({"writeErrors": []}))
