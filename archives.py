"""Safe extraction of ZIP/RAR archives with size, count, traversal, encryption guards."""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath


class ArchiveError(Exception):
    pass


@dataclass
class ArchiveEntry:
    name: str          # original (possibly nested) path inside archive
    data: bytes | None # None if skipped
    skipped: bool = False
    skip_reason: str | None = None


_NESTED_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"}


def _is_safe_path(name: str) -> bool:
    """Reject absolute paths, drive letters, and `..` traversal."""
    if not name or name.endswith("/"):
        return False
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute():
        return False
    if any(part == ".." for part in p.parts):
        return False
    if len(p.parts) > 0 and ":" in p.parts[0]:  # Windows drive letter
        return False
    return True


def extract_archive(
    data: bytes,
    *,
    mime_type: str,
    max_files: int,
    max_uncompressed_mb: int,
) -> list[ArchiveEntry]:
    if mime_type in ("application/zip", "application/x-zip-compressed"):
        return _extract_zip(data, max_files=max_files, max_uncompressed_mb=max_uncompressed_mb)
    if mime_type in ("application/x-rar", "application/vnd.rar", "application/x-rar-compressed"):
        return _extract_rar(data, max_files=max_files, max_uncompressed_mb=max_uncompressed_mb)
    raise ArchiveError(f"unsupported archive mime: {mime_type}")


def _extract_zip(data: bytes, *, max_files: int, max_uncompressed_mb: int) -> list[ArchiveEntry]:
    max_bytes = max_uncompressed_mb * 1024 * 1024
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ArchiveError(f"corrupt zip: {e}") from e

    infos = zf.infolist()

    # Guard 1: count
    real_files = [i for i in infos if not i.is_dir()]
    if len(real_files) > max_files:
        raise ArchiveError(f"too many files in archive: {len(real_files)} > {max_files}")

    # Guard 2: encrypted (any entry)
    for i in real_files:
        if i.flag_bits & 0x1:
            raise ArchiveError("encrypted archive entries not supported")

    # Guard 3: pre-extraction size sum (anti zip-bomb)
    total = sum(i.file_size for i in real_files)
    if total > max_bytes:
        raise ArchiveError(
            f"archive exceeds uncompressed cap: {total // (1024*1024)} MB > {max_uncompressed_mb} MB"
        )

    out: list[ArchiveEntry] = []
    for info in real_files:
        # Guard 4: path traversal — silently drop (no entry returned)
        if not _is_safe_path(info.filename):
            continue
        # Guard 5: depth=1 — skip nested archives
        ext = os.path.splitext(info.filename)[1].lower()
        if ext in _NESTED_EXTS:
            out.append(ArchiveEntry(name=info.filename, data=None, skipped=True,
                                    skip_reason="nested archive (depth=1 policy)"))
            continue
        out.append(ArchiveEntry(name=info.filename, data=zf.read(info)))
    return out


def _extract_rar(data: bytes, *, max_files: int, max_uncompressed_mb: int) -> list[ArchiveEntry]:
    import rarfile
    max_bytes = max_uncompressed_mb * 1024 * 1024
    try:
        rf = rarfile.RarFile(io.BytesIO(data))
    except rarfile.Error as e:
        raise ArchiveError(f"corrupt rar: {e}") from e

    if rf.needs_password():
        raise ArchiveError("encrypted archive not supported")

    infos = [i for i in rf.infolist() if not i.isdir()]
    if len(infos) > max_files:
        raise ArchiveError(f"too many files: {len(infos)} > {max_files}")

    total = sum(i.file_size for i in infos)
    if total > max_bytes:
        raise ArchiveError(
            f"archive exceeds uncompressed cap: {total // (1024*1024)} MB > {max_uncompressed_mb} MB"
        )

    out: list[ArchiveEntry] = []
    for info in infos:
        if not _is_safe_path(info.filename):
            out.append(ArchiveEntry(name=info.filename, data=None, skipped=True,
                                    skip_reason="unsafe path"))
            continue
        ext = os.path.splitext(info.filename)[1].lower()
        if ext in _NESTED_EXTS:
            out.append(ArchiveEntry(name=info.filename, data=None, skipped=True,
                                    skip_reason="nested archive (depth=1 policy)"))
            continue
        out.append(ArchiveEntry(name=info.filename, data=rf.read(info)))
    return out
