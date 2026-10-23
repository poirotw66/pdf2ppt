"""Safe zip extraction with size / zip-bomb / path-traversal guards.

Limits are configurable via environment variables (bytes / counts / ratios):

- ``PDF2PPT_ZIP_MAX_COMPRESSED_BYTES`` (default 50 MiB)
- ``PDF2PPT_ZIP_MAX_UNCOMPRESSED_BYTES`` (default 200 MiB)
- ``PDF2PPT_ZIP_MAX_ENTRIES`` (default 10_000)
- ``PDF2PPT_ZIP_MAX_ENTRY_BYTES`` (default 50 MiB)
- ``PDF2PPT_ZIP_MAX_COMPRESSION_RATIO`` (default 100.0)

Used by ``POST /from-ir`` assets uploads and any other zip → assets extraction.
"""

from __future__ import annotations

import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .core import InputValidationError

DEFAULT_ZIP_MAX_COMPRESSED_BYTES = 50 * 1024 * 1024
DEFAULT_ZIP_MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
DEFAULT_ZIP_MAX_ENTRIES = 10_000
DEFAULT_ZIP_MAX_ENTRY_BYTES = 50 * 1024 * 1024
DEFAULT_ZIP_MAX_COMPRESSION_RATIO = 100.0

# Zipinfo external_attr high bits encode Unix mode when present.
_ZIP_UNIX_SYMLINK_MODE = stat.S_IFLNK


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        value = int(str(raw).strip(), 10)
    except ValueError as error:
        raise InputValidationError(f"Invalid integer for {name}: {raw!r}") from error
    if value < 1:
        raise InputValidationError(f"{name} must be >= 1 (got {value}).")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(str(raw).strip())
    except ValueError as error:
        raise InputValidationError(f"Invalid number for {name}: {raw!r}") from error
    if value < 1.0:
        raise InputValidationError(f"{name} must be >= 1 (got {value}).")
    return value


@dataclass(frozen=True, slots=True)
class ZipSafetyLimits:
    max_compressed_bytes: int = DEFAULT_ZIP_MAX_COMPRESSED_BYTES
    max_uncompressed_bytes: int = DEFAULT_ZIP_MAX_UNCOMPRESSED_BYTES
    max_entries: int = DEFAULT_ZIP_MAX_ENTRIES
    max_entry_bytes: int = DEFAULT_ZIP_MAX_ENTRY_BYTES
    max_compression_ratio: float = DEFAULT_ZIP_MAX_COMPRESSION_RATIO

    @classmethod
    def from_environ(cls) -> ZipSafetyLimits:
        return cls(
            max_compressed_bytes=_env_int(
                "PDF2PPT_ZIP_MAX_COMPRESSED_BYTES", DEFAULT_ZIP_MAX_COMPRESSED_BYTES
            ),
            max_uncompressed_bytes=_env_int(
                "PDF2PPT_ZIP_MAX_UNCOMPRESSED_BYTES", DEFAULT_ZIP_MAX_UNCOMPRESSED_BYTES
            ),
            max_entries=_env_int("PDF2PPT_ZIP_MAX_ENTRIES", DEFAULT_ZIP_MAX_ENTRIES),
            max_entry_bytes=_env_int(
                "PDF2PPT_ZIP_MAX_ENTRY_BYTES", DEFAULT_ZIP_MAX_ENTRY_BYTES
            ),
            max_compression_ratio=_env_float(
                "PDF2PPT_ZIP_MAX_COMPRESSION_RATIO", DEFAULT_ZIP_MAX_COMPRESSION_RATIO
            ),
        )


def reject_oversized_compressed(size: int, *, limits: ZipSafetyLimits | None = None, label: str = "zip") -> None:
    """Reject an upload whose compressed size exceeds the configured cap."""

    resolved = limits or ZipSafetyLimits.from_environ()
    if size > resolved.max_compressed_bytes:
        raise InputValidationError(
            f"{label} exceeds the {resolved.max_compressed_bytes // (1024 * 1024)} MiB "
            f"compressed upload limit ({size} bytes)."
        )


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    # Unix symlinks set the high 16 bits of external_attr to S_IFLNK.
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and (mode & 0xF000) == _ZIP_UNIX_SYMLINK_MODE:
        return True
    # Some creators only set create_system=3 (Unix) with symlink mode.
    if info.create_system == 3 and stat.S_ISLNK(mode):
        return True
    return False


def _validate_member_name(name: str) -> Path:
    if not name or name.endswith("/"):
        # Directories are allowed as names but skipped for size accounting.
        return Path(name)
    path = Path(name)
    if path.is_absolute() or name.startswith("/") or name.startswith("\\"):
        raise InputValidationError(f"Unsafe absolute path in zip: {name!r}")
    # Reject Windows drive / UNC style and empty / dot segments used for traversal.
    if len(path.parts) > 0 and path.parts[0].endswith(":"):
        raise InputValidationError(f"Unsafe absolute path in zip: {name!r}")
    if ".." in path.parts:
        raise InputValidationError(f"Unsafe path traversal in zip: {name!r}")
    if any(part == "" for part in path.parts if part is not None):
        raise InputValidationError(f"Unsafe empty path segment in zip: {name!r}")
    return path


def _compression_ratio(compressed: int, uncompressed: int) -> float:
    if uncompressed <= 0:
        return 0.0
    # Treat missing/zero compressed size as highly suspicious for bomb checks.
    denom = max(1, int(compressed))
    return float(uncompressed) / float(denom)


def validate_zip_archive(
    archive: zipfile.ZipFile,
    *,
    limits: ZipSafetyLimits | None = None,
) -> list[zipfile.ZipInfo]:
    """Validate members for traversal, symlinks, sizes, and compression ratio.

    Returns the file members (non-directory) that may be extracted.
    """

    resolved = limits or ZipSafetyLimits.from_environ()
    members = archive.infolist()
    if len(members) > resolved.max_entries:
        raise InputValidationError(
            f"zip has too many members ({len(members)}; limit {resolved.max_entries})."
        )

    total_uncompressed = 0
    file_members: list[zipfile.ZipInfo] = []
    for info in members:
        name = info.filename
        if not name:
            raise InputValidationError("zip contains an entry with an empty name.")
        if _is_symlink_member(info):
            raise InputValidationError(f"Symlinks are not allowed in zip: {name!r}")
        _validate_member_name(name)
        if name.endswith("/"):
            continue
        uncompressed = int(info.file_size)
        compressed = int(info.compress_size)
        if uncompressed < 0 or compressed < 0:
            raise InputValidationError(f"Invalid sizes for zip entry {name!r}.")
        if uncompressed > resolved.max_entry_bytes:
            raise InputValidationError(
                f"zip entry {name!r} exceeds per-entry size limit "
                f"({uncompressed} > {resolved.max_entry_bytes} bytes)."
            )
        ratio = _compression_ratio(compressed, uncompressed)
        if ratio > resolved.max_compression_ratio + 1e-9:
            raise InputValidationError(
                f"zip entry {name!r} fails compression-ratio check "
                f"({ratio:.1f} > {resolved.max_compression_ratio})."
            )
        total_uncompressed += uncompressed
        if total_uncompressed > resolved.max_uncompressed_bytes:
            raise InputValidationError(
                f"zip total uncompressed size exceeds limit "
                f"({total_uncompressed} > {resolved.max_uncompressed_bytes} bytes)."
            )
        file_members.append(info)
    return file_members


def _os_error_detail(error: OSError) -> str:
    """Human-readable OSError detail without absolute filesystem paths."""

    detail = error.strerror or type(error).__name__
    if error.errno is not None:
        return f"[Errno {error.errno}] {detail}"
    return detail


def safe_extract_zip(
    archive: zipfile.ZipFile,
    dest: Path,
    *,
    limits: ZipSafetyLimits | None = None,
) -> None:
    """Extract ``archive`` into ``dest`` after safety checks; stream with size caps."""

    resolved = limits or ZipSafetyLimits.from_environ()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    file_members = validate_zip_archive(archive, limits=resolved)

    written_total = 0
    chunk_size = 64 * 1024
    for info in file_members:
        rel = Path(info.filename)
        target = (dest / rel).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as error:
            raise InputValidationError(f"Unsafe path in zip: {info.filename!r}") from error
        if target == dest_resolved:
            raise InputValidationError(
                f"Unsafe zip entry resolves to the extraction root: {info.filename!r}"
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise InputValidationError(
                f"Cannot create directory for zip entry {info.filename!r}: {_os_error_detail(error)}"
            ) from error

        # Extract via streaming so a lying file_size cannot blow past limits.
        written_entry = 0
        try:
            with archive.open(info, "r") as source, target.open("wb") as out:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    written_entry += len(chunk)
                    written_total += len(chunk)
                    if written_entry > resolved.max_entry_bytes:
                        raise InputValidationError(
                            f"zip entry {info.filename!r} exceeded per-entry size limit while extracting."
                        )
                    if written_total > resolved.max_uncompressed_bytes:
                        raise InputValidationError(
                            "zip exceeded total uncompressed size limit while extracting."
                        )
                    # Compression-ratio check against observed bytes (zip bomb).
                    compressed = max(1, int(info.compress_size))
                    if written_entry / compressed > resolved.max_compression_ratio + 1e-9:
                        raise InputValidationError(
                            f"zip entry {info.filename!r} failed compression-ratio check while extracting."
                        )
                    out.write(chunk)
        except InputValidationError:
            raise
        except OSError as error:
            raise InputValidationError(
                f"Failed to extract zip entry {info.filename!r}: {_os_error_detail(error)}"
            ) from error

        # Refuse to leave a symlink behind even if the archive lied about type.
        if target.is_symlink():
            target.unlink(missing_ok=True)
            raise InputValidationError(f"Symlinks are not allowed in zip: {info.filename!r}")


def safe_extract_zip_bytes(
    data: bytes,
    dest: Path,
    *,
    limits: ZipSafetyLimits | None = None,
    label: str = "assets zip",
) -> None:
    """Validate compressed size then extract bytes into ``dest``."""

    import io

    resolved = limits or ZipSafetyLimits.from_environ()
    reject_oversized_compressed(len(data), limits=resolved, label=label)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            safe_extract_zip(archive, dest, limits=resolved)
    except zipfile.BadZipFile as error:
        raise InputValidationError(f"Invalid {label}: {error}") from error


__all__ = [
    "DEFAULT_ZIP_MAX_COMPRESSED_BYTES",
    "DEFAULT_ZIP_MAX_COMPRESSION_RATIO",
    "DEFAULT_ZIP_MAX_ENTRIES",
    "DEFAULT_ZIP_MAX_ENTRY_BYTES",
    "DEFAULT_ZIP_MAX_UNCOMPRESSED_BYTES",
    "ZipSafetyLimits",
    "reject_oversized_compressed",
    "safe_extract_zip",
    "safe_extract_zip_bytes",
    "validate_zip_archive",
]
