"""Safe zip extraction: size caps, zip bombs, traversal, symlinks."""

from __future__ import annotations

import io
import stat
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from pdf2ppt.core import InputValidationError
from pdf2ppt.safe_zip import (
    ZipSafetyLimits,
    reject_oversized_compressed,
    safe_extract_zip,
    safe_extract_zip_bytes,
    validate_zip_archive,
)


def _zip_bytes(write) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write(archive)
    return buffer.getvalue()


class SafeZipTests(unittest.TestCase):
    def test_rejects_oversized_compressed(self) -> None:
        limits = ZipSafetyLimits(max_compressed_bytes=100)
        with self.assertRaises(InputValidationError) as ctx:
            reject_oversized_compressed(101, limits=limits, label="assets zip")
        self.assertIn("compressed upload limit", str(ctx.exception))

    def test_rejects_path_traversal(self) -> None:
        data = _zip_bytes(lambda z: z.writestr("../evil.txt", b"pwned"))
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=ZipSafetyLimits())
            self.assertIn("traversal", str(ctx.exception).lower())
            self.assertFalse((Path(tmp) / "evil.txt").exists())

    def test_rejects_absolute_path(self) -> None:
        data = _zip_bytes(lambda z: z.writestr("/tmp/evil.txt", b"pwned"))
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError):
                safe_extract_zip_bytes(data, dest, limits=ZipSafetyLimits())

    def test_rejects_symlink_member(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            info = zipfile.ZipInfo("link-name")
            info.create_system = 3  # Unix
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"/etc/passwd")
        data = buffer.getvalue()
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=ZipSafetyLimits())
            self.assertIn("symlink", str(ctx.exception).lower())

    def test_rejects_zip_bomb_compression_ratio(self) -> None:
        # Highly compressible payload: tiny compressed, huge declared/actual size.
        payload = b"\x00" * (200_000)
        data = _zip_bytes(lambda z: z.writestr("bomb.bin", payload))
        limits = ZipSafetyLimits(
            max_compressed_bytes=10 * 1024 * 1024,
            max_uncompressed_bytes=10 * 1024 * 1024,
            max_entry_bytes=10 * 1024 * 1024,
            max_compression_ratio=10.0,
            max_entries=100,
        )
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=limits)
            message = str(ctx.exception).lower()
            self.assertTrue(
                "compression-ratio" in message or "compression ratio" in message,
                msg=message,
            )

    def test_rejects_too_many_entries(self) -> None:
        def write_many(archive: zipfile.ZipFile) -> None:
            for index in range(20):
                archive.writestr(f"f{index}.txt", b"x")

        data = _zip_bytes(write_many)
        limits = ZipSafetyLimits(max_entries=5)
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=limits)
            self.assertIn("too many members", str(ctx.exception))

    def test_rejects_total_uncompressed_over_limit(self) -> None:
        data = _zip_bytes(
            lambda z: (
                z.writestr("a.bin", b"a" * 60),
                z.writestr("b.bin", b"b" * 60),
            )
        )
        limits = ZipSafetyLimits(
            max_uncompressed_bytes=100,
            max_entry_bytes=80,
            max_compression_ratio=1000.0,
        )
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=limits)
            self.assertIn("uncompressed", str(ctx.exception).lower())

    def test_happy_path_extracts_files(self) -> None:
        data = _zip_bytes(
            lambda z: (
                z.writestr("pages/p1.png", b"png-bytes"),
                z.writestr("meta.txt", b"ok"),
            )
        )
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            safe_extract_zip_bytes(data, dest, limits=ZipSafetyLimits())
            self.assertEqual((dest / "pages" / "p1.png").read_bytes(), b"png-bytes")
            self.assertEqual((dest / "meta.txt").read_bytes(), b"ok")

    def test_validate_then_extract_via_zipfile(self) -> None:
        data = _zip_bytes(lambda z: z.writestr("nested/a.txt", b"hello"))
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = validate_zip_archive(archive, limits=ZipSafetyLimits())
                self.assertEqual(len(members), 1)
                safe_extract_zip(archive, dest, limits=ZipSafetyLimits())
            self.assertEqual((dest / "nested" / "a.txt").read_text(encoding="utf-8"), "hello")

    def test_rejects_dot_entry_name(self) -> None:
        data = _zip_bytes(lambda z: z.writestr(".", b"root"))
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=ZipSafetyLimits())
            self.assertIn("extraction root", str(ctx.exception).lower())

    def test_rejects_file_then_nested_path(self) -> None:
        def write(archive: zipfile.ZipFile) -> None:
            archive.writestr("pages", b"i-am-a-file")
            archive.writestr("pages/y.txt", b"nested")

        data = _zip_bytes(write)
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=ZipSafetyLimits())
            message = str(ctx.exception).lower()
            self.assertTrue(
                "failed to extract" in message
                or "cannot create directory" in message
                or "file exists" in message,
                msg=message,
            )

    def test_rejects_overlong_filename(self) -> None:
        long_name = "a" * 400 + ".bin"
        data = _zip_bytes(lambda z: z.writestr(long_name, b"x"))
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            with self.assertRaises(InputValidationError) as ctx:
                safe_extract_zip_bytes(data, dest, limits=ZipSafetyLimits())
            message = str(ctx.exception).lower()
            self.assertTrue(
                "failed to extract" in message or "file name" in message or "too long" in message,
                msg=message,
            )


if __name__ == "__main__":
    unittest.main()
