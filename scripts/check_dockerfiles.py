#!/usr/bin/env python3
"""Lightweight Dockerfile sanity checks when hadolint is unavailable."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "Dockerfile", ROOT / "Dockerfile.frontend"]

REQUIRED_SNIPPETS = {
    "Dockerfile": [
        r"^FROM\s+python:",
        r"fonts-noto-cjk",
        r"pip install",
        r"\[ocr,server\]",
        r"HEALTHCHECK",
        r"GEMINI_API_KEY",
        r"INSTALL_LIBREOFFICE",
        r"USER\s+pdf2ppt",
        r"EXPOSE\s+8008",
        r"PDF2PPT_ROOT=/data",
    ],
    "Dockerfile.frontend": [
        r"^FROM\s+node:",
        r"npm ci",
        r"npm run build",
        r"^FROM\s+nginx:",
        r"HEALTHCHECK",
    ],
}


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing file: {path}"]
    text = path.read_text(encoding="utf-8")
    if "\t" in text:
        errors.append(f"{path.name}: contains tab characters")
    for pattern in REQUIRED_SNIPPETS[path.name]:
        if re.search(pattern, text, flags=re.MULTILINE) is None:
            errors.append(f"{path.name}: expected pattern not found: {pattern}")
    # Secrets must not be assigned in ENV/ARG with literal non-empty values.
    for match in re.finditer(r"(?i)^\s*(?:ENV|ARG)\s+.*(?:API_KEY|TOKEN|PASSWORD)\s*=\s*(\S+)", text, re.M):
        value = match.group(1).strip().strip("\"'")
        if value and value not in {"", "0", "1"} and not value.startswith("$"):
            errors.append(f"{path.name}: suspicious secret assignment: {match.group(0).strip()}")
    return errors


def main() -> int:
    all_errors: list[str] = []
    for path in FILES:
        all_errors.extend(check_file(path))
    compose = ROOT / "docker-compose.yml"
    if not compose.is_file():
        all_errors.append("missing docker-compose.yml")
    else:
        compose_text = compose.read_text(encoding="utf-8")
        for needle in (
            "healthcheck:",
            "GEMINI_API_KEY",
            "/data/model",
            "pdf2ppt-jobs",
            "pdf2ppt-models",
            "PDF2PPT_ROOT",
        ):
            if needle not in compose_text:
                all_errors.append(f"docker-compose.yml: missing {needle!r}")
    ignore = ROOT / ".dockerignore"
    if not ignore.is_file():
        all_errors.append("missing .dockerignore")
    if all_errors:
        print("Dockerfile checks failed:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Dockerfile checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
