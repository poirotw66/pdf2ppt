from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_repo_root() -> Path:
    env_root = os.environ.get("PDF2PPT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    package_dir = Path(__file__).resolve().parent
    for candidate in (package_dir.parent.parent, *package_dir.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "pdf2ppt").is_dir():
            return candidate
    return Path.cwd().resolve()


def resolve_repo_relative_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (get_repo_root() / path).resolve()
