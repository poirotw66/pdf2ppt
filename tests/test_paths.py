from __future__ import annotations

import os
import unittest
from pathlib import Path

from pdf2ppt.paths import get_repo_root, resolve_repo_relative_path


class RepoPathResolutionTests(unittest.TestCase):
    def test_resolve_repo_relative_path_from_nested_working_directory(self) -> None:
        repo_root = get_repo_root()
        previous_cwd = Path.cwd()
        nested_dir = repo_root / "frontend"
        nested_dir.mkdir(exist_ok=True)
        os.chdir(nested_dir)
        try:
            resolved = resolve_repo_relative_path("lama/big-lama")
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(resolved, repo_root / "lama" / "big-lama")
        self.assertTrue((resolved / "config.yaml").exists() or not (repo_root / "lama" / "big-lama").exists())

    def test_resolve_repo_relative_path_keeps_absolute_paths(self) -> None:
        absolute = Path("/tmp/lama-model")
        self.assertEqual(resolve_repo_relative_path(absolute), absolute.resolve())
