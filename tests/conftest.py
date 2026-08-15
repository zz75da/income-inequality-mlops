"""
Shared pytest fixtures. Adds the service directories to sys.path so tests can
`import app` / `import common` the same way uvicorn does when each service
runs with its own directory as the working directory (see each Dockerfile's
WORKDIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for path in [
    ROOT / "train-api",
    ROOT / "train-api" / "services",
    ROOT / "predict-api",
    ROOT / "ingestion",
    ROOT / "features",
]:
    sys.path.insert(0, str(path))

import pytest


@pytest.fixture
def tmp_repo_root(tmp_path, monkeypatch):
    """A throwaway repo layout (data/raw, data/processed, data/artifacts) for tests that write files."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "artifacts").mkdir(parents=True)
    return tmp_path
