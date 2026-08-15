"""
Shared pytest fixtures. Adds the service directories to sys.path so tests can
`import app` / `import common` the same way uvicorn does when each service
runs with its own directory as the working directory (see each Dockerfile's
WORKDIR).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

for path in [
    ROOT / "train-api",
    ROOT / "train-api" / "services",
    ROOT / "predict-api",
    ROOT / "ingestion",
    ROOT / "features",
]:
    sys.path.insert(0, str(path))


@pytest.fixture
def import_service_app():
    """Import a service's app.py by explicit directory, bypassing sys.modules'
    by-name cache.

    train-api/app.py and predict-api/app.py are both literally named "app.py",
    and both their directories sit on sys.path at once (above). A plain
    `import app` caches under the single key "app" — whichever service's
    app.py gets imported first "wins" for the rest of the pytest process, so a
    later `import app` in another test file silently gets served the wrong
    service's module instead of raising ImportError. Only bites when the full
    suite runs together (exactly what CI does), not when running a single
    test file in isolation — evict the cache and re-resolve against the
    intended service's directory every time instead.
    """

    def _import(service_dir: str):
        sys.modules.pop("app", None)
        target = str(ROOT / service_dir)
        if target in sys.path:
            sys.path.remove(target)
        sys.path.insert(0, target)
        return importlib.import_module("app")

    return _import


@pytest.fixture
def tmp_repo_root(tmp_path, monkeypatch):
    """A throwaway repo layout (data/raw, data/processed, data/artifacts) for tests that write files."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "artifacts").mkdir(parents=True)
    return tmp_path
