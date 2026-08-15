"""
Shared pytest fixtures. Adds the service directories to sys.path so tests can
`import app` / `import common` the same way uvicorn does when each service
runs with its own directory as the working directory (see each Dockerfile's
WORKDIR).
"""

from __future__ import annotations

import importlib.util
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


def _import_from(dir_relpath: str, filename: str):
    """Import a module by explicit file path, under a name unique to its
    directory — not the ambiguous, colliding `import <module-name-only>`.

    Several same-named files sit in different service directories that are
    ALL on sys.path at once (above): train-api/app.py vs predict-api/app.py,
    and ingestion/common.py vs train-api/services/common.py. A plain `import
    app` (or `import common`) caches under that single bare name — whichever
    one gets imported first "wins" for the rest of the pytest process, so a
    later import in another test file silently gets served the wrong
    module instead of raising ImportError. Only bites when the full suite
    runs together (exactly what CI does), not when running a single test
    file in isolation — which is why this went unnoticed until CI actually
    started running.

    Loading by explicit path under a distinct, path-derived module name (e.g.
    "_module_train_api_services_common") avoids that collision. Cached in
    sys.modules per (dir, filename) pair — not re-executed on every call —
    because predict-api/app.py registers module-level Prometheus Counter/
    Histogram objects into the global default CollectorRegistry at import
    time; re-running that registration a second time raises "Duplicated
    timeseries in CollectorRegistry", so re-importing fresh on every call
    would trade the collision bug for a re-registration crash instead of
    actually fixing anything.
    """
    cache_key = "_module_" + f"{dir_relpath}/{filename}".replace("-", "_").replace("/", "_").removesuffix(".py")
    if cache_key in sys.modules:
        return sys.modules[cache_key]
    module_path = ROOT / dir_relpath / filename
    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def import_service_app():
    """Import a service's app.py by explicit directory — see _import_from()."""

    def _import(service_dir: str):
        return _import_from(service_dir, "app.py")

    return _import


@pytest.fixture
def import_service_module():
    """Import any module by explicit (directory, filename) — see _import_from()."""
    return _import_from


@pytest.fixture
def tmp_repo_root(tmp_path, monkeypatch):
    """A throwaway repo layout (data/raw, data/processed, data/artifacts) for tests that write files."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "artifacts").mkdir(parents=True)
    return tmp_path
