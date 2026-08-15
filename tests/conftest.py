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


@pytest.fixture
def import_service_app():
    """Import a service's app.py by explicit file path, under a name unique
    to that service — not the ambiguous, colliding `import app`.

    train-api/app.py and predict-api/app.py are both literally named "app.py",
    and both their directories sit on sys.path at once (above). A plain
    `import app` caches under the single key "app" — whichever service's
    app.py gets imported first "wins" for the rest of the pytest process, so a
    later `import app` in another test file silently gets served the wrong
    service's module instead of raising ImportError. Only bites when the full
    suite runs together (exactly what CI does), not when running a single
    test file in isolation.

    Loading by explicit path under a distinct module name (e.g.
    "_service_app_predict_api") avoids that collision. Cached in sys.modules
    per service (not re-executed on every call) because each app.py registers
    module-level Prometheus Counter/Histogram objects into the global default
    CollectorRegistry at import time — re-running that registration a second
    time raises "Duplicated timeseries in CollectorRegistry", so re-importing
    fresh on every test would trade the collision bug for a re-registration
    crash instead of actually fixing anything.
    """

    def _import(service_dir: str):
        cache_key = f"_service_app_{service_dir.replace('-', '_')}"
        if cache_key in sys.modules:
            return sys.modules[cache_key]
        app_path = ROOT / service_dir / "app.py"
        spec = importlib.util.spec_from_file_location(cache_key, app_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[cache_key] = module
        spec.loader.exec_module(module)
        return module

    return _import


@pytest.fixture
def tmp_repo_root(tmp_path, monkeypatch):
    """A throwaway repo layout (data/raw, data/processed, data/artifacts) for tests that write files."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "artifacts").mkdir(parents=True)
    return tmp_path
