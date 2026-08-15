# train-api/app.py
# ============================================================
# MODULE SUMMARY
# ------------------------------------------------------------
# Role: FastAPI service that runs the ingestion -> feature ->
# train pipeline for the three prediction targets (gini,
# mobility, income_group) as background jobs, mirroring the
# async job pattern from the Rakuten train-api (POST returns a
# job_id immediately; poll GET /train/status/{id}).
#
# Endpoints:
#   POST /train              {"target": "gini|mobility|income_group|all",
#                              "run_ingestion": false}
#                            -> 202 {"job_id": "...", "status": "running"} | 409 if busy
#   GET  /train/status/{id}  -> {"status": "success|running|failed", "log": [...]}
#   GET  /health
#   GET  /metrics            (Prometheus)
# ============================================================
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train-api")

ROOT = Path(__file__).resolve().parent.parent
SERVICES_DIR = Path(__file__).resolve().parent / "services"

app = FastAPI(title="Income Inequality — Train API", version="0.1")
Instrumentator().instrument(app).expose(app)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
_ACTIVE_JOB: str | None = None

TARGET_SCRIPTS = {
    "gini": SERVICES_DIR / "train_gini.py",
    "mobility": SERVICES_DIR / "train_mobility.py",
    "income_group": SERVICES_DIR / "train_income_group.py",
}
INGEST_SCRIPTS = [
    ROOT / "ingestion" / "ingest_worldbank.py",
    ROOT / "ingestion" / "ingest_oecd.py",
    ROOT / "ingestion" / "ingest_eurostat.py",
    ROOT / "ingestion" / "ingest_wid.py",
    ROOT / "ingestion" / "ingest_gdim.py",
    ROOT / "ingestion" / "merge_sources.py",
    ROOT / "features" / "build_features.py",
]


class TrainRequest(BaseModel):
    target: str = "all"          # "gini" | "mobility" | "income_group" | "all"
    run_ingestion: bool = False  # re-pull data from all 4 sources + rebuild features first


def _run_script(path: Path, job_id: str) -> bool:
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), capture_output=True, text=True)
    JOBS[job_id]["log"].append({"script": str(path.relative_to(ROOT)), "returncode": result.returncode,
                                 "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]})
    return result.returncode == 0


def _run_job(job_id: str, target: str, run_ingestion: bool) -> None:
    global _ACTIVE_JOB
    try:
        JOBS[job_id]["status"] = "running"
        if run_ingestion:
            for script in INGEST_SCRIPTS:
                ok = _run_script(script, job_id)
                if not ok and script.name not in ("ingest_gdim.py", "ingest_wid.py"):  # GDIM needs a manual file; WID's public API is best-effort (see ingest_wid.py docstring)
                    raise RuntimeError(f"{script.name} failed — see log")

        targets = list(TARGET_SCRIPTS) if target == "all" else [target]
        for t in targets:
            ok = _run_script(TARGET_SCRIPTS[t], job_id)
            if not ok:
                raise RuntimeError(f"train_{t}.py failed — see log")

        JOBS[job_id]["status"] = "success"
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)
    finally:
        with JOBS_LOCK:
            _ACTIVE_JOB = None


@app.post("/train", status_code=202)
def train(request: TrainRequest):
    global _ACTIVE_JOB
    if request.target not in {"gini", "mobility", "income_group", "all"}:
        raise HTTPException(status_code=400, detail="target must be one of gini|mobility|income_group|all")

    with JOBS_LOCK:
        if _ACTIVE_JOB is not None and JOBS.get(_ACTIVE_JOB, {}).get("status") == "running":
            raise HTTPException(status_code=409, detail=f"Job {_ACTIVE_JOB} already running")
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued", "target": request.target, "log": []}
        _ACTIVE_JOB = job_id

    thread = threading.Thread(target=_run_job, args=(job_id, request.target, request.run_ingestion), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/train/status/{job_id}")
def train_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, **job}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "train-api", "active_job": _ACTIVE_JOB}


@app.get("/")
def root():
    return {"status": "API up and running", "endpoints": ["/train", "/train/status/{id}", "/health", "/metrics"]}
