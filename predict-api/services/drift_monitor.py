"""
Lightweight Evidently-based drift monitor for predict-api: buffers incoming
feature rows in memory, and once params.yaml's monitoring.drift_reference_min_rows
is reached, auto-generates an HTML report comparing the buffer against a
reference sample lazily read from data/processed/features.csv (and dropped
on /reload-artifacts so it tracks the current model's training data, not
whatever features.csv looked like at first use). Simplified for a low-QPS
tabular API — portfolio-project traffic, not production scale.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from prometheus_client import Gauge

logger = logging.getLogger("predict-api.drift")

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "data" / "artifacts" / "drift_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

with open(ROOT / "params.yaml") as _f:
    _PARAMS = yaml.safe_load(_f)
DRIFT_REFERENCE_MIN_ROWS = _PARAMS["monitoring"]["drift_reference_min_rows"]
DRIFT_PSI_THRESHOLD = _PARAMS["monitoring"]["drift_psi_threshold"]

# Registered on the default registry, same as app.py's PREDICTION_COUNT/
# PREDICTION_LATENCY, so Instrumentator's expose(app) picks these up for
# free on the next Prometheus scrape — no pushgateway needed, this process
# is already scraped directly.
DRIFT_SHARE = Gauge(
    "drift_share_of_columns",
    "Share of features flagged as drifted in the last drift report (Evidently DataDriftPreset)",
)
DRIFT_DETECTED = Gauge(
    "drift_dataset_detected",
    "1 if the last drift report flagged dataset-level drift, else 0",
)

_buffer: list[dict] = []
_lock = threading.Lock()
_reference_df: pd.DataFrame | None = None
MAX_REPORTS_KEPT = 10


def _load_reference() -> pd.DataFrame | None:
    global _reference_df
    if _reference_df is not None:
        return _reference_df
    if not FEATURES_PATH.exists():
        logger.warning("No features.csv yet — drift reference unavailable until training data exists.")
        return None
    _reference_df = pd.read_csv(FEATURES_PATH)
    return _reference_df


def refresh_reference() -> None:
    """Drop the cached reference sample so the next drift report re-reads
    features.csv from disk. Without this, a long-running predict-api process
    keeps comparing against whatever features.csv looked like at the first
    prediction after startup — even though training regenerates it on every
    Airflow run and POST /reload-artifacts fires right after. Called from
    app.py's /reload-artifacts so the drift reference actually tracks what
    the current model was trained on."""
    global _reference_df
    _reference_df = None


def record_prediction(features: dict) -> None:
    with _lock:
        _buffer.append(features)
    # Auto-trigger once the buffer reaches params.yaml's configured minimum —
    # previously this threshold was declared but never actually read
    # anywhere, so a report only ever got built via a manual POST
    # /drift-trigger-report call; the buffer would otherwise grow forever
    # and the Streamlit Drift Reports page's "generates one once the buffer
    # fills up" message was simply false. Run inline rather than in a
    # background thread — low-QPS portfolio traffic, consistent with this
    # service's "kept deliberately simple" scope (see README's Design Scope).
    if buffer_size() >= DRIFT_REFERENCE_MIN_ROWS:
        trigger_report(list(features.keys()))


def buffer_size() -> int:
    with _lock:
        return len(_buffer)


def reference_exists() -> bool:
    return _load_reference() is not None


def _prune_old_reports() -> None:
    reports = sorted(REPORTS_DIR.glob("drift_report_*.html"))
    while len(reports) > MAX_REPORTS_KEPT:
        oldest = reports.pop(0)
        oldest.unlink(missing_ok=True)


REPORT_NAME_PATTERN = re.compile(r"^drift_report_\d{8}T\d{6}Z\.html$")


def list_reports() -> list[str]:
    """Newest first — served over HTTP (GET /drift-reports) so Streamlit
    Cloud can list them. predict-api and the Streamlit frontend run on
    separate hosts (Render / Streamlit Cloud) with no shared filesystem —
    reading REPORTS_DIR locally, which is all the Streamlit app used to do,
    only ever worked in docker-compose where both containers bind-mount the
    same ./data directory."""
    return [p.name for p in sorted(REPORTS_DIR.glob("drift_report_*.html"), reverse=True)]


def get_report_html(name: str) -> str | None:
    """Read one report's HTML by filename (GET /drift-reports/{name}).
    Validates against REPORT_NAME_PATTERN first — name comes straight from a
    URL path parameter, and without this a caller could pass `../../` style
    input and read arbitrary files off the container."""
    if not REPORT_NAME_PATTERN.match(name):
        return None
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def trigger_report(feature_columns: list[str]) -> str | None:
    """Build an Evidently HTML drift report from the current buffer vs. reference."""
    reference = _load_reference()
    if reference is None:
        return None
    with _lock:
        if not _buffer:
            return None
        current = pd.DataFrame(_buffer)
        _buffer.clear()

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("evidently not installed — skipping drift report generation")
        return None

    # A column entirely null in either side (e.g. top10/bottom50_income_share
    # — WID.world's public API is retired, so the reference sample has zero
    # real values for them) makes Evidently raise instead of just skipping
    # it, crashing the whole report.
    cols = [
        c
        for c in feature_columns
        if c in reference.columns and c in current.columns and reference[c].notna().any() and current[c].notna().any()
    ]
    # params.yaml's drift_psi_threshold was declared but never actually
    # wired to a stattest before — explicitly use PSI (rather than
    # Evidently's default per-column-type heuristic test) so that config
    # value has a real effect.
    report = Report(metrics=[DataDriftPreset(stattest="psi", stattest_threshold=DRIFT_PSI_THRESHOLD)])
    report.run(reference_data=reference[cols], current_data=current[cols])

    try:
        drift_result = report.as_dict()["metrics"][0]["result"]
        DRIFT_SHARE.set(drift_result["share_of_drifted_columns"])
        DRIFT_DETECTED.set(1 if drift_result["dataset_drift"] else 0)
    except (KeyError, IndexError):
        logger.warning("Unexpected Evidently report shape — skipping drift gauge update", exc_info=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"drift_report_{ts}.html"
    report.save_html(str(out_path))
    _prune_old_reports()
    logger.info("Drift report written -> %s", out_path)
    return str(out_path)
