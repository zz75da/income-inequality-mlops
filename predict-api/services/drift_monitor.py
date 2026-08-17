"""
Lightweight Evidently-based drift monitor for predict-api, adapted from
rakuten_mlops_services' buffer-then-report pattern but simplified for a
low-QPS tabular API (portfolio-project traffic, not production scale):
buffers incoming feature rows, and once params.yaml's
monitoring.drift_reference_min_rows is reached compares the buffer against a
reference sample built from data/processed/features.csv at startup.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge

logger = logging.getLogger("predict-api.drift")

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "data" / "artifacts" / "drift_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

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


def record_prediction(features: dict) -> None:
    with _lock:
        _buffer.append(features)


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
    report = Report(metrics=[DataDriftPreset()])
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
