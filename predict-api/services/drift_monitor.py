"""
Lightweight Evidently-based drift monitor for predict-api, adapted from
rakuten_mlops_services' buffer-then-report pattern but simplified for a
low-QPS tabular API (portfolio-project traffic, not production scale):
buffers incoming feature rows, and once params.yaml's
monitoring.drift_reference_min_rows is reached compares the buffer against a
reference sample built from data/processed/features.csv at startup.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("predict-api.drift")

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "data" / "artifacts" / "drift_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

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
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        logger.warning("evidently not installed — skipping drift report generation")
        return None

    cols = [c for c in feature_columns if c in reference.columns and c in current.columns]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[cols], current_data=current[cols])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"drift_report_{ts}.html"
    report.save_html(str(out_path))
    _prune_old_reports()
    logger.info("Drift report written -> %s", out_path)
    return str(out_path)
