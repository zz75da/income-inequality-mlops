"""
Shared helpers for the ingestion/ scripts.

All four ingestion scripts (World Bank, OECD, WID, Eurostat) write a single
long-format CSV to data/raw/<source>.csv with the columns:

    country_code (ISO3), country_name, year, indicator, value, source

merge_sources.py pivots and joins these into the wide feature table consumed
by features/build_features.py.

Note on connectivity: this sandbox environment cannot reach external hosts
(api.worldbank.org, sdmx.oecd.org, ec.europa.eu, wid.world are all outside the
network allowlist here), so these scripts are written against each source's
documented, stable public API contract but have not been executed end-to-end
in this environment. Run them from your own machine / CI, where they should
work as-is; if an API shape has drifted since this was written, the error
messages below point at exactly which request failed.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LONG_COLUMNS = ["country_code", "country_name", "year", "indicator", "value", "source"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_with_retry(url: str, params: dict | None = None, max_retries: int = 3, backoff: float = 2.0, **kwargs) -> requests.Response:
    """GET with simple exponential backoff — public stat APIs rate-limit aggressively."""
    logger = logging.getLogger("ingestion.http")
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("GET %s failed (attempt %d/%d): %s", url, attempt, max_retries, exc)
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"GET {url} failed after {max_retries} attempts") from last_exc


def write_long_csv(df: pd.DataFrame, source_name: str) -> Path:
    """Validate the long-format schema and write data/raw/<source_name>.csv."""
    missing = set(LONG_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{source_name}: missing required columns {missing}")
    df = df[LONG_COLUMNS].dropna(subset=["value"])
    out_path = RAW_DIR / f"{source_name}.csv"
    df.to_csv(out_path, index=False)
    logging.getLogger("ingestion").info("Wrote %d rows -> %s", len(df), out_path)
    return out_path
