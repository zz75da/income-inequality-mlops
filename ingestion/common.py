"""
Shared helpers for the ingestion/ scripts.

All four ingestion scripts (World Bank, OECD, WID, Eurostat) write a single
long-format CSV to data/raw/<source>.csv with the columns:

    country_code (ISO3), country_name, year, indicator, value, source

merge_sources.py pivots and joins these into the wide feature table consumed
by features/build_features.py.

Verified end-to-end against the live APIs (2026-08) — see the README's
Known Limitations for the couple of source-specific gaps that turned up
(WID.world's public API being retired, Eurostat's non-ISO3 country codes).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pycountry
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LONG_COLUMNS = ["country_code", "country_name", "year", "indicator", "value", "source"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_with_retry(
    url: str, params: dict | None = None, max_retries: int = 3, backoff: float = 2.0, **kwargs
) -> requests.Response:
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
            time.sleep(backoff**attempt)
    raise RuntimeError(f"GET {url} failed after {max_retries} attempts") from last_exc


def iso2_to_iso3(code: str, overrides: dict[str, str] | None = None) -> str | None:
    """Convert an ISO 3166-1 alpha-2 country code to alpha-3 via pycountry, so
    every source can join on the same country_code key.

    `overrides` covers codes a source uses that aren't standard ISO alpha-2 in
    the first place — e.g. Eurostat's `geo` dimension uses "EL" for Greece and
    "UK" for the United Kingdom instead of the real ISO codes "GR"/"GB".
    Returns None for anything that doesn't map to a real country (regional
    aggregates like WID's "QF" or Eurostat's "EU27_2020") — callers should
    drop those rows rather than feed a fabricated code into the panel.
    """
    if overrides and code in overrides:
        return overrides[code]
    try:
        country = pycountry.countries.get(alpha_2=code)
        return country.alpha_3 if country else None
    except (LookupError, AttributeError):
        return None


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
