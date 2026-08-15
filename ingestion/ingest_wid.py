"""
Pull top-10% and bottom-50% pre-tax national income shares from the World
Inequality Database (WID.world) API.

    GET https://wid.world/api/v3.php
        ?method=getData
        &country=<comma-separated ISO2 codes, or "all">
        &indicators=sptinc992j        (share of pre-tax national income, per adult, individuals)
        &perc=p90p100,p0p50           (top 10% and bottom 50% percentile groups)
        &years=1990:2024
        &format=csv

WID uses ISO2 country codes (plus synthetic regional codes like "QF" for
France's historical series); we convert to ISO3 via pycountry so this project
can join on the same key as World Bank/OECD/Eurostat data. Rows that don't
map to a real ISO3 (regional aggregates, "QM"/"XX"-style WID-specific codes)
are dropped — those are supra-national aggregates, not countries, and aren't
useful features for a per-country model anyway.

KNOWN ISSUE (as of 2026-08): the public `/api/v3.php` endpoint documented
above 404s. WID.world's site and R package (`wid-r-tool`) now pull from a
private AWS API Gateway endpoint that requires a server-issued API key not
obtainable through any public signup flow — there is no drop-in public
replacement. Because of this, this script is treated as best-effort:
train-api and the Airflow DAG both tolerate its failure (same as
ingest_gdim.py) and merge_sources.py/build_features.py already handle the
resulting missing top10_income_share/bottom50_income_share columns via
per-country/global median imputation. If WID.world ever republishes a public
CSV bulk-download or reinstates the v3 API, point API_URL at it and this
script starts working again with no other changes needed.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
import pycountry

from common import get_with_retry, write_long_csv

logger = logging.getLogger("ingestion.wid")

API_URL = "https://wid.world/api/v3.php"
INDICATOR = "sptinc992j"
PERCENTILES = {"p90p100": "top10_income_share", "p0p50": "bottom50_income_share"}
START_YEAR = 1990
END_YEAR = 2024


def iso2_to_iso3(iso2: str) -> str | None:
    try:
        country = pycountry.countries.get(alpha_2=iso2)
        return country.alpha_3 if country else None
    except (LookupError, AttributeError):
        return None


def fetch_percentile(perc_code: str, feature_name: str) -> pd.DataFrame:
    params = {
        "method": "getData",
        "country": "all",
        "indicators": INDICATOR,
        "perc": perc_code,
        "years": f"{START_YEAR}:{END_YEAR}",
        "format": "csv",
    }
    headers = {}
    api_key = os.getenv("WID_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = get_with_retry(API_URL, params=params, headers=headers)
    from io import StringIO

    raw = pd.read_csv(StringIO(resp.text), sep=";" if ";" in resp.text.splitlines()[0] else ",")
    # WID CSV export commonly uses columns: country;variable;percentile;year;value;age;pop
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "country" not in raw.columns or "value" not in raw.columns or "year" not in raw.columns:
        logger.warning("Unexpected WID CSV shape for perc=%s: columns=%s", perc_code, list(raw.columns))
        return pd.DataFrame()

    raw["country_code"] = raw["country"].map(iso2_to_iso3)
    raw = raw.dropna(subset=["country_code"])
    raw["indicator"] = feature_name
    raw["source"] = "wid"
    raw["country_name"] = raw["country_code"].map(
        lambda c: (pycountry.countries.get(alpha_3=c).name if pycountry.countries.get(alpha_3=c) else c)
    )
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    return raw[["country_code", "country_name", "year", "indicator", "value", "source"]].dropna(subset=["year", "value"])


def main() -> None:
    frames = [fetch_percentile(code, name) for code, name in PERCENTILES.items()]
    long_df = pd.concat(frames, ignore_index=True)
    write_long_csv(long_df, "wid")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
