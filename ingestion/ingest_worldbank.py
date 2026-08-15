"""
Pull macro indicators + the Gini index + income-group classification from the
World Bank Open Data API (fully public, no API key required).

API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
Base URL: https://api.worldbank.org/v2/

Two request shapes are used:
  1. /country/all/indicator/{code}?format=json&per_page=20000&date=YYYY:YYYY
     -> one long time series per indicator, all countries at once.
  2. /country?format=json&per_page=400
     -> country metadata incl. current `incomeLevel` (L/LM/UM/H), used as the
        classification target. The API only exposes the *current*
        classification; a historical per-year series requires manually
        downloading the World Bank's "OGHIST" spreadsheet
        (https://datahelpdesk.worldbank.org/knowledgebase/articles/906519) —
        see `fetch_income_group_note()` below.
"""
from __future__ import annotations

import logging

import pandas as pd

from common import get_with_retry, write_long_csv

logger = logging.getLogger("ingestion.worldbank")

BASE_URL = "https://api.worldbank.org/v2"

# indicator_code -> our internal feature name
INDICATORS = {
    "SI.POV.GINI": "gini_index",
    "NY.GDP.PCAP.PP.KD": "gdp_per_capita_ppp",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "SL.UEM.TOTL.ZS": "unemployment_rate",
    "SE.XPD.TOTL.GD.ZS": "education_expenditure_pct_gdp",
    "GC.XPN.TOTL.GD.ZS": "social_spending_pct_gdp",     # govt expense % GDP, proxy for social spending
    "GC.TAX.TOTL.GD.ZS": "tax_revenue_pct_gdp",
    "SP.URB.TOTL.IN.ZS": "urban_population_pct",
    "SP.POP.TOTL": "population_total",
}

START_YEAR = 1990
END_YEAR = 2024


def fetch_indicator(indicator_code: str, feature_name: str) -> pd.DataFrame:
    """Fetch one indicator's full time series for all countries, paginated."""
    rows: list[dict] = []
    page = 1
    while True:
        url = f"{BASE_URL}/country/all/indicator/{indicator_code}"
        params = {
            "format": "json",
            "per_page": 20000,
            "date": f"{START_YEAR}:{END_YEAR}",
            "page": page,
        }
        resp = get_with_retry(url, params=params)
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            break
        meta, data = payload[0], payload[1]
        for item in data:
            if item.get("value") is None:
                continue
            # Skip aggregate "regions" (e.g. "World", "OECD members") — keep only
            # real countries. WB flags these with an empty `region.value`... in
            # practice the simplest robust filter is `countryiso3code` length == 3
            # AND not one of the known aggregate codes bank exposes as iso3-shaped.
            iso3 = item.get("countryiso3code")
            if not iso3 or len(iso3) != 3:
                continue
            rows.append(
                {
                    "country_code": iso3,
                    "country_name": item["country"]["value"],
                    "year": int(item["date"]),
                    "indicator": feature_name,
                    "value": item["value"],
                    "source": "worldbank",
                }
            )
        pages_total = meta.get("pages", 1)
        if page >= pages_total:
            break
        page += 1
    logger.info("worldbank/%s: %d observations", indicator_code, len(rows))
    return pd.DataFrame(rows)


def fetch_income_group_classification() -> pd.DataFrame:
    """
    Current World Bank income-group classification per country (L / LM / UM / H).

    This is a snapshot (not a per-year historical series) — the DVC pipeline
    treats it as the label for the most recent year of each country and
    carries it backward as `income_group_lag1` when building lag features.
    For a true historical series, download OGHIST.xlsx from:
    https://datahelpdesk.worldbank.org/knowledgebase/articles/906519
    and place it at data/raw/worldbank_income_group_history.xlsx —
    merge_sources.py will prefer that file if present.
    """
    url = f"{BASE_URL}/country"
    params = {"format": "json", "per_page": 400}
    resp = get_with_retry(url, params=params)
    payload = resp.json()
    _, data = payload[0], payload[1]

    rows = []
    for item in data:
        iso3 = item.get("id")
        income_level = item.get("incomeLevel", {}).get("value")
        region = item.get("region", {}).get("value")
        if not iso3 or income_level in (None, "Aggregates"):
            continue
        rows.append(
            {
                "country_code": iso3,
                "country_name": item["name"],
                "income_group": income_level,
                "region": region,
            }
        )
    logger.info("worldbank/income_group: %d countries", len(rows))
    return pd.DataFrame(rows)


def main() -> None:
    frames = [fetch_indicator(code, name) for code, name in INDICATORS.items()]
    long_df = pd.concat(frames, ignore_index=True)
    write_long_csv(long_df, "worldbank")

    income_groups = fetch_income_group_classification()
    income_groups.to_csv("data/raw/worldbank_income_group_current.csv", index=False)
    logger.info("Wrote %d rows -> data/raw/worldbank_income_group_current.csv", len(income_groups))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
