"""
Pull the EU-SILC-derived Gini coefficient (ilc_di12) from Eurostat's public
JSON-stat REST API (no key required).

Docs: https://wikis.ec.europa.eu/display/EUROSTATHELP/API+-+detailed+guidelines

    GET https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?format=JSON&lang=EN

Note — this is deliberately NOT the EU-SILC household microdata referenced at
https://ec.europa.eu/eurostat/web/microdata/collections-research/european-union-statistics-on-income-and-living-conditions .
That microdata is only available under a research-purposes data access
contract with Eurostat (individual/household-level rows are not exposed via
the free API). This project therefore uses only Eurostat's published
*aggregate* indicators (Gini, income quintile share ratio S80/S20, at-risk-
of-poverty rate) and substitutes the World Bank's income-group classification
(see ingest_worldbank.py) for the "individual income bracket" style
classification target, rather than a genuinely per-household bracket, which
would require that restricted microdata.
"""
from __future__ import annotations

import logging

import pandas as pd

from common import get_with_retry, write_long_csv

logger = logging.getLogger("ingestion.eurostat")

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

DATASETS = {
    "ilc_di12": "gini_index_eurostat",              # Gini coefficient of equivalised disposable income
    "ilc_di11": "income_quintile_share_ratio",       # S80/S20 ratio
    "ilc_li02": "at_risk_of_poverty_rate",
}


def jsonstat_to_df(payload: dict, value_name: str) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 response into a long DataFrame."""
    dim_ids = payload["id"]
    sizes = payload["size"]
    dims = payload["dimension"]

    # Build ordered label lists per dimension, respecting each dimension's own index map.
    dim_value_lists = []
    for dim_id in dim_ids:
        cat = dims[dim_id]["category"]
        index_map = cat.get("index")
        labels = cat.get("label", {})
        if isinstance(index_map, dict):
            ordered_codes = sorted(index_map, key=lambda k: index_map[k])
        else:  # some responses give index as a list already in order
            ordered_codes = list(index_map) if index_map else list(labels)
        dim_value_lists.append([(code, labels.get(code, code)) for code in ordered_codes])

    values = payload["value"]  # dict {flat_index: value} or list
    rows = []

    def flat_index(coords: list[int]) -> int:
        idx = 0
        for pos, size in zip(coords, sizes):
            idx = idx * size + pos
        return idx

    # Iterate the full cartesian product (Eurostat responses are usually small
    # once filtered to one indicator) — geo x time is typically a few thousand cells.
    import itertools

    ranges = [range(s) for s in sizes]
    for coords in itertools.product(*ranges):
        fi = flat_index(list(coords))
        val = values.get(str(fi)) if isinstance(values, dict) else (values[fi] if fi < len(values) else None)
        if val is None:
            continue
        row = {}
        for dim_id, pos in zip(dim_ids, coords):
            code, label = dim_value_lists[dim_ids.index(dim_id)][pos]
            row[dim_id] = code
            row[dim_id + "_label"] = label
        row["value"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df["indicator"] = value_name
    return df


def fetch_dataset(dataset_code: str, value_name: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{dataset_code}"
    params = {"format": "JSON", "lang": "EN"}
    resp = get_with_retry(url, params=params)
    df = jsonstat_to_df(resp.json(), value_name)
    if df.empty:
        logger.warning("eurostat/%s parsed to 0 rows", dataset_code)
        return df

    df = df.rename(columns={"geo": "country_code", "geo_label": "country_name", "time": "year"})
    keep = [c for c in ["country_code", "country_name", "year", "value"] if c in df.columns]
    df = df[keep].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["indicator"] = value_name
    df["source"] = "eurostat"
    return df.dropna(subset=["year", "value"])


def main() -> None:
    frames = [fetch_dataset(code, name) for code, name in DATASETS.items()]
    long_df = pd.concat(frames, ignore_index=True)
    write_long_csv(long_df, "eurostat")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
