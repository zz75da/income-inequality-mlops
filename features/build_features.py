"""
Turn data/processed/merged_panel.csv into the model-ready feature table at
data/processed/features.csv, following the feature list and cleaning rules in
params.yaml.

Steps:
  1. Drop countries with fewer than params.data.min_years_required rows.
  2. Restrict to params.data.start_year..end_year.
  3. Median-impute numeric features per country, then globally for any
     country with 100% missing on a given feature (keeps tree-based models
     from choking on NaN-only columns; XGBoost otherwise handles NaN natively
     via its own split logic, but the classifier target-encoding step needs
     dense categoricals).
  4. Label-encode `region` and `income_group_lag1` as categorical codes
     (stored alongside human-readable labels for the Streamlit dashboard).
  5. Write features.csv with all raw target/feature columns intact — the
     train-api scripts pick their own X/y slices from params.yaml so this
     file is shared across all three targets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("features.build")


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    data_cfg = params["data"]
    feat_cfg = params["features"]

    panel = pd.read_csv(PROCESSED_DIR / "merged_panel.csv")
    panel = panel[(panel["year"] >= data_cfg["start_year"]) & (panel["year"] <= data_cfg["end_year"])]

    counts = panel.groupby("country_code")["year"].nunique()
    keep_countries = counts[counts >= data_cfg["min_years_required"]].index
    dropped = set(panel["country_code"].unique()) - set(keep_countries)
    if dropped:
        logger.info(
            "Dropping %d countries with < %d years of data: %s",
            len(dropped),
            data_cfg["min_years_required"],
            sorted(dropped)[:20],
        )
    panel = panel[panel["country_code"].isin(keep_countries)].copy()

    missing_numeric = set(feat_cfg["numeric_features"]) - set(panel.columns)
    if missing_numeric:
        logger.warning(
            "Configured numeric features not present in merged panel (upstream ingestion may have failed) — "
            "adding as all-NaN so downstream column selection doesn't KeyError: %s",
            missing_numeric,
        )
        for col in missing_numeric:
            panel[col] = pd.NA

    # Per-country median impute, then global median for still-missing values.
    # A column that's NaN everywhere (e.g. a source that failed ingestion)
    # stays NaN here — XGBoost handles NaN natively via its own split logic.
    for col in feat_cfg["numeric_features"]:
        panel[col] = panel.groupby("country_code")[col].transform(lambda s: s.fillna(s.median()))
        panel[col] = panel[col].fillna(panel[col].median())

    # Categorical encodings. The category -> code mapping is persisted to
    # data/artifacts/categorical_mappings.json so predict-api can encode a
    # single incoming request the exact same way training data was encoded
    # (pandas' .cat.codes order isn't guaranteed stable across separate runs).
    mappings: dict[str, dict[str, int]] = {}
    for col in feat_cfg["categorical_features"]:
        if col not in panel.columns:
            logger.warning("Categorical feature %s missing from merged panel — skipping encoding", col)
            continue
        panel[col] = panel[col].fillna("UNKNOWN").astype(str)
        categories = sorted(panel[col].unique())
        code_map = {cat: i for i, cat in enumerate(categories)}
        mappings[col] = code_map
        panel[col + "_code"] = panel[col].map(code_map)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mappings_path = ARTIFACTS_DIR / "categorical_mappings.json"
    with open(mappings_path, "w") as f:
        json.dump(mappings, f, indent=2)
    logger.info("Wrote categorical mappings -> %s", mappings_path)

    out_path = PROCESSED_DIR / "features.csv"
    panel.to_csv(out_path, index=False)
    logger.info("Wrote %d rows x %d cols -> %s", len(panel), len(panel.columns), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
