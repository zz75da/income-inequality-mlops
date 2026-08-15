"""
Merge the four raw long-format sources (World Bank, OECD, Eurostat, WID) plus
the WB income-group classification and the (optional, manually downloaded)
GDIM mobility data into a single wide country-year panel at
data/processed/merged_panel.csv.

Gini coalescing: World Bank's SI.POV.GINI has the widest country coverage but
sparse year coverage for many countries; OECD and Eurostat fill in gaps for
their member countries with more consistent annual coverage. We coalesce in
that priority order into a single `gini_index` column and record which source
won in `gini_source`, so the model can (optionally) use `gini_source` as a
feature to account for methodological differences between the three Gini
definitions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ingestion.merge")


def load_long(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        logger.warning("%s not found — run ingestion/ingest_%s.py first. Continuing without it.", path, name)
        return pd.DataFrame(columns=["country_code", "country_name", "year", "indicator", "value", "source"])
    return pd.read_csv(path)


def main() -> None:
    long_frames = [load_long(s) for s in ("worldbank", "oecd", "eurostat", "wid")]
    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df = long_df.dropna(subset=["year"])
    long_df["year"] = long_df["year"].astype(int)

    # Pivot every non-Gini indicator straight across.
    non_gini = long_df[~long_df["indicator"].str.startswith("gini_index")]
    wide = non_gini.pivot_table(
        index=["country_code", "year"], columns="indicator", values="value", aggfunc="mean"
    ).reset_index()

    # Coalesce the three Gini variants: worldbank > oecd > eurostat.
    gini_long = long_df[long_df["indicator"].str.startswith("gini_index")]
    gini_pivot = gini_long.pivot_table(
        index=["country_code", "year"], columns="indicator", values="value", aggfunc="mean"
    ).reset_index()
    priority = ["gini_index", "gini_index_oecd", "gini_index_eurostat"]
    present = [c for c in priority if c in gini_pivot.columns]
    if present:
        gini_pivot["gini_index"] = gini_pivot[present].bfill(axis=1).iloc[:, 0]
        gini_pivot["gini_source"] = gini_pivot[present].apply(
            lambda row: next((c for c in present if pd.notna(row[c])), None), axis=1
        )
    wide = wide.merge(gini_pivot[["country_code", "year", "gini_index", "gini_source"]], on=["country_code", "year"], how="outer")

    # Country name lookup (prefer World Bank's naming).
    names = long_df.dropna(subset=["country_name"]).drop_duplicates("country_code")[["country_code", "country_name"]]
    wide = wide.merge(names, on="country_code", how="left")

    # WB income group + region (static per country -> broadcast to every year).
    wb_income_path = RAW_DIR / "worldbank_income_group_current.csv"
    if wb_income_path.exists():
        income = pd.read_csv(wb_income_path)[["country_code", "income_group", "region"]]
        wide = wide.merge(income, on="country_code", how="left")
        wide = wide.sort_values(["country_code", "year"])
        wide["income_group_lag1"] = wide.groupby("country_code")["income_group"].shift(1)
    else:
        logger.warning("%s not found — income_group classification target will be all-NaN.", wb_income_path)
        wide["income_group"] = pd.NA
        wide["income_group_lag1"] = pd.NA

    # GDIM mobility (static per country, optional).
    gdim_path = RAW_DIR / "gdim_mobility.csv"
    if gdim_path.exists():
        gdim = pd.read_csv(gdim_path)[["country_code", "intergen_income_elasticity", "intergen_rank_correlation"]]
        wide = wide.merge(gdim, on="country_code", how="left")
    else:
        logger.warning("%s not found — run ingestion/ingest_gdim.py after manually downloading GDIM. Mobility target will be all-NaN.", gdim_path)
        wide["intergen_income_elasticity"] = pd.NA
        wide["intergen_rank_correlation"] = pd.NA

    wide = wide.sort_values(["country_code", "year"]).reset_index(drop=True)
    out_path = PROCESSED_DIR / "merged_panel.csv"
    wide.to_csv(out_path, index=False)
    logger.info("Wrote %d rows x %d cols -> %s", len(wide), len(wide.columns), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
