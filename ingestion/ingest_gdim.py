"""
Load intergenerational mobility data from the World Bank's Global Database on
Intergenerational Mobility (GDIM).

GDIM has no query-style REST API, but — unlike WID.world (see ingest_wid.py) —
the World Bank Data Catalog *does* serve the dataset as a stable, public,
unauthenticated file download, so this no longer needs a manual step:

    https://datacatalog.worldbank.org/search/dataset/0050771/global-database-on-intergenerational-mobility
    -> https://datacatalogfiles.worldbank.org/ddh-published/0050771/3/DR0065670/GDIM_2023_03.csv

If that URL ever 404s (a new GDIM release usually reuses the dataset id 0050771
but bumps the version/DR-resource id in the path — check the Data Catalog page
above for the current one), this script falls back to
data/raw/gdim_manual_download.csv if present, and otherwise degrades to an
empty output (same best-effort pattern as ingest_wid.py) rather than failing
the whole training job.

Column notes (per the GDIM public codebook): one row per country x birth
cohort x subgroup. We keep the `parent="avg"` (pooled mother/father) and
`child="all"` (pooled daughter/son) rows — the single population-level
mobility estimate per country/cohort — and use:
  - `BETA` (regression coefficient of child's outcome on parent's, i.e.
    intergenerational persistence) as our `intergen_income_elasticity` proxy
  - `COR` (rank-rank correlation) as `intergen_rank_correlation`
GDIM's outcome variable is years-of-schooling based (there's no free dataset
with this country coverage using actual income), so this is the same kind of
documented proxy as `income_group` standing in for a true income bracket —
see the README's Known Limitations.
"""

from __future__ import annotations

import logging

import pandas as pd
from common import RAW_DIR, get_with_retry

logger = logging.getLogger("ingestion.gdim")

GDIM_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0050771/3/DR0065670/GDIM_2023_03.csv"
MANUAL_FILE = RAW_DIR / "gdim_manual_download.csv"
OUT_COLUMNS = ["country_code", "country_name", "intergen_income_elasticity", "intergen_rank_correlation"]

COLUMN_MAP = {
    "code": "country_code",
    "country": "country_name",
    "cohort": "birth_cohort",
    "BETA": "intergen_income_elasticity",
    "COR": "intergen_rank_correlation",
}


def _load_raw() -> pd.DataFrame:
    """Prefer a manually-provided override; otherwise auto-download the public file."""
    if MANUAL_FILE.exists():
        logger.info("Using manually-provided GDIM file at %s", MANUAL_FILE)
        return pd.read_csv(MANUAL_FILE, low_memory=False)

    logger.info("Downloading GDIM data from %s", GDIM_URL)
    resp = get_with_retry(GDIM_URL)
    from io import StringIO

    return pd.read_csv(StringIO(resp.text), low_memory=False)


def main() -> None:
    out_path = RAW_DIR / "gdim_mobility.csv"
    try:
        df = _load_raw()
    except Exception:
        logger.warning(
            "GDIM download failed and no override at %s — writing an empty %s "
            "(merge_sources.py will leave intergen_income_elasticity as NaN). "
            "Check https://datacatalog.worldbank.org/search/dataset/0050771/ for a "
            "current download URL if this dataset has moved.",
            MANUAL_FILE,
            out_path,
            exc_info=True,
        )
        pd.DataFrame(columns=OUT_COLUMNS).to_csv(out_path, index=False)
        return

    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(
            f"GDIM data doesn't have the expected columns {missing} (found: {list(df.columns)}). "
            "The dataset schema has likely changed — update COLUMN_MAP in ingest_gdim.py."
        )

    # Keep the pooled parent/child estimate: parent="avg" (mother+father averaged),
    # child="all" (daughters+sons pooled) — the single population-level figure
    # per country/cohort, rather than the gender-split subgroup rows.
    if "parent" in df.columns and "child" in df.columns:
        df = df[(df["parent"] == "avg") & (df["child"] == "all")]

    df = df.rename(columns=COLUMN_MAP)

    # GDIM reports one row per birth cohort per country; keep the most recent
    # cohort per country as a single representative mobility value.
    df = df.dropna(subset=["country_code"])
    if "birth_cohort" in df.columns:
        df = df.sort_values("birth_cohort").groupby("country_code", as_index=False).last()

    out_cols = [c for c in OUT_COLUMNS if c in df.columns]
    out = df[out_cols]
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(out), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
