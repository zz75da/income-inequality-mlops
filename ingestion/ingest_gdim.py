"""
Load intergenerational income mobility data from the World Bank's Global
Database on Intergenerational Mobility (GDIM).

GDIM has no REST API — the World Bank distributes it as a downloadable
spreadsheet from:
    https://www.worldbank.org/en/topic/poverty/brief/global-database-on-intergenerational-mobility

This is a *manual* step:
    1. Download the latest GDIM file (an .xlsx, historically named e.g.
       "GDIMMay2018.xlsx" — check the page above for the current filename).
    2. Save it as data/raw/gdim_manual_download.xlsx
    3. Run this script to normalize it into the project's long format.

If the file is missing, main() logs instructions and exits without error so
the rest of the pipeline (Gini + income-group targets) can still run —
merge_sources.py treats the mobility target as optional and will simply
leave intergen_income_elasticity as NaN for every row when this file is
absent, which XGBoost handles natively.

Column expectations follow the GDIM public codebook: a `wbcode` (ISO3)
column, a `birth_cohort` column, and mobility measures including
`icm_transmission` (intergenerational income elasticity, IGE) and rank-rank
correlation `icm_rank`. Adjust COLUMN_MAP below if a newer GDIM release
renames these.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common import RAW_DIR

logger = logging.getLogger("ingestion.gdim")

MANUAL_FILE = RAW_DIR / "gdim_manual_download.xlsx"
OUT_COLUMNS = ["country_code", "country_name", "intergen_income_elasticity", "intergen_rank_correlation"]

COLUMN_MAP = {
    "wbcode": "country_code",
    "country": "country_name",
    "birth_cohort": "birth_cohort",
    "icm_transmission": "intergen_income_elasticity",
    "icm_rank": "intergen_rank_correlation",
}


def main() -> None:
    out_path = RAW_DIR / "gdim_mobility.csv"
    if not MANUAL_FILE.exists():
        logger.warning(
            "GDIM file not found at %s — download it manually from "
            "https://www.worldbank.org/en/topic/poverty/brief/global-database-on-intergenerational-mobility "
            "and re-run this script. Writing an empty %s for now (merge_sources.py "
            "will leave intergen_income_elasticity as NaN).",
            MANUAL_FILE, out_path,
        )
        pd.DataFrame(columns=OUT_COLUMNS).to_csv(out_path, index=False)
        return

    df = pd.read_excel(MANUAL_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    available = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if "wbcode" not in available and "country_code" not in df.columns:
        raise ValueError(
            f"GDIM file at {MANUAL_FILE} doesn't have the expected 'wbcode' column "
            f"(found: {list(df.columns)}). Update COLUMN_MAP in ingest_gdim.py for this release."
        )
    df = df.rename(columns=available)

    # GDIM reports one row per birth cohort per country; keep the most recent
    # cohort per country as a single representative mobility value.
    if "birth_cohort" in df.columns:
        df = df.sort_values("birth_cohort").groupby("country_code", as_index=False).last()

    out_cols = [c for c in OUT_COLUMNS if c in df.columns]
    out = df[out_cols].dropna(subset=["country_code"])
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(out), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
