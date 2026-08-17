"""
Pandera schema for data/processed/merged_panel.csv — a sanity check on the
merge step's output before build_features.py trains anything on it.

Bounds are deliberately loose: the goal is to catch obviously-broken rows
(a parsing bug producing year=1800, a negative population, a Gini index
outside its mathematical [0, 100] range) rather than to enforce a tight
statistical prior on legitimate-but-extreme values (e.g. gdp_growth_pct can
genuinely be very negative during a crisis year). Year bounds are read from
params.yaml so there's one source of truth with build_features.py's own
start_year/end_year filter, not a second hardcoded copy.
"""

from __future__ import annotations

import pandera as pa


def build_schema(start_year: int, end_year: int) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "country_code": pa.Column(str, pa.Check.str_length(3, 3), nullable=False),
            "year": pa.Column(int, pa.Check.in_range(start_year, end_year), nullable=False, coerce=True),
            "gini_index": pa.Column(float, pa.Check.in_range(0, 100), nullable=True, required=False, coerce=True),
            "gdp_per_capita_ppp": pa.Column(float, pa.Check.ge(0), nullable=True, required=False, coerce=True),
            "unemployment_rate": pa.Column(
                float, pa.Check.in_range(0, 100), nullable=True, required=False, coerce=True
            ),
            "urban_population_pct": pa.Column(
                float, pa.Check.in_range(0, 100), nullable=True, required=False, coerce=True
            ),
            "population_total": pa.Column(float, pa.Check.ge(0), nullable=True, required=False, coerce=True),
            # World Bank's own published series for a handful of crisis/structurally-unusual
            # economies (Sudan 1998-99 hyperinflation, Kuwait 1991 post-Gulf War GDP collapse,
            # Timor-Leste's oil-fund-dominated GDP) puts these over 100% — a government can't
            # spend or tax more than its entire economy, so treat those specific rows as bad data
            # rather than real extremes (unlike gdp_growth_pct, where large swings are plausible).
            "social_spending_pct_gdp": pa.Column(
                float, pa.Check.in_range(0, 100), nullable=True, required=False, coerce=True
            ),
            "tax_revenue_pct_gdp": pa.Column(
                float, pa.Check.in_range(0, 100), nullable=True, required=False, coerce=True
            ),
        },
        strict=False,  # the panel has many more columns than we bother to constrain
    )
