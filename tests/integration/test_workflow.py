"""
Integration test: synthetic raw source CSVs -> merge_sources -> build_features,
verifying the full offline data pipeline wiring (not the live external APIs,
which ingest_*.py scripts hit directly and which this sandbox/CI environment
cannot reach — see ingestion/common.py's module docstring).
"""
import pandas as pd
import pytest
import yaml


@pytest.mark.integration
def test_merge_and_build_features_pipeline(tmp_path, monkeypatch):
    import build_features
    import merge_sources

    root = tmp_path
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    artifacts = root / "data" / "artifacts"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)

    worldbank = pd.DataFrame(
        {
            "country_code": ["FRA", "FRA", "FRA", "USA", "USA", "USA"],
            "country_name": ["France"] * 3 + ["United States"] * 3,
            "year": [2018, 2019, 2020, 2018, 2019, 2020],
            "indicator": ["gini_index"] * 3 + ["gini_index"] * 3,
            "value": [32.4, 32.1, 31.9, 41.5, 41.2, 41.0],
            "source": ["worldbank"] * 6,
        }
    )
    worldbank.to_csv(raw / "worldbank.csv", index=False)
    pd.DataFrame(columns=["country_code", "country_name", "year", "indicator", "value", "source"]).to_csv(raw / "oecd.csv", index=False)
    pd.DataFrame(columns=["country_code", "country_name", "year", "indicator", "value", "source"]).to_csv(raw / "eurostat.csv", index=False)
    pd.DataFrame(columns=["country_code", "country_name", "year", "indicator", "value", "source"]).to_csv(raw / "wid.csv", index=False)
    pd.DataFrame({"country_code": ["FRA", "USA"], "income_group": ["High income", "High income"], "region": ["Europe", "North America"]}).to_csv(
        raw / "worldbank_income_group_current.csv", index=False
    )

    monkeypatch.setattr(merge_sources, "RAW_DIR", raw)
    monkeypatch.setattr(merge_sources, "PROCESSED_DIR", processed)
    merge_sources.main()

    merged = pd.read_csv(processed / "merged_panel.csv")
    assert len(merged) == 6
    assert "gini_index" in merged.columns
    assert merged["gini_index"].notna().all()

    params = {
        "data": {"start_year": 2015, "end_year": 2024, "min_years_required": 2},
        "features": {
            "numeric_features": ["gini_index"],
            "categorical_features": ["region"],
            "target_gini": "gini_index",
            "target_mobility": "intergen_income_elasticity",
            "target_income_group": "income_group",
        },
    }
    (root / "params.yaml").write_text(yaml.dump(params))

    monkeypatch.setattr(build_features, "ROOT", root)
    monkeypatch.setattr(build_features, "PROCESSED_DIR", processed)
    monkeypatch.setattr(build_features, "ARTIFACTS_DIR", artifacts)
    build_features.main()

    features = pd.read_csv(processed / "features.csv")
    assert set(features["country_code"]) == {"FRA", "USA"}
    assert "region_code" in features.columns
