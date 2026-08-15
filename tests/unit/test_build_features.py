import json

import build_features
import pandas as pd
import pytest
import yaml


def _write_params(root, min_years=2):
    params = {
        "data": {"start_year": 2015, "end_year": 2024, "min_years_required": min_years},
        "features": {
            "numeric_features": ["gdp_per_capita_ppp"],
            "categorical_features": ["region"],
            "target_gini": "gini_index",
            "target_mobility": "intergen_income_elasticity",
            "target_income_group": "income_group",
        },
    }
    (root / "params.yaml").write_text(yaml.dump(params))
    return params


def test_build_features_drops_sparse_countries_and_imputes(tmp_path, monkeypatch):
    root = tmp_path
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    artifacts = root / "data" / "artifacts"

    _write_params(root)

    panel = pd.DataFrame(
        {
            "country_code": ["FRA", "FRA", "FRA", "TUV"],  # TUV has only 1 year -> dropped (min_years=2)
            "year": [2015, 2016, 2017, 2018],
            "gdp_per_capita_ppp": [40000, None, 42000, 5000],
            "region": ["Europe", "Europe", None, "Pacific"],
        }
    )
    panel.to_csv(processed / "merged_panel.csv", index=False)

    monkeypatch.setattr(build_features, "ROOT", root)
    monkeypatch.setattr(build_features, "PROCESSED_DIR", processed)
    monkeypatch.setattr(build_features, "ARTIFACTS_DIR", artifacts)

    build_features.main()

    out = pd.read_csv(processed / "features.csv")
    assert set(out["country_code"]) == {"FRA"}
    assert out["gdp_per_capita_ppp"].isna().sum() == 0  # imputed
    assert "region_code" in out.columns

    mapping_path = artifacts / "categorical_mappings.json"
    assert mapping_path.exists()
    mappings = json.loads(mapping_path.read_text())
    assert "region" in mappings

    medians_path = artifacts / "feature_medians.json"
    assert medians_path.exists()
    medians = json.loads(medians_path.read_text())
    assert medians["gdp_per_capita_ppp"] == pytest.approx(41000.0)  # median of [40000, 42000]


def test_build_features_drops_only_invalid_rows(tmp_path, monkeypatch):
    """A schema-invalid row (bad year, out-of-range Gini) is dropped without
    aborting the whole run, matching the pipeline's existing degrade-gracefully
    pattern for missing data sources elsewhere."""
    root = tmp_path
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    artifacts = root / "data" / "artifacts"

    _write_params(root, min_years=1)

    panel = pd.DataFrame(
        {
            "country_code": ["FRA", "FRA", "DEU"],
            "year": [2015, 1800, 2016],  # 1800 is out of [2015, 2024] -> dropped
            "gini_index": [30.0, 31.0, 250.0],  # 250 is out of [0, 100] -> dropped
            "gdp_per_capita_ppp": [40000, 41000, 42000],
            "region": ["Europe", "Europe", "Europe"],
        }
    )
    panel.to_csv(processed / "merged_panel.csv", index=False)

    monkeypatch.setattr(build_features, "ROOT", root)
    monkeypatch.setattr(build_features, "PROCESSED_DIR", processed)
    monkeypatch.setattr(build_features, "ARTIFACTS_DIR", artifacts)

    build_features.main()

    out = pd.read_csv(processed / "features.csv")
    assert len(out) == 1
    assert out.iloc[0]["country_code"] == "FRA"
    assert out.iloc[0]["year"] == 2015
