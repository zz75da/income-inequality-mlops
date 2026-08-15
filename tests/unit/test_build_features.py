import json

import build_features
import pandas as pd
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
