import pandas as pd
import pytest
from common import LONG_COLUMNS, iso2_to_iso3, write_long_csv


def test_write_long_csv_happy_path(tmp_path, monkeypatch):
    import common as ingestion_common

    monkeypatch.setattr(ingestion_common, "RAW_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "country_code": ["FRA", "USA"],
            "country_name": ["France", "United States"],
            "year": [2020, 2020],
            "indicator": ["gini_index", "gini_index"],
            "value": [32.4, 41.5],
            "source": ["worldbank", "worldbank"],
        }
    )
    out_path = write_long_csv(df, "worldbank")
    assert out_path.exists()
    result = pd.read_csv(out_path)
    assert list(result.columns) == LONG_COLUMNS
    assert len(result) == 2


def test_write_long_csv_missing_column_raises():
    df = pd.DataFrame({"country_code": ["FRA"], "value": [32.4]})
    with pytest.raises(ValueError, match="missing required columns"):
        write_long_csv(df, "worldbank")


def test_write_long_csv_drops_null_values(tmp_path, monkeypatch):
    import common as ingestion_common

    monkeypatch.setattr(ingestion_common, "RAW_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "country_code": ["FRA", "USA"],
            "country_name": ["France", "United States"],
            "year": [2020, 2020],
            "indicator": ["gini_index", "gini_index"],
            "value": [32.4, None],
            "source": ["worldbank", "worldbank"],
        }
    )
    out_path = write_long_csv(df, "worldbank")
    result = pd.read_csv(out_path)
    assert len(result) == 1


def test_iso2_to_iso3_standard_codes():
    assert iso2_to_iso3("FR") == "FRA"
    assert iso2_to_iso3("DE") == "DEU"


def test_iso2_to_iso3_applies_overrides():
    # Eurostat uses "EL"/"UK" instead of the real ISO codes "GR"/"GB" for
    # Greece/the United Kingdom — this bit Eurostat ingestion for real: those
    # rows silently failed to join against World Bank/OECD/WID's ISO3 keys
    # until this override was added.
    overrides = {"EL": "GRC", "UK": "GBR"}
    assert iso2_to_iso3("EL", overrides=overrides) == "GRC"
    assert iso2_to_iso3("UK", overrides=overrides) == "GBR"
    assert iso2_to_iso3("EL") is None  # not a real ISO alpha-2 code without the override


def test_iso2_to_iso3_returns_none_for_aggregates():
    assert iso2_to_iso3("EU27_2020") is None
    assert iso2_to_iso3("EA20") is None
