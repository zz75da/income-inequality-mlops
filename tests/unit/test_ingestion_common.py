import pandas as pd
import pytest

from common import LONG_COLUMNS, write_long_csv


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
