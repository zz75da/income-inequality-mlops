"""
Pull the OECD Income Distribution Database (IDD) Gini index via the OECD.Stat
SDMX REST API (public, no key required).

Source (browsable): https://data-explorer.oecd.org/vis?fs[0]=Topic,1%7CSociety%23SOC%23%7CInequality%23SOC_INE%23&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_WISE_IDD%40DF_IDD&df[ag]=OECD.WISE.INE&df[vs]=1.0

That UI maps to the SDMX REST endpoint:
    https://sdmx.oecd.org/public/rest/data/{agency},{dataflow},{version}/{filter}

  agency    = OECD.WISE.INE
  dataflow  = DSD_WISE_IDD@DF_IDD
  version   = 1.0
  filter    = .A.INC_DISP_GINI..._T.METH2012.D_CUR.
              (annual frequency, disposable-income Gini, total population,
               2012 methodology, current definition — matches the filters
               selected in the Data Explorer link above)

Response format is SDMX-JSON. OECD's SDMX-JSON payload nests observations
under dataSets[0].series[<dim-key>].observations[<obs-index>], where <dim-key>
and <obs-index> are colon-joined positional indices into
structure.dimensions.series / structure.dimensions.observation. This module
includes a small generic SDMX-JSON -> long DataFrame flattener since that
shape is common to every OECD dataflow (swap FILTER/DATAFLOW to reuse it for
other OECD inequality series, e.g. wealth Gini, poverty rates).
"""
from __future__ import annotations

import logging

import pandas as pd

from common import get_with_retry, write_long_csv

logger = logging.getLogger("ingestion.oecd")

SDMX_BASE = "https://sdmx.oecd.org/public/rest/data"
AGENCY_DATAFLOW = "OECD.WISE.INE,DSD_WISE_IDD@DF_IDD,1.0"
FILTER = ".A.INC_DISP_GINI..._T.METH2012.D_CUR."  # annual, Gini, total pop, 2012 methodology
START_PERIOD = 1990


def flatten_sdmx_json(payload: dict) -> pd.DataFrame:
    """Generic SDMX-JSON (v2 'jsondata') series/observations -> long DataFrame."""
    dataset = payload["data"]["dataSets"][0]
    structure = payload["data"]["structure"]
    series_dims = structure["dimensions"]["series"]
    obs_dims = structure["dimensions"]["observation"]

    rows = []
    for series_key, series_val in dataset.get("series", {}).items():
        dim_indices = [int(i) for i in series_key.split(":")]
        dim_values = {
            dim["id"]: dim["values"][idx]["id"]
            for dim, idx in zip(series_dims, dim_indices)
        }
        dim_labels = {
            dim["id"] + "_label": dim["values"][idx]["name"]
            for dim, idx in zip(series_dims, dim_indices)
        }
        for obs_key, obs_val in series_val.get("observations", {}).items():
            obs_index = int(obs_key.split(":")[0])
            time_value = obs_dims[0]["values"][obs_index]["id"]
            value = obs_val[0] if isinstance(obs_val, list) else obs_val
            row = {**dim_values, **dim_labels, "TIME_PERIOD": time_value, "value": value}
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_gini() -> pd.DataFrame:
    url = f"{SDMX_BASE}/{AGENCY_DATAFLOW}/{FILTER}"
    params = {"format": "jsondata", "startPeriod": START_PERIOD, "dimensionAtObservation": "TIME_PERIOD"}
    resp = get_with_retry(url, params=params, headers={"Accept": "application/vnd.sdmx.data+json"})
    df = flatten_sdmx_json(resp.json())
    if df.empty:
        logger.warning("OECD SDMX response parsed to 0 rows — check FILTER string against the current dataflow structure")
        return df

    df = df.rename(columns={"REF_AREA": "country_code", "REF_AREA_label": "country_name"})
    df["year"] = df["TIME_PERIOD"].astype(int)
    df["indicator"] = "gini_index_oecd"
    df["source"] = "oecd"
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["country_code", "country_name", "year", "indicator", "value", "source"]]


def main() -> None:
    df = fetch_gini()
    write_long_csv(df, "oecd")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
