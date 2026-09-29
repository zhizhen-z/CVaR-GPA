"""USGS NWIS helpers for the Ohio River basin (HUC-05) daily-discharge data, 2005-01-01 to 2024-12-31.

- get_huc05_gauges(): surface-water gauges with parameterCd=00060 (discharge, ft^3/s) and their drainage areas.
- pull_dv(site): the USGS `dv` (daily values) series, mean statistic (statCd=00003), exactly as published:
  no log transform, no gap filling, no standardization.
- hill_alpha(x): Hill estimate of the tail exponent on the raw series.
Used by pull_ohio_all_complete.py.
"""
import os, sys, time, json, io
import numpy as np, requests, pandas as pd

BASE_SITE = "https://nwis.waterservices.usgs.gov/nwis/site/"
BASE_DV   = "https://nwis.waterservices.usgs.gov/nwis/dv/"

START = "2005-01-01"
END   = "2024-12-31"
NEED_DAYS = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1  # 7305

def _fetch_rdb(params):
    r = requests.get(BASE_SITE, params=params, timeout=90); r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if not ln.startswith("#")]
    if len(lines) < 3: return pd.DataFrame()
    hdr = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[2:] if ln.strip()]
    return pd.DataFrame(rows, columns=hdr)

def get_huc05_gauges():
    """Fetch all HUC-05 surface-water gauges with active 00060 dv data 2005-2024.

    Two RDB queries and a join, because USGS site service doesn't return
    both drain_area_va AND begin/end_date in one call.
    """
    # 1) series catalog (has begin_date/end_date/parm_cd/stat_cd)
    ser = _fetch_rdb(dict(huc="05", parameterCd="00060", format="rdb",
                          hasDataTypeCd="dv", siteType="ST", siteStatus="all",
                          outputDataTypeCd="dv", startDt=START, endDt=END,
                          seriesCatalogOutput="true"))
    ser = ser[(ser["parm_cd"] == "00060") & (ser["stat_cd"].astype(str) == "00003")]
    ser["begin_date"] = pd.to_datetime(ser["begin_date"], errors="coerce")
    ser["end_date"]   = pd.to_datetime(ser["end_date"],   errors="coerce")
    ok = (ser["begin_date"] <= pd.Timestamp(START)) & (ser["end_date"] >= pd.Timestamp(END))
    ser = ser[ok].drop_duplicates(subset=["site_no"])[["site_no", "begin_date", "end_date"]].reset_index(drop=True)
    print(f"  series-catalog gauges with full-period dv: {len(ser)}")

    # 2) expanded site info (has drain_area_va + name + lat/lon)
    exp = _fetch_rdb(dict(huc="05", parameterCd="00060", format="rdb",
                          hasDataTypeCd="dv", siteType="ST", siteStatus="all",
                          siteOutput="expanded"))
    keep = ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "drain_area_va"]
    exp = exp[[c for c in keep if c in exp.columns]].drop_duplicates("site_no")
    exp["drain_area_km2"] = pd.to_numeric(exp["drain_area_va"], errors="coerce") * 2.58999
    exp = exp.dropna(subset=["drain_area_km2"])
    print(f"  expanded site info with drainage area: {len(exp)}")

    df = ser.merge(exp, on="site_no", how="inner").reset_index(drop=True)
    return df

def pull_dv(site_no):
    """Pull daily-mean discharge for one site in ft^3/s."""
    params = dict(sites=site_no, parameterCd="00060", startDT=START, endDT=END,
                  format="json", statCd="00003")  # 00003 = daily mean
    r = requests.get(BASE_DV, params=params, timeout=120)
    r.raise_for_status()
    j = r.json()
    ts_list = j["value"]["timeSeries"]
    if not ts_list:
        return None
    values = ts_list[0]["values"][0]["value"]
    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["dateTime"]).dt.tz_localize(None).dt.normalize()
    df["val"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[df["val"] < 0, "val"] = np.nan   # -999999 sentinel
    df = df[["date", "val"]].drop_duplicates("date").set_index("date")
    return df

def hill_alpha(x, k_frac=0.05):
    """Hill tail index estimator on the top k_frac of positive values."""
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 50: return np.nan
    xs = np.sort(x)
    k = max(20, int(len(xs) * k_frac))
    top = xs[-k:]; thresh = xs[-k-1]
    if thresh <= 0: return np.nan
    return float(1.0 / np.mean(np.log(top / thresh)))
