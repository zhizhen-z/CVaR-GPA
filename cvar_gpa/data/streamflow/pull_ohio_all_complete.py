"""Pull EVERY HUC-05 daily-discharge gauge with a gap-free 2005-01-01..2024-12-31 record.

All catalog-span gauges are pulled (helpers in usgs_nwis.py) and every one with all 7305 days present is kept.
Output: ohio_all_complete_dv_2005_2024.npz with keys discharge (7305, d) float32 raw ft^3/s, time_days, site_id,
site_name, site_lat, site_lon, drain_area_km2, hill_alpha, median_iqr (reference statistics, not applied).
select_smallest_64.py then builds the dataset of the paper.
"""
import os, sys, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import usgs_nwis as P
OUT = os.path.join(HERE, "ohio_all_complete_dv_2005_2024.npz")
cat = P.get_huc05_gauges().sort_values("drain_area_km2").reset_index(drop=True)
print(f"catalog-span gauges with drainage area: {len(cat)}", flush=True)
dates = pd.date_range(P.START, P.END, freq="D"); assert len(dates) == P.NEED_DAYS
kept_sites, kept_series, kept_meta, n_gap, n_err = [], [], [], 0, 0
for i, row in cat.iterrows():
    s = row["site_no"]
    try: d = P.pull_dv(s)
    except Exception as e: n_err += 1; print(f"    {s}: ERR {e}", flush=True); continue
    if d is None: n_err += 1; continue
    s_re = d["val"].reindex(dates); missing = int(s_re.isna().sum())
    if missing > 0: n_gap += 1
    else:
        kept_sites.append(s); kept_series.append(s_re.values.astype(np.float32)); kept_meta.append(row)
    if (i + 1) % 25 == 0: print(f"[{i+1}/{len(cat)}] kept {len(kept_sites)}, gaps {n_gap}, errors {n_err}", flush=True)
    time.sleep(0.15)
print(f"done: kept {len(kept_sites)} complete gauges of {len(cat)} (gaps {n_gap}, errors {n_err})", flush=True)
raw = np.stack(kept_series, axis=1)
med = np.median(raw, axis=0); iqr = np.subtract(*np.percentile(raw, [75, 25], axis=0))
alphas = np.array([P.hill_alpha(raw[:, j]) if hasattr(P, "hill_alpha") else np.nan for j in range(raw.shape[1])], dtype=np.float32)
meta = pd.DataFrame(kept_meta)
np.savez(OUT, discharge=raw, time_days=((dates - pd.Timestamp("1970-01-01")).days.values).astype(np.int64),
         site_id=np.array(kept_sites), site_name=meta["station_nm"].values.astype(str),
         site_lat=pd.to_numeric(meta["dec_lat_va"], errors="coerce").values, site_lon=pd.to_numeric(meta["dec_long_va"], errors="coerce").values,
         drain_area_km2=meta["drain_area_km2"].values.astype(np.float64), hill_alpha=alphas, median_iqr=np.stack([med, iqr], axis=1))
print(f"wrote {OUT}: discharge {raw.shape}, areas {meta['drain_area_km2'].min():.1f}-{meta['drain_area_km2'].max():.0f} km2, hill median {np.nanmedian(alphas):.2f}", flush=True)
