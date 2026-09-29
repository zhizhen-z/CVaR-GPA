"""USGS Ohio-basin daily streamflow (64 gauges, 2005-2024, 7305 x 64) target loader.

load_streamflow(N_Q, N_P, random_seed, scale="area")
  Returns (X_, Y_).
  X_ : (N_Q, 64) float32, target rows: specific discharge q = Q / drainage area in mm/day. The drainage area is
       static site metadata, so no data-derived statistic enters. If N_Q < 7305, a fixed random subset is drawn.
  Y_ : (N_P, 64) float32, Gaussian sample with the per-gauge mean and standard deviation of X_.

The on-disk npz holds the USGS-published raw values in ft^3/s; the scaling is applied at load time.
"""
import os
import numpy as np

STREAM_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamflow", "ohio_smallest64_dv_2005_2024.npz")


def load_streamflow(N_Q=None, N_P=None, random_seed=0, scale="area", npz=None):
    """scale='area' : X = specific discharge q = Q/A in mm/day (the target of the paper);
                   1 ft^3/s over 1 km^2 = 2.4465755455 mm/day.
    scale='iqr'  : X = discharge / per-gauge interquartile range."""
    z = np.load(npz or STREAM_NPZ)   # npz=None -> the default 64-gauge file
    raw = z["discharge"].astype(np.float32)          # (7305, d) ft^3/s
    if scale == "area":
        denom = np.asarray(z["drain_area_km2"], dtype=np.float64) / 2.4465755455
        tag = "specific discharge mm/day (Q / drain_area)"
    elif scale == "iqr":
        denom = np.asarray(z["median_iqr"][:, 1], dtype=np.float64)
        tag = "IQR-scaled (no median subtract, no log)"
    else:
        raise ValueError(f"scale must be 'iqr' or 'area' (got {scale!r})")
    T = raw.shape[0]
    print(f"USGS streamflow (Ohio, {raw.shape[1]} gauges): {T} days, {tag}")
    rng = np.random.default_rng(random_seed)
    if N_Q is None or N_Q >= T:
        X_ = raw.copy(); N_Q = T
    else:
        X_ = raw[rng.choice(T, size=N_Q, replace=False)]
    if N_P is None:
        N_P = N_Q
    X_ = (X_ / denom[None, :].astype(np.float32)).astype(np.float32)
    mu, sigma = X_.mean(axis=0), X_.std(axis=0)
    Y_ = (rng.standard_normal((N_P, X_.shape[1])).astype(np.float32) * sigma[None, :] + mu[None, :]).astype(np.float32)
    print(f"  Q shape: {X_.shape},  P shape: {Y_.shape}")
    print(f"  Q ||R|| mean={np.linalg.norm(X_, axis=1).mean():.3f}  std={np.linalg.norm(X_, axis=1).std():.3f}")
    print(f"  P ||R|| mean={np.linalg.norm(Y_, axis=1).mean():.3f}  std={np.linalg.norm(Y_, axis=1).std():.3f}")
    return X_, Y_
