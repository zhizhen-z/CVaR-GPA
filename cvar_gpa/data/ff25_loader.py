#!/usr/bin/env python3
"""
Loader for Fama-French 25 Portfolios dataset.

Q (target)  : actual historical returns  R_t ∈ R^25  (in percent)
Y_ (initial P): multivariate Gaussian N(0, diag(sigma^2))
               — same marginal variances as Q, but independent components
               — captures the scale but not the heavy tails or correlations
               — what the algorithm must learn to transport toward Q

Data files required (run data/ff25/download_ff25.py once first):
    data/ff25/FF25_monthly.csv
    data/ff25/FF25_daily.csv
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ff25")


def load_ff25(
    freq='monthly',
    N_Q=None,
    N_P=None,
    random_seed=0,
    start_date=None,   # e.g. '1963-07-01'
    end_date=None,     # e.g. '2007-12-31'
    center=True,       # subtract empirical mean from returns
    bootstrap_Q=False, # if True, resample Q to reach N_Q
):
    """
    Returns
    -------
    X_ : np.ndarray (N_Q, 25)  — target distribution samples (actual returns)
    Y_ : np.ndarray (N_P, 25)  — initial distribution samples (Gaussian)
    """
    fname = 'FF25_monthly.csv' if freq == 'monthly' else 'FF25_daily.csv'
    fpath = os.path.join(DATA_DIR, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"{fpath} not found.\n"
            "Run 'python data/ff25/download_ff25.py' on a machine with internet access first."
        )

    df = pd.read_csv(fpath, index_col=0, parse_dates=True)

    # optional date filter
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    returns = df.values.astype(np.float32)   # shape (T, 25), in percent

    if center:
        returns = returns - returns.mean(axis=0, keepdims=True)

    T = len(returns)
    print(f"FF25 {freq}: {T} observations, {returns.shape[1]} portfolios")

    # ── Build Q (target) ──────────────────────────────────────────────────────
    rng = np.random.default_rng(random_seed)

    if bootstrap_Q and N_Q is not None:
        # Bootstrap-resample to exactly N_Q (allows N_Q > T)
        idx = rng.integers(0, T, size=N_Q)
        X_ = returns[idx]
    elif N_Q is None or N_Q >= T:
        X_ = returns.copy()
        N_Q = T
    else:
        idx = rng.choice(T, size=N_Q, replace=False)
        X_ = returns[idx]

    # ── Build P (initial, Gaussian with matching marginal std) ────────────────
    if N_P is None:
        N_P = N_Q

    sigma = returns.std(axis=0)          # (25,) — per-portfolio std
    Z = rng.standard_normal((N_P, 25)).astype(np.float32)
    Y_ = Z * sigma[np.newaxis, :]        # independent Gaussian, same scale

    print(f"  Q shape: {X_.shape},  P shape: {Y_.shape}")
    print(f"  Q ||R|| mean={np.linalg.norm(X_, axis=1).mean():.3f}  "
          f"std={np.linalg.norm(X_, axis=1).std():.3f}")
    print(f"  P ||R|| mean={np.linalg.norm(Y_, axis=1).mean():.3f}  "
          f"std={np.linalg.norm(Y_, axis=1).std():.3f}")

    return X_, Y_
