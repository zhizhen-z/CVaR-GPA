"""The two error metrics of the paper, and the Hill estimate of a tail index.

Both errors compare a generated sample with the target sample through a nonnegative risk value:
the radius ||x|| for the joint error, |x_i| for the error of the i-th marginal.
"""
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def radius(samples):
    """||x|| of every row."""
    return np.linalg.norm(np.asarray(samples), axis=1)


def global_l1_error(generated, target, n_grid=2000):
    """Global L^1 error: integral of |CCDF_generated - CCDF_target| over [0, largest risk value in either sample],
    by the trapezoidal rule on a uniform grid of n_grid points. Arguments are 1-d arrays of risk values."""
    r_max = float(max(generated.max(), target.max()))
    grid = np.linspace(0.0, r_max, n_grid)
    ccdf_generated = 1.0 - np.searchsorted(np.sort(generated), grid, side='right') / len(generated)
    ccdf_target = 1.0 - np.searchsorted(np.sort(target), grid, side='right') / len(target)
    return float(_trapezoid(np.abs(ccdf_generated - ccdf_target), grid))


def tail_error(generated, target, p_lo=95.0, p_hi=99.9, n_levels=20, eps=1e-10):
    """Tail error: mean absolute difference of the log-CCDFs at the target quantiles of
    n_levels equi-spaced levels in [0.95, 0.999]. Arguments are 1-d arrays of risk values."""
    thresholds = np.percentile(target, np.linspace(p_lo, p_hi, n_levels))
    differences = [
        abs(np.log((generated > t).mean() + eps) - np.log((target > t).mean() + eps))
        for t in thresholds
    ]
    return float(np.mean(differences))


def hill_tail_index(values, k_frac=0.05):
    """Hill estimate of the tail index from the largest k_frac of the positive values (at least 20 of them).
    Used to describe the real targets; the streamflow data file stores the same estimate for every gauge."""
    x = np.asarray(values, float)
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    k = max(20, int(len(x) * k_frac))
    return float(1.0 / np.mean(np.log(x[-k:] / x[-k - 1])))
