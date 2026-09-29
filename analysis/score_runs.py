#!/usr/bin/env python3
"""Score every run of the paper and write the metric CSVs to analysis/metrics/.

  lipkl_metrics.csv          Lip-KL-GPA before and after fine-tuning: 2-d Student-t (four tail indices), Neal's funnel,
                             Fama-French 25, streamflow, and the study of the level alpha on the 2-d Cauchy target.
  seven_models_metrics.csv   the seven pre-trained models before and after fine-tuning, on the 2-d Student-t targets
                             (tail index 1.0 = Cauchy, 1.2, 1.5, 1.8), Neal's funnel, Fama-French 25 and streamflow;
                             joint error and the error of every marginal.
  tedm_sweep_metrics.csv     pre-trained t-EDM for each tail parameter nu in {3, 5, 7}, and which nu is reported.
  streamflow_gauges.csv      USGS site number, drainage area and Hill tail index of the 64 gauges.
  ff25_portfolios.csv        name and Hill tail index of the 25 Fama-French portfolios.

Every error is measured against the target sample stored in the fine-tuned Lip-KL-GPA run of that target.
Joint errors use the radius ||x||, marginal errors |x_i|.
"""
import csv
import pickle

import numpy as np

import paths
from metrics import global_l1_error, hill_tail_index, tail_error


# ----------------------------------------------------------------------------- reading the run outputs
def load_particles(pickle_file):
    """Particles of the last snapshot of a run."""
    with open(pickle_file, "rb") as fh:
        return np.asarray(pickle.load(fh)[1]["trajectories"][-1], float)


def load_target(pickle_file):
    """Target sample stored in a run."""
    with open(pickle_file, "rb") as fh:
        return np.asarray(pickle.load(fh)[0]["X_"], float)


def load_pair(target, model, tedm_nu=None, alpha=None):
    """-> (target sample, particles of the pre-trained model, particles of the fine-tuned model).
    Only the first M particles of the pre-trained model enter the fine-tuning, so only those are scored."""
    finetuned = paths.finetuned_file(target, model, tedm_nu, alpha)
    fine_tuned_particles = load_particles(finetuned)
    pretrained_particles = load_particles(paths.pretrained_file(target, model, tedm_nu))[:fine_tuned_particles.shape[0]]
    return load_target(finetuned), pretrained_particles, fine_tuned_particles


# ----------------------------------------------------------------------------- the two errors
def joint_errors(particles, target_sample):
    """(global L1 error, tail error) of the radius ||x||."""
    r_generated, r_target = np.linalg.norm(particles, axis=1), np.linalg.norm(target_sample, axis=1)
    return global_l1_error(r_generated, r_target), tail_error(r_generated, r_target)


def marginal_errors(particles, target_sample, k):
    """(global L1 error, tail error) of |x_k|."""
    generated, target = np.abs(particles[:, k]), np.abs(target_sample[:, k])
    return global_l1_error(generated, target), tail_error(generated, target)


# ----------------------------------------------------------------------------- Lip-KL-GPA
def score_lipkl():
    """Rows (experiment, row, pretrained_l1, finetuned_l1, pretrained_tail, finetuned_tail)."""
    rows = []

    def add(experiment, label, sample, pretrained, finetuned, k=None):
        errors = joint_errors if k is None else (lambda particles, target: marginal_errors(particles, target, k))
        (pre_l1, pre_tail), (ft_l1, ft_tail) = errors(pretrained, sample), errors(finetuned, sample)
        rows.append([experiment, label, pre_l1, ft_l1, pre_tail, ft_tail])

    for nu in ("1.0", "1.2", "1.5", "1.8"):
        add("student_t", nu, *load_pair(f"student_t_nu{nu}", "Lip-KL"))

    sample, pretrained, finetuned = load_pair("neal_funnel", "Lip-KL")
    for k, coordinate in enumerate(("x_1", "x_2")):
        add("neal_funnel", coordinate, sample, pretrained, finetuned, k)
    add("neal_funnel", "joint", sample, pretrained, finetuned)

    sample, pretrained, finetuned = load_pair("ff25", "Lip-KL")
    add("ff25", "joint", sample, pretrained, finetuned)
    for k in range(sample.shape[1]):
        add("ff25", str(k), sample, pretrained, finetuned, k)             # portfolio k

    sample, pretrained, finetuned = load_pair("streamflow", "Lip-KL")
    gauges = np.load(paths.STREAMFLOW_NPZ, allow_pickle=True)
    site_ids = [str(s) for s in gauges["site_id"]]
    add("streamflow", "joint", sample, pretrained, finetuned)
    for k in np.argsort(np.asarray(gauges["drain_area_km2"], float)):       # gauges sorted by drainage area
        add("streamflow", site_ids[k], sample, pretrained, finetuned, k)

    for alpha in ("0.85", "0.95", "0.9994"):                                # 0.9994 = 1 - 3/5000 is the run of the paper
        add("alpha_level", alpha, *load_pair("student_t_nu1.0", "Lip-KL", alpha=None if alpha == "0.9994" else alpha))
    return rows


# ----------------------------------------------------------------------------- the seven pre-trained models
def score_tedm_sweep(target, target_sample):
    """t-EDM is trained for nu in {3, 5, 7}, as its authors do. -> {nu: (joint errors, mean marginal errors)} of the pre-trained models."""
    sweep = {}
    n_particles = paths.TARGETS[target]["particles"]
    for nu in (3, 5, 7):
        particles = load_particles(paths.pretrained_file(target, "t-EDM", nu))[:n_particles]
        marginals = np.array([marginal_errors(particles, target_sample, k) for k in range(target_sample.shape[1])]).mean(0)
        sweep[nu] = (joint_errors(particles, target_sample), (float(marginals[0]), float(marginals[1])))
    return sweep


def score_seven_models():
    """-> rows (target, model, tedm_nu, quantity, stage, l1_error, tail_error) and
          sweep rows (target, tedm_nu, reported, joint_l1, joint_tail, marginal_mean_l1, marginal_mean_tail)."""
    rows, sweep_rows = [], []
    for target in paths.SEVEN_MODEL_TARGETS:
        target_sample = load_target(paths.finetuned_file(target, "Lip-KL"))
        dim = target_sample.shape[1]

        if target == "neal_funnel":
            reported_nu = paths.TEDM_NU_NEAL
        else:   # the reported t-EDM row is the nu with the smallest mean marginal tail error of the pre-trained model
            sweep = score_tedm_sweep(target, target_sample)
            reported_nu = min(sweep, key=lambda nu: sweep[nu][1][1])
            for nu, (joint, marginal_mean) in sweep.items():
                sweep_rows.append([target, nu, "yes" if nu == reported_nu else "no", *joint, *marginal_mean])

        for model in paths.MODELS:
            tedm_nu = reported_nu if model == "t-EDM" else None
            sample, pretrained, finetuned = load_pair(target, model, tedm_nu)
            assert np.array_equal(sample, target_sample), f"{model} on {target} was fine-tuned on another target sample"
            nu_label = "" if tedm_nu is None else str(tedm_nu).replace("-", ",")
            for stage, particles in (("pretrained", pretrained), ("finetuned", finetuned)):
                rows.append([target, model, nu_label, "joint", stage, *joint_errors(particles, target_sample)])
                for k in range(dim):
                    rows.append([target, model, nu_label, f"marginal_{k}", stage, *marginal_errors(particles, target_sample, k)])
    return rows, sweep_rows


def streamflow_gauges():
    gauges = np.load(paths.STREAMFLOW_NPZ, allow_pickle=True)
    return [[str(site), f"{area:.1f}", f"{hill:.3f}"] for site, area, hill in
            zip(gauges["site_id"], np.asarray(gauges["drain_area_km2"], float), np.asarray(gauges["hill_alpha"], float))]


def ff25_portfolios():
    """Hill tail index of |x_k| for every portfolio k of the centered monthly returns, which are the fine-tuning target."""
    with open(paths.FF25_CSV) as fh:
        names = fh.readline().strip().split(",")[1:]
    returns = np.loadtxt(paths.FF25_CSV, delimiter=",", skiprows=1, usecols=range(1, len(names) + 1)).astype(np.float32)
    returns = returns - returns.mean(axis=0, keepdims=True)
    return [[k, name, f"{hill_tail_index(np.abs(returns[:, k])):.3f}"] for k, name in enumerate(names)]


def write_csv(name, header, rows):
    paths.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.METRICS_DIR / name, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print("wrote", paths.METRICS_DIR / name, f"({len(rows)} rows)")


if __name__ == "__main__":
    write_csv("lipkl_metrics.csv", ["experiment", "row", "pretrained_l1", "finetuned_l1", "pretrained_tail", "finetuned_tail"], score_lipkl())
    seven, sweep = score_seven_models()
    write_csv("seven_models_metrics.csv", ["target", "model", "tedm_nu", "quantity", "stage", "l1_error", "tail_error"], seven)
    write_csv("tedm_sweep_metrics.csv", ["target", "tedm_nu", "reported", "joint_l1", "joint_tail", "marginal_mean_l1", "marginal_mean_tail"], sweep)
    write_csv("streamflow_gauges.csv", ["gauge", "drain_area_km2", "hill_alpha"], streamflow_gauges())
    write_csv("ff25_portfolios.csv", ["portfolio", "name", "hill_alpha"], ff25_portfolios())
