"""
Train TTF-M (Tail Transform Flow, Hickling & Prangle 2025, ICML) on a target sample set and
emit a pickle in the pretrain schema of the CVaR-GPA driver.

Datasets, each under the matching protocol of the TTF paper:
  - student_t_2D, student_t_2D_nu{1.2,1.5,1.8}, Neal_funnel : TTF synthetic density-estimation protocol
  - FF25_monthly, Streamflow_ohio                             : TTF real-data protocol (their fama5 experiment)

pretrain/settings/TTF.md lists the source of every setting and the deviations from the reference code.
This file also holds the target samples of every dataset (sample_target), used by all the wrappers.
"""
from __future__ import annotations
import argparse
import copy
import pickle
import sys
from functools import partial
from math import ceil
from pathlib import Path

import numpy as np
import torch
from torch import optim

REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party"
sys.path.insert(0, str(REPO_ROOT / "tailnflows"))

from tailnflows.models import flows

# Match TTF's synthetic-DE default dtype.
DEFAULT_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Per-dataset protocol configs
# ---------------------------------------------------------------------------
DATASETS = {
    # 2-d isotropic Student-t with nu = 1 (Cauchy); TTF synthetic density-estimation protocol.
    "student_t_2D": dict(
        protocol="synthetic-DE",
        dim=2,
        n_default=5000,
        source_n=10000,      # the first N rows of a draw of size 10000
        target_kwargs=dict(df=1.0),
        model=dict(num_bins=8, tail_bound=3.0, depth=None,
                   u_linear=False, final_rotation=None),
        opt=dict(lr=5e-3, num_steps=5000, batch_size=None, early_stop_patience=100,
                 eval_period=1),
    ),
    # 2-d isotropic Student-t with nu = 1.2, 1.5, 1.8. source_n = 5000, so the training sample equals the target of the
    # CVaR-GPA driver (CVaR_student_t_nu<NU> at N_samples_Q = 5000, seed 0). Same protocol as student_t_2D, only df changes.
    **{f"student_t_2D_nu{_nu}": dict(
        protocol="synthetic-DE",
        dim=2,
        n_default=5000,
        source_n=5000,
        target_kwargs=dict(df=float(_nu)),
        model=dict(num_bins=8, tail_bound=3.0, depth=None,
                   u_linear=False, final_rotation=None),
        opt=dict(lr=5e-3, num_steps=5000, batch_size=None, early_stop_patience=100,
                 eval_period=1),
    ) for _nu in ("1.2", "1.5", "1.8")},
    # Fama-French 25 monthly portfolios, the raw 1182 centered months; TTF real-data protocol.
    "FF25_monthly": dict(
        protocol="real-data",
        dim=25,
        n_default=1182,
        source_n=1182,
        target_kwargs=dict(freq="monthly", center=True, bootstrap=False),
        model=dict(num_bins=5, tail_bound=2.5, depth=1,
                   u_linear=True, final_rotation="lu"),
        opt=dict(lr=5e-4, num_steps=5000, batch_size=100, early_stop_patience=500,
                 eval_period=25),
    ),
    # Neal's funnel: v ~ N(0, 3^2), x | v ~ N(0, e^v). Same sample as cvar_gpa/util/generate_data.py at N = 5000, seed 0
    # (the generator is not prefix-stable, so source_n equals the driver's N_samples_Q). Synthetic protocol.
    "Neal_funnel": dict(
        protocol="synthetic-DE",
        dim=2,
        n_default=5000,
        source_n=5000,
        target_kwargs=dict(),
        model=dict(num_bins=8, tail_bound=3.0, depth=None,
                   u_linear=False, final_rotation=None),
        opt=dict(lr=5e-3, num_steps=5000, batch_size=None, early_stop_patience=100,
                 eval_period=1),
    ),
    # USGS daily streamflow, Ohio River basin, 64 gauges, specific discharge in mm/day; TTF real-data protocol.
    "Streamflow_ohio": dict(
        protocol="real-data",
        dim=64,
        n_default=7305,
        source_n=7305,
        target_kwargs=dict(),
        model=dict(num_bins=5, tail_bound=2.5, depth=1,
                   u_linear=True, final_rotation="lu"),
        opt=dict(lr=5e-4, num_steps=5000, batch_size=100, early_stop_patience=500,
                 eval_period=10),
    ),
}


# ---------------------------------------------------------------------------
# Target sampling — one branch per dataset
# ---------------------------------------------------------------------------
def sample_target(dataset: str, N: int, seed: int) -> np.ndarray:
    """Return the (N, dim) float64 target sample of `dataset`.

    Synthetic targets replicate the generator of cvar_gpa/util/generate_data.py at size source_n and take
    the first N rows (the generators are not prefix-stable across sizes). Real targets call the loaders
    of cvar_gpa/data with matching arguments.
    """
    cfg = DATASETS[dataset]
    source_n = cfg["source_n"]
    dim = cfg["dim"]
    assert N <= source_n, f"N={N} must be <= source_n={source_n}"
    data_dir = str(Path(__file__).resolve().parents[2].parent / "cvar_gpa" / "data")

    if dataset == "student_t_2D":
        from scipy.stats import multivariate_t
        df = cfg["target_kwargs"]["df"]
        P_ = multivariate_t(np.zeros(dim), np.eye(dim), df=df)
        X_full = P_.rvs(size=source_n, random_state=seed)
        return X_full[:N]

    if dataset in ("student_t_2D_nu1.2", "student_t_2D_nu1.5", "student_t_2D_nu1.8"):
        from scipy.stats import multivariate_t
        P_ = multivariate_t(np.zeros(dim), np.eye(dim), df=cfg["target_kwargs"]["df"])
        return P_.rvs(size=source_n, random_state=seed)[:N]

    if dataset == "Neal_funnel":
        # one default_rng(seed), v then x, at n = source_n
        rng = np.random.default_rng(seed)
        v = rng.normal(0.0, 3.0, size=source_n)
        x = rng.normal(0.0, np.exp(v / 2.0), size=source_n)
        X = np.stack([v, x], axis=1).astype(np.float32).astype(np.float64)   # the driver stores float32
        return X[:N]

    if dataset == "Streamflow_ohio":
        sys.path.insert(0, data_dir)
        from streamflow_loader import load_streamflow
        X, _ = load_streamflow(N_Q=N, N_P=N, random_seed=seed, scale="area")
        return X.astype(np.float64)

    if dataset == "FF25_monthly":
        sys.path.insert(0, data_dir)
        from ff25_loader import load_ff25
        tk = cfg["target_kwargs"]
        X, _Y_gauss = load_ff25(
            freq=tk["freq"], N_Q=N, N_P=N,
            random_seed=seed,
            center=tk["center"], bootstrap_Q=tk["bootstrap"],
        )
        return X.astype(np.float64)

    raise ValueError(f"unknown dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# Model construction — one branch per protocol
# ---------------------------------------------------------------------------
def build_base_transformation(cfg_model):
    """Assemble the base_nsf_transform call with per-protocol kwargs."""
    def _bt(dim):
        kwargs = dict(
            num_bins=cfg_model["num_bins"],
            tail_bound=cfg_model["tail_bound"],
            affine_autoreg_layer=True,
        )
        if cfg_model["depth"] is not None:
            kwargs["depth"] = cfg_model["depth"]
        if cfg_model["u_linear"]:
            kwargs["u_linear_layer"] = True
        return flows.base_nsf_transform(dim, **kwargs)
    return _bt


def build_ttf_m_model(dataset: str) -> torch.nn.Module:
    """Instantiate ttf_m per the corresponding TTF paper protocol.

    Synthetic-DE : run_synthetic_density_estimation.py:33-43 (ttf_m)
    Real-data    : run_density_estimation.py:49-64 (ttf_rqs)
    """
    cfg = DATASETS[dataset]
    dim = cfg["dim"]
    cfg_m = cfg["model"]

    base_transformation_init = build_base_transformation(cfg_m)

    build_kwargs = dict(
        dim=dim,
        use="density_estimation",
        base_transformation_init=base_transformation_init,
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
            neg_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
        ),
    )
    if cfg_m["final_rotation"] is not None:
        # Real-data ttf_rqs uses final_rotation="lu".
        build_kwargs["final_rotation"] = cfg_m["final_rotation"]

    return flows.build_ttf_m(**build_kwargs)


# ---------------------------------------------------------------------------
# Local trainer: mirrors data_fit.train step-for-step + best-val restore.
# (unchanged from previous version)
# ---------------------------------------------------------------------------
def _batch_loader(n, batch_size):
    """Verbatim from data_fit.batch_loader (data_fit.py:7-12)."""
    batches = torch.randperm(n).split(batch_size)
    while True:
        for batch_ix in batches:
            yield batch_ix
        batches = torch.randperm(n).split(batch_size)


def train_ttf_best_val(
    model, x_trn, x_val, x_tst,
    lr, num_steps, batch_size, early_stop_patience,
    label, eval_period=1,
):
    # eval_period: 1 for synthetic-DE (data_fit.train default, not overridden by
    # run_synthetic_density_estimation.py), 25 for real-data fama5
    # (run_density_estimation.py:250-259 optimisation_overrides).
    """Port of data_fit.train (lines 14-135) + best-val state_dict restore.

    The reference data_fit.train tracks best_val_loss but never touches model
    weights, so its returned model is in last-step weights. TTF paper reports
    test-NLL AT best-val. We snapshot and restore weights at best-val so
    sampling is from the paper-consistent checkpoint.
    """
    parameters = list(model.parameters())
    optimizer = optim.Adam(parameters, lr=lr)   # data_fit.py:32-33

    loop = range(num_steps)
    num_evals = ceil(num_steps / eval_period) + 1
    losses = torch.empty(num_steps)
    vlosses = torch.empty(num_evals)
    steps = torch.empty(num_evals, dtype=torch.int)
    tst_loss = torch.tensor(torch.inf)
    best_val_loss = torch.tensor(torch.inf)
    tst_eval = 0
    best_state_dict = None  # ADDED

    n = x_trn.shape[0]
    if batch_size is None:
        batch_size = n
    batches = _batch_loader(n, batch_size)

    step = 0
    eval_number = 0
    for step in loop:
        model.train()
        batch_ix = next(batches)
        batch = x_trn[batch_ix, :]
        optimizer.zero_grad()
        trn_loss = -model.log_prob(batch).mean()
        trn_loss.backward()
        optimizer.step()
        losses[step] = trn_loss.detach()

        if step % eval_period == 0 or (step + 1) == num_steps:
            model.eval()
            with torch.no_grad():
                eval_number = ceil(step / eval_period)
                val_loss = -model.log_prob(x_val).mean()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tst_loss = -model.log_prob(x_tst).mean()
                    tst_eval = eval_number
                    best_state_dict = copy.deepcopy(model.state_dict())  # ADDED

                steps[eval_number] = step
                vlosses[eval_number] = val_loss.detach()

                if (
                    early_stop_patience is not None
                    and step - steps[tst_eval] > early_stop_patience
                ):
                    break

        if step % 200 == 0 or step == num_steps - 1:
            print(f"[TTF] {label} step={step} "
                  f"trn={losses[step]:.4f} val={vlosses[eval_number]:.4f} "
                  f"best_val={best_val_loss:.4f} best_step={int(steps[tst_eval])}",
                  flush=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[TTF] {label} restored best-val weights from step {int(steps[tst_eval])} "
              f"(best_val_loss={best_val_loss:.4f}, tst_loss={tst_loss:.4f})", flush=True)
    else:
        print(f"[TTF] {label} WARN: no best-val snapshot; last-step weights", flush=True)

    return dict(
        best_val_loss=float(best_val_loss),
        tst_loss=float(tst_loss),
        best_step=int(steps[tst_eval]),
        n_steps_run=step + 1,
    )


# ---------------------------------------------------------------------------
# Pickle schema (matches pretrain format)
# ---------------------------------------------------------------------------
def save_pretrain_pickle(out_path: Path, X_, Y_, args):
    """Emit 2-list-of-dicts pickle matching pretrain schema.

    Downstream driver (the CVaR-GPA driver:114-125) reads
    `_prev_result['trajectories'][-1][:N_P]` from --init_P_file.
    """
    cfg = DATASETS[args.dataset]
    params = dict(
        N_samples_Q=args.N, N_samples_P=args.M, N_dim=cfg["dim"],
        dataset=args.dataset, random_seed=args.seed,
        generative_model="TTF", exp_no="pretrain",
        expname=out_path.stem,
        nu=cfg["target_kwargs"].get("df", cfg["target_kwargs"].get("nu_vec", None)),
        X_=X_, Y_=Y_,
        L=None, lam=None, alpha=None, formulation=None,
        f="KL", Gamma="TTF", constraint=None, no_cvar=True,
        beta=None, sigma_P=None, sigma_Q=None, interval_length=None,
        label=None, pts_P=None, pts_Q=None, pts_P_2=None, pts_Q_2=None,
        y0=None,
    )
    telemetry = dict(
        trajectories=[Y_],
        vectorfields=[], QoIs=[], divergences=[], KE_Ps=[],
        FIDs=[], wasserstein1s=[], comp_times=[], cvar_report=[],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump([params, telemetry], f)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def _split_indices(N, val_frac, tst_frac, seed):
    """Random 40/20/40-style split per synth_de_data_generation.ipynb cell 0."""
    n_val = int(N * val_frac)
    n_tst = int(N * tst_frac)
    n_trn = N - n_val - n_tst
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    return perm[:n_trn], perm[n_trn:n_trn + n_val], perm[n_trn + n_val:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=None, help="target size; default = dataset n_default")
    ap.add_argument("--M", type=int, default=None, help="pretrain particle count; default = N")
    # Opt-override (each defaults to dataset's protocol value)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="0 => full-batch")
    ap.add_argument("--early_stop_patience", type=int, default=None)
    ap.add_argument("--eval_period", type=int, default=None)
    # Split
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--tst_frac", type=float, default=0.4)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    opt = cfg["opt"]
    # Fill defaults from cfg where CLI didn't override
    if args.N is None: args.N = cfg["n_default"]
    if args.M is None: args.M = args.N
    if args.lr is None: args.lr = opt["lr"]
    if args.num_steps is None: args.num_steps = opt["num_steps"]
    if args.batch_size is None:
        args.batch_size = 0 if opt["batch_size"] is None else opt["batch_size"]
    if args.early_stop_patience is None:
        args.early_stop_patience = opt["early_stop_patience"]
    if args.eval_period is None:
        args.eval_period = opt["eval_period"]

    # -- Seeding --
    torch.manual_seed(args.seed)
    torch.set_default_dtype(DEFAULT_DTYPE)
    np.random.seed(args.seed)

    print(f"[TTF] dataset={args.dataset} protocol={cfg['protocol']} "
          f"dim={cfg['dim']} seed={args.seed} N={args.N} M={args.M}", flush=True)
    print(f"[TTF] opt: lr={args.lr} num_steps={args.num_steps} "
          f"batch_size={args.batch_size} early_stop={args.early_stop_patience} "
          f"eval_period={args.eval_period}", flush=True)
    print(f"[TTF] model: {cfg['model']}", flush=True)

    # -- 1. Target --
    x_all = sample_target(args.dataset, args.N, args.seed)
    r = np.linalg.norm(x_all, axis=1)
    print(f"[TTF] target: shape={x_all.shape}, ||x|| P50={np.median(r):.2f} "
          f"P99={np.quantile(r,0.99):.2f} max={r.max():.2e}", flush=True)

    # -- 2. Split --
    trn_ix, val_ix, tst_ix = _split_indices(args.N, args.val_frac, args.tst_frac, args.seed)

    # -- 2b. Standardization (real-data protocol only) --
    # TTF real-data protocol standardizes per-dim by trn+val mean/std:
    # generate_splits.py:29-33 computes them over trn_val_mask (all but tst),
    # run_density_estimation.py:174-179 applies (x - mean) / std to all splits.
    # Synthetic-DE trains on raw generate_data output (no standardization).
    # Samples are de-standardized back to data units before pickling, so the
    # emitted pickle (X_ and Y_) is always in original data scale.
    if cfg["protocol"] == "real-data":
        trn_val_mask = np.ones(args.N, dtype=bool)
        trn_val_mask[tst_ix] = False
        std_mean = x_all[trn_val_mask].mean(axis=0)
        std_std = x_all[trn_val_mask].std(axis=0, ddof=1)  # torch .std default is unbiased
        x_fit = (x_all - std_mean) / std_std
        print(f"[TTF] standardized (trn+val stats): mean |m| max={np.abs(std_mean).max():.3f}, "
              f"std range=[{std_std.min():.3f}, {std_std.max():.3f}]", flush=True)
    else:
        std_mean, std_std = None, None
        x_fit = x_all

    x_trn = torch.from_numpy(x_fit[trn_ix].astype(np.float32))
    x_val = torch.from_numpy(x_fit[val_ix].astype(np.float32))
    x_tst = torch.from_numpy(x_fit[tst_ix].astype(np.float32))
    print(f"[TTF] split: trn={len(trn_ix)} val={len(val_ix)} tst={len(tst_ix)}", flush=True)

    # -- 3. Model --
    model = build_ttf_m_model(args.dataset).to(DEFAULT_DTYPE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TTF] built ttf_m ({cfg['protocol']}): {n_params} trainable params", flush=True)

    # -- 4. Train --
    bs = None if args.batch_size == 0 else args.batch_size
    train_result = train_ttf_best_val(
        model=model,
        x_trn=x_trn, x_val=x_val, x_tst=x_tst,
        lr=args.lr, num_steps=args.num_steps, batch_size=bs,
        early_stop_patience=args.early_stop_patience,
        label=f"TTF-{args.dataset}-s{args.seed}",
        eval_period=args.eval_period,
    )
    print(f"[TTF] train result: {train_result}", flush=True)

    # -- 5. Sample --
    model.eval()
    with torch.no_grad():
        Y_ = model.sample(args.M).detach().cpu().numpy().astype(np.float32)
    if std_mean is not None:
        # De-standardize back to data units so the pickle matches driver scale.
        Y_ = (Y_ * std_std.astype(np.float32) + std_mean.astype(np.float32))
    rY = np.linalg.norm(Y_, axis=1)
    print(f"[TTF] emitted {args.M} particles: shape={Y_.shape}, "
          f"||y|| P50={np.median(rY):.2f} P99={np.quantile(rY,0.99):.2f} "
          f"max={rY.max():.2e}", flush=True)

    # -- 6. Save --
    save_pretrain_pickle(Path(args.out), X_=x_all, Y_=Y_, args=args)
    print(f"[TTF] pretrain pickle written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
