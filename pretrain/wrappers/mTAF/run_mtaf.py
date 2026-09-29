"""
Train mTAF (Laszkiewicz, Lederer & Fischer, ICML 2022) on a target sample set
and emit a pickle in the pretrain schema of the CVaR-GPA driver.

PROVENANCE: hyperparameters and code path follow the authors' synthetic
NSF pipeline (github.com/MikeLasz/marginalTailAdaptiveFlow @ cbcfeb4:
synthetic_experiments/run_nsf_df2.sh + main.py + utils/flows.py). Their NSF
class, norm_tDist base, TailRandomPermutation/TailLU linearities, training
loop (train()), and tail_estimation.py CLI are called directly.

Their settings used here (synthetic, d != 50):
  NSF, 5 layers, num_bins 3, tail_bound 2, hidden 30, num_blocks 1, ReLU,
  residual blocks, tails linear; Adam lr 1e-4 (NSF.config default), df group
  lr 0 for mTAF(fix) / lr_df 0.005 for learnable mTAF; 10000 steps, batch 512,
  no grad clip, best-val checkpoint every 250 steps restored at the end;
  z-score by full-sample stats; train:val 3:2; tail estimation on abs(val)
  via tail_estimation.py --noise 0; estimate > 10 -> light.

ADAPTATIONS (ledger in pretrain/settings/mTAF.md): 5000-sample budget (3000/2000),
min_bin_width/height=1e-3 restore, all-heavy degenerate linearity
(RandomPermutation + LULinear with all-t base), absolute-path tail-estimator
invocation, per-dataset variant (fix for synthetic, learnable for FF25),
wandb/openturns stubs, comparison seeding.
"""
from __future__ import annotations
import argparse
import pickle
import subprocess
import sys
import types
from functools import partial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "third_party" / "mTAF_official"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "TTF"))

# --- stub deps absent from the env and unused on our path (ledger #7) ---
sys.modules["wandb"] = types.SimpleNamespace(
    log=lambda *a, **k: None, init=lambda *a, **k: None,
    config=types.SimpleNamespace(update=lambda *a, **k: None))
sys.modules["openturns"] = types.ModuleType("openturns")

import utils.flows as uflows                      # noqa: E402
from utils.flows import NSF                       # noqa: E402
from utils.tail_permutation import TailRandomPermutation, TailLU, LULinear  # noqa: E402
from nflows.transforms.permutations import RandomPermutation  # noqa: E402
from nflows.transforms.autoregressive import (                # noqa: E402
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform as RQSTransform)
from sklearn.model_selection import train_test_split          # noqa: E402
from run_ttf import sample_target, DATASETS as TTF_DATASETS   # noqa: E402

BATCH = 512            # flows.py train() default
TRAIN_STEPS = 10000    # main.py:117-119, d != 50
NUM_LAYERS = 5         # run_nsf_df2.sh
TAIL_BOUND = 2.0       # run_nsf_df2.sh
NUM_BINS = 3           # main.py:151
NUM_HIDDEN = 30        # main.py:115-119, d != 50
NUM_BLOCKS = 1         # main.py default
LR_DF = 0.005          # main.py:113 (synthetic pipeline; used only for --variant mTAF)
# real-data (weather) protocol: real_world_experiments/weather.sh:2-4 + main.py
REAL_NUM_LAYERS = 5    # weather.sh:4 (overrides main.py default 10)
REAL_NUM_BLOCKS = 2    # weather.sh:2
REAL_NUM_HIDDEN = 100  # weather.sh:3
REAL_TRAIN_STEPS = 20000  # main.py --train_steps default
REAL_LR_DF = 0.01      # main.py --lr_df default
REAL_TAIL_BOUND = 2.5  # main.py:81


def patch_reference_classes():
    """Ledger #3 + #4 + TailLU device quirk (AUDIT §3)."""
    dev = uflows.device

    def tail_perm(features_light, features_heavy, dim=1):
        if features_light == 0:   # all-heavy degenerate case (ledger #4)
            return RandomPermutation(int(features_heavy))
        return TailRandomPermutation(features_light, features_heavy, dim)

    def tail_lu(features, num_heavy, using_cache=False):
        if num_heavy == features:  # all-heavy: W = single full LU block
            return LULinear(features)
        return TailLU(features, num_heavy, using_cache, device=dev)

    uflows.TailRandomPermutation = tail_perm
    uflows.TailLU = tail_lu
    # restore official nflows-0.14 min_bin defaults (fork changed to 1e-4)
    uflows.MaskedPiecewiseRationalQuadraticAutoregressiveTransform = partial(
        RQSTransform, min_bin_width=1e-3, min_bin_height=1e-3)


def estimate_tail_indices(x_val: np.ndarray, workdir: Path) -> np.ndarray:
    """Their estimate_tails() (flows.py:268-290) with absolute paths (ledger #5).
    Runs tail_estimation.py per marginal on abs(val); returns the raw estimates."""
    est_script = REPO / "utils" / "tail_estimation.py"
    out = np.zeros(x_val.shape[1])
    for j in range(x_val.shape[1]):
        marg = np.abs(x_val[:, j])
        dat = np.column_stack([marg, np.ones(len(marg), dtype=int)])
        p_dat = workdir / f"marginal{j + 1}.dat"
        p_txt = workdir / f"tail_estimator{j + 1}.txt"
        np.savetxt(p_dat, dat, fmt=["%10.5f", "%d"])
        subprocess.run(
            [sys.executable, str(est_script), str(p_dat),
             str(workdir / f"marginal{j + 1}_results.pdf"),
             "--noise", "0", "--path_estimator", str(p_txt)],
            check=True, cwd=str(workdir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out[j] = float(np.loadtxt(p_txt))
        print(f"[mTAF] marginal {j + 1}: raw tail-index estimate {out[j]:.3f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(TTF_DATASETS.keys()))
    ap.add_argument("--variant", default="mTAF(fix)",
                    choices=["mTAF(fix)", "mTAF"],
                    help="mTAF(fix): synthetic protocol (df frozen at estimates). "
                         "mTAF: learnable df (FF25, per their real-data practice).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--protocol", default="synthetic", choices=["synthetic", "real"],
                    help="synthetic: run_nsf_df2.sh config (5 layers, 1 block, 30 hidden, "
                         "10000 steps, lr_df 0.005, tail_bound 2.0, no BN, no cosine). "
                         "real: their real-data (weather) config, "
                         "real_world_experiments/weather.sh + main.py:73-85: "
                         "5 layers, 2 blocks, 100 hidden, 20000 steps, lr 1e-4, "
                         "lr_df 0.01, cosine annealing, tail_bound 2.5, 3 bins, "
                         "BatchNorm layer after each RQS block, LU linear. "
                         "Tail labels/df init still come from the EVT estimates "
                         "(weather hard-codes labels + df init 5; FF25 has no "
                         "such domain labels).")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override ONLY for smoke tests (default: protocol value).")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    if args.steps is None:
        args.steps = TRAIN_STEPS if args.protocol == "synthetic" else REAL_TRAIN_STEPS

    d = TTF_DATASETS[args.dataset]["dim"]
    print(f"[mTAF] dataset={args.dataset} d={d} variant={args.variant} "
          f"N={args.N} M={args.M} device={uflows.device} steps={args.steps}",
          flush=True)

    out_path = Path(args.out).resolve()
    workdir = out_path.parent / (f"mtaf_workdir_{args.dataset}"
                                 + ("" if args.protocol == "synthetic" else "_real"))
    workdir.mkdir(parents=True, exist_ok=True)
    import os
    os.chdir(workdir)  # contain their relative-path writes (trained_models/ etc.)

    x_all = sample_target(args.dataset, args.N, args.seed).astype(np.float64)
    r = np.linalg.norm(x_all, axis=1)
    print(f"[mTAF] target: ||x|| P50={np.median(r):.2f} "
          f"P99={np.quantile(r, 0.99):.2f} max={r.max():.2e}", flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # normalization by full-sample stats, mirrors flows.py:146-151 (ledger #2)
    mu, s = x_all.mean(axis=0), x_all.std(axis=0)
    x_norm = (x_all - mu) / s
    # their 3:2 train:val ratio (flows.py:141-145 at d != 50; ledger #1);
    # raw val carried alongside for tail estimation (ledger #9)
    data_train, data_val, _, data_val_raw = train_test_split(
        x_norm, x_all, test_size=2 / 5)

    raw_estimates = estimate_tail_indices(data_val_raw, workdir)

    patch_reference_classes()
    model = NSF(args.variant)

    # emulate load_data attribute state (their synthetic branch), then the
    # get_tails() logic (flows.py:205-266) on our estimates
    model.data = ""
    model.D = d
    model.df = 0
    model.setting = args.dataset
    model.data_train, model.data_val = data_train, data_val
    model.data_test = data_val.copy()  # only feeds their final test print (ledger #1)

    for j, tail_index in enumerate(raw_estimates):
        if tail_index > 10:  # their light-tail rule, flows.py:225-227
            tail_index = 0
        if tail_index == 0:
            model.heavy_tailed.append(False)
            print(f"[mTAF] marginal {j + 1} classified light-tailed", flush=True)
        else:
            model.heavy_tailed.append(True)
            print(f"[mTAF] marginal {j + 1} heavy-tailed, df init {tail_index:.3f}",
                  flush=True)
        model.list_tailindexes.append(tail_index)

    model.num_heavy = int(np.sum(np.array(model.heavy_tailed)))
    model.num_light = int(d - model.num_heavy)
    model.permutation = np.argsort(np.array(model.heavy_tailed))
    model.inv_perm = np.zeros(d, dtype=np.int32)
    for j in range(d):
        model.inv_perm[model.permutation[j]] = j
    model.tail_index_permuted = np.array(model.list_tailindexes)[model.permutation]
    model.data_train = model.data_train[:, model.permutation]
    model.data_val = model.data_val[:, model.permutation]
    model.data_test = model.data_test[:, model.permutation]
    print(f"[mTAF] {model.num_light} light / {model.num_heavy} heavy", flush=True)

    # their NSF config call (main.py:150-152) with the sh-file values
    from torch.nn import functional as F
    if args.protocol == "synthetic":
        cfg = dict(num_layers=NUM_LAYERS, tail_bound=TAIL_BOUND,
                   train_steps=args.steps, num_bins=NUM_BINS, linear_layer="LU",
                   track_results=True, lr_df=LR_DF, model_nr=0,
                   num_hidden=NUM_HIDDEN, activation=F.relu,
                   num_blocks=NUM_BLOCKS)
    else:
        # real_world_experiments/main.py:73-85 (config call) with weather.sh
        # values num_layers=5, num_blocks=2, num_hidden=100; train_steps and
        # lr_df are main.py argparse defaults (20000, 0.01).
        cfg = dict(linear_layer="LU", num_layers=REAL_NUM_LAYERS,
                   train_steps=args.steps, num_hidden=REAL_NUM_HIDDEN,
                   lr_df=REAL_LR_DF, lr_wd=0.0, cosine_annealing=True,
                   tail_bound=REAL_TAIL_BOUND, num_bins=NUM_BINS,
                   num_blocks=REAL_NUM_BLOCKS, lr=1e-4, batch_norm_layer=True,
                   track_results=True, model_nr=0, activation=F.relu)
    print(f"[mTAF] protocol={args.protocol} config={cfg}", flush=True)
    model.config(**cfg)
    n_params = sum(q.numel() for q in model.flow.parameters())
    print(f"[mTAF] NSF flow: {n_params} params", flush=True)

    model.train(batch_size=BATCH, grad_clip=False)

    # sampling, mirrors compute_area (flows.py:388-390): sample -> inv_perm
    model.flow.eval()
    with torch.no_grad():
        chunks, remaining = [], args.M
        while remaining > 0:
            b = min(BATCH * 4, remaining)
            y = model.flow.sample(b).detach().cpu().numpy()
            chunks.append(y)
            remaining -= b
    Y_norm = np.concatenate(chunks, axis=0)[:args.M]
    Y_norm = Y_norm[:, model.inv_perm]
    Y_ = (Y_norm * s + mu).astype(np.float32)
    rY = np.linalg.norm(Y_, axis=1)
    print(f"[mTAF] emitted {Y_.shape[0]} particles: ||y|| P50={np.median(rY):.2f} "
          f"P99={np.quantile(rY, 0.99):.2f} max={rY.max():.2e}", flush=True)

    params = dict(
        N_samples_Q=args.N, N_samples_P=args.M, N_dim=d,
        dataset=args.dataset, random_seed=args.seed,
        generative_model="mTAF", exp_no="pretrain",
        expname=Path(args.out).stem, nu=None,
        X_=x_all, Y_=Y_,
        L=None, lam=None, alpha=None, formulation=None,
        f="KL", Gamma=args.variant, constraint=None, no_cvar=True,
        mtaf_protocol=args.protocol,
        beta=None, sigma_P=None, sigma_Q=None, interval_length=None,
        label=None, pts_P=None, pts_Q=None, pts_P_2=None, pts_Q_2=None,
        y0=None,
        mtaf_tail_estimates=list(raw_estimates),
        mtaf_heavy_tailed=list(model.heavy_tailed),
    )
    telemetry = dict(trajectories=[Y_], vectorfields=[], QoIs=[], divergences=[],
                     KE_Ps=[], FIDs=[], wasserstein1s=[], comp_times=[],
                     cvar_report=[])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump([params, telemetry], f)
    print(f"[mTAF] pretrain pickle written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
