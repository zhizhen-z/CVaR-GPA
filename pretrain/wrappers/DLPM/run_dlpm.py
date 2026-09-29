"""
Train DLPM (Shariatian, Simsekli & Durmus, ICLR 2025, arXiv:2407.18609) on a
target sample set and emit a pickle in the pretrain schema of the CVaR-GPA driver.

Settings = the authors' 2-d data protocol (dlpm/configs/2d_data.yml), with their model, method and optimizer
classes imported directly. pretrain/settings/DLPM.md lists every setting and the adaptations ledger.
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "third_party" / "DLPM"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "TTF"))

from dlpm.models.Model import MLPModel  # noqa: E402
from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess  # noqa: E402
from run_ttf import sample_target, DATASETS as TTF_DATASETS  # noqa: E402

BATCH = 1024
LR = 5e-3
ALPHA = 1.8
REVERSE_STEPS = 100
EPOCHS_MATCHED = 125   # 625 AdamW steps at N=5000 (ledger item 1)


def build_p(d: int, device) -> dict:
    """Parameter dict mirroring dlpm/configs/2d_data.yml (relevant sections)."""
    return {
        "device": device,
        "method": "dlpm",
        "data": {"nfeatures": d},
        "dlpm": {
            "alpha": ALPHA, "reverse_steps": REVERSE_STEPS, "isotropic": True,
            "mean_predict": "EPSILON", "var_predict": "FIXED",
            "rescale_timesteps": True, "scale": "scale_preserving",
            "input_scaling": False,
        },
        "model": {
            "a_emb_size": 32, "a_pos_emb": False, "act": "silu",
            "dropout_rate": 0.0, "group_norm": True, "nblocks": 4,
            "nunits": 64, "skip_connection": True, "time_emb_size": 32,
            "time_emb_type": "learnable", "no_a": True, "use_a_t": False,
            "learn_variance": False,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(TTF_DATASETS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=EPOCHS_MATCHED)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="stop after exactly this many AdamW steps (used for the "
                         "raw-FF25 run: N=1182 gives 2 batches/epoch, so the "
                         "625-step matched budget cannot be hit on an epoch "
                         "boundary; pass --epochs large and --max_steps 625)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = TTF_DATASETS[args.dataset]["dim"]
    print(f"[DLPM] dataset={args.dataset} d={d} alpha={ALPHA} N={args.N} "
          f"M={args.M} device={device} epochs={args.epochs} "
          f"(~{args.epochs * max(1, args.N // BATCH + (args.N % BATCH > 0))} steps)", flush=True)

    x_all = sample_target(args.dataset, args.N, args.seed).astype(np.float64)
    r = np.linalg.norm(x_all, axis=1)
    print(f"[DLPM] target: ||x|| P50={np.median(r):.2f} P99={np.quantile(r,0.99):.2f} "
          f"max={r.max():.2e}", flush=True)

    mean, std = x_all.mean(0), x_all.std(0, ddof=1)
    # BEM's 2D pipeline stores samples with a channel dim: (N, 1, d)
    # (bem/datasets/__init__.py:146) — the MLPModel shapes require it.
    x_norm = torch.as_tensor((x_all - mean) / std, dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    p = build_p(d, device)
    model = MLPModel(p).to(device)  # dlpm_experiment.init_model_by_parameter does .to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[DLPM] MLPModel: {n_params} params", flush=True)

    method = GenerativeLevyProcess(
        alpha=ALPHA, device=device, reverse_steps=REVERSE_STEPS,
        model_mean_type="EPSILON", model_var_type="FIXED",
        rescale_timesteps=True, isotropic=True, LIM=False,
        scale="scale_preserving", input_scaling=False,
    )

    # AdamW per InitUtils.init_default_optimizer (utils_exp.py:297-302)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_norm), batch_size=BATCH, shuffle=True)

    models = {"default": model}
    step = 0
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for (xb,) in loader:
            if args.max_steps is not None and step >= args.max_steps:
                done = True
                break
            # mirrors bem/TrainingManager.py:116-125
            out = method.training_losses(models, xb, loss_type="EPS_LOSS",
                                         lploss=2.0, loss_monte_carlo="mean",
                                         monte_carlo_outer=1, monte_carlo_inner=1)
            loss = out["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        if epoch % 25 == 0 or epoch == args.epochs - 1 or done:
            print(f"[DLPM] epoch={epoch} step={step} loss={loss.item():.5f}", flush=True)
    print(f"[DLPM] training finished: {step} AdamW steps", flush=True)

    model.eval()
    with torch.no_grad():
        chunks = []
        remaining = args.M
        while remaining > 0:
            b = min(BATCH, remaining)   # eval batch_size 1024 (2d_data.yml)
            x = method.sample(models, shape=(b, 1, d), reverse_steps=REVERSE_STEPS,
                              clip_denoised=False, deterministic=False)
            if isinstance(x, (tuple, list)):
                x = x[0]
            chunks.append(x.cpu().numpy().reshape(b, d))
            remaining -= b
    Y_norm = np.concatenate(chunks, axis=0)[:args.M]
    Y_ = (Y_norm * std + mean).astype(np.float32)
    rY = np.linalg.norm(Y_, axis=1)
    print(f"[DLPM] emitted {Y_.shape[0]} particles: ||y|| P50={np.median(rY):.2f} "
          f"P99={np.quantile(rY,0.99):.2f} max={rY.max():.2e}", flush=True)

    params = dict(
        N_samples_Q=args.N, N_samples_P=args.M, N_dim=d,
        dataset=args.dataset, random_seed=args.seed,
        generative_model="DLPM", exp_no="pretrain",
        expname=Path(args.out).stem, nu=None,
        X_=x_all, Y_=Y_,
        L=None, lam=None, alpha=ALPHA, formulation=None,
        f="KL", Gamma="DLPM", constraint=None, no_cvar=True,
        beta=None, sigma_P=None, sigma_Q=None, interval_length=None,
        label=None, pts_P=None, pts_Q=None, pts_P_2=None, pts_Q_2=None,
        y0=None,
    )
    telemetry = dict(trajectories=[Y_], vectorfields=[], QoIs=[], divergences=[],
                     KE_Ps=[], FIDs=[], wasserstein1s=[], comp_times=[], cvar_report=[])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump([params, telemetry], f)
    print(f"[DLPM] pretrain pickle written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
