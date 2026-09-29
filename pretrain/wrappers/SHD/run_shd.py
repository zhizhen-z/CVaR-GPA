"""
Train SHD (Score-based Heavy-tailed Diffusion; "Learning to Simulate from
Heavy-tailed Distribution via Diffusion Model") on a target sample set and
emit a pickle in the pretrain schema for CVaR-GPA fine-tuning.

pretrain/settings/SHD.md lists the source of every setting and the deviations ledger.

Protocol: unconditional single-step structure (their base_stock.yaml,
condition_L=0/target_L=1) + heavy-tail diffusion block (their
base_pareto.yaml: student_t noise t_param=3.5, student_t start noise,
non-homogeneous sigma scaling) + pareto optimizer with epochs=352 for an
exact matched gradient-step budget (1760 steps) at N=5000. reflect_normalize for symmetric
synthetic targets, plain normalize for FF25.
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "third_party" / "heavy_tail_diffusion"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "TTF"))  # byte-equal targets

from main_model import Score_Based_Diffusion  # noqa: E402
from utils import train as shd_train  # noqa: E402
from run_ttf import sample_target, DATASETS as TTF_DATASETS  # noqa: E402

GLOBAL_SEED = 1  # reference exe scripts default --seed 1 (never applied there;
                 # we apply it — deviations ledger item 4)

NORMALIZE_METHOD = {
    "student_t_2D": "reflect_normalize",     # symmetric marginals
    "student_t_2D_nu1.2": "reflect_normalize",
    "student_t_2D_nu1.5": "reflect_normalize",
    "student_t_2D_nu1.8": "reflect_normalize",
    "FF25_monthly": "normalize",            # real returns, keep skew
    "Streamflow_ohio": "normalize",    # specific discharge q = Q/A, mm/day
    "Neal_funnel": "reflect_normalize",         # both marginals symmetric about 0
}


def build_config(dataset: str, d: int, epochs: int, protocol: str = "pareto",
                 n_train: int | None = None) -> dict:
    """protocol="pareto": base_pareto.yaml diffusion/optimizer + stock's
    unconditional structure (the original runs; see pretrain/settings/SHD.md).

    protocol="stock": the paper's own
    real-returns protocol, App. D.5.2: "We set nu = 3.5 for the student-t
    noises added to the real samples, T = 20 noise levels, sigma_1 = 0.01,
    sigma_T = 0.8, ... epochs 300, batch size 1200 (entire training set in
    each epoch), Adam 1e-2 decaying to 1e-5". base_stock.yaml carries those
    numbers but says noise_type/start_noise "gaussian"; the paper text says
    student-t, so we override those two keys to student_t (t_param 3.5 is
    already in the yaml). Full batch = our train split size (n_train).
    Lt=100 (n_steps_each), sigma_scale False are the yaml's stock values.
    """
    name = "base_pareto.yaml" if protocol == "pareto" else "base_stock.yaml"
    with open(REPO / "config" / name) as f:
        config = yaml.safe_load(f)
    config["data"]["target_dim"] = d
    config["data"]["condition_L"] = 0   # stock structure: fully unconditional
    config["data"]["target_L"] = 1
    config["data"]["normalize_method"] = NORMALIZE_METHOD[dataset]
    config["model"]["num_sample_features"] = d
    if protocol == "pareto":
        config["train"]["epochs"] = epochs  # matched-step budget
    else:
        assert n_train is not None
        config["diffusion"]["noise_type"] = "student_t"    # paper D.5.2
        config["diffusion"]["start_noise"] = "student_t"   # paper D.5.2
        config["train"]["batch_size"] = int(n_train)       # full batch
        config["train"]["epochs"] = epochs                 # 300 per paper
    return config


class VectorDataset(torch.utils.data.Dataset):
    """Mirror of Forecasting_Dataset (dataset_forecasting.py:8-57) for
    in-memory (N, 1, K) data. Normalization stats over the FULL array
    (their lines 16-25), 90/10 sequential split (their lines 30-37)."""

    def __init__(self, main_data: np.ndarray, mode: str, method: str):
        self.L, self.K = main_data.shape[1], main_data.shape[2]
        if method == "normalize":
            self.mean_data = main_data.mean((0,))
            self.std_data = main_data.std((0,))
        elif method == "reflect_normalize":
            self.mean_data = np.zeros(main_data.shape[1:])
            self.std_data = np.sqrt((main_data ** 2).mean((0,)))
        else:
            raise ValueError(method)
        self.data = (main_data - self.mean_data) / self.std_data

        n = len(self.data)
        if mode == "train":
            self.use_index = np.arange(0, int(n * 0.9))
        else:
            self.use_index = np.arange(int(n * 0.9), n)

    def __getitem__(self, i):
        idx = self.use_index[i]
        target_mask = np.ones((self.L, self.K))
        target_mask[0:, :] = 0.0  # condition_L=0 -> all-zero gt_mask
        return {
            "observed_data": self.data[idx],
            "observed_mask": np.ones((self.L, self.K)),
            "gt_mask": target_mask,
            "timepoints": np.arange(self.L) * 1.0,
            "feature_id": np.arange(self.K) * 1.0,
        }

    def __len__(self):
        return len(self.use_index)


def generate_samples(model, M: int, d: int, device) -> np.ndarray:
    """Mirror Score_Based_Diffusion.evaluate + generate with all-zero
    cond_mask (fully unconditional). Returns (M, d) in normalized units."""
    model.eval()
    model.target_dim = model.target_dim_base
    observed_data = torch.zeros(M, d, 1).to(device)   # (B, K, L) shape only
    cond_mask = torch.zeros(M, d, 1).to(device)
    observed_tp = torch.zeros(M, 1).to(device)
    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples = model.generate(observed_data, cond_mask, side_info, n_samples=1)
    return samples[:, 0, :, 0].cpu().numpy()  # (B, K)


def save_pretrain_pickle(out_path: Path, X_, Y_, dataset, N, M, seed):
    params = dict(
        N_samples_Q=N, N_samples_P=M, N_dim=X_.shape[1],
        dataset=dataset, random_seed=seed,
        generative_model="SHD", exp_no="pretrain",
        expname=out_path.stem, nu=None,
        X_=X_, Y_=Y_,
        L=None, lam=None, alpha=None, formulation=None,
        f="KL", Gamma="SHD", constraint=None, no_cvar=True,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(NORMALIZE_METHOD.keys()))
    ap.add_argument("--seed", type=int, default=0, help="target-sample seed")
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--protocol", default="pareto", choices=["pareto", "stock"],
                    help="pareto: original runs. stock: paper D.5.2 real-returns "
                         "protocol (student-t 3.5 noise, sigma 0.8->0.01, Lt 100, "
                         "lr 1e-2->1e-5, 300 full-batch epochs).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="pareto: 352 x 5 batches = 1760 steps (their budget, "
                         "matched to the reference). stock: 300 (paper D.5.2).")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = TTF_DATASETS[args.dataset]["dim"]
    if args.epochs is None:
        args.epochs = 352 if args.protocol == "pareto" else 300
    n_train = int(args.N * 0.9)  # VectorDataset 90/10 sequential split
    config = build_config(args.dataset, d, args.epochs, args.protocol, n_train)
    print(f"[SHD] protocol={args.protocol} train={config['train']} "
          f"diffusion={config['diffusion']}", flush=True)
    print(f"[SHD] dataset={args.dataset} d={d} N={args.N} M={args.M} "
          f"device={device} epochs={args.epochs} "
          f"normalize={config['data']['normalize_method']} "
          f"t_param={config['diffusion']['t_param']}", flush=True)

    # -- 1. Target (byte-equal to the other baselines) --
    x_all = sample_target(args.dataset, args.N, args.seed).astype(np.float64)
    r = np.linalg.norm(x_all, axis=1)
    print(f"[SHD] target: shape={x_all.shape}, ||x|| P50={np.median(r):.2f} "
          f"P99={np.quantile(r, 0.99):.2f} max={r.max():.2e}", flush=True)

    # -- 2. Data (N, L=1, K) --
    main_data = x_all[:, None, :]
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    trn_ds = VectorDataset(main_data, "train", config["data"]["normalize_method"])
    val_ds = VectorDataset(main_data, "valid", config["data"]["normalize_method"])
    trn_loader = torch.utils.data.DataLoader(
        trn_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["train"]["batch_size"], shuffle=False)
    print(f"[SHD] split: trn={len(trn_ds)} val={len(val_ds)}; "
          f"std range=[{trn_ds.std_data.min():.3f}, {trn_ds.std_data.max():.3f}]",
          flush=True)

    # -- 3. Model + train (their utils.train, last-epoch weights kept) --
    model = Score_Based_Diffusion(config, device, d).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SHD] model built: {n_params} trainable params", flush=True)
    # per-run weights dir (their utils.train writes foldername/model.pth)
    out_dir = Path(args.out).parent / f"weights_{Path(args.out).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shd_train(model, config["train"], trn_loader, valid_loader=val_loader,
              foldername=str(out_dir))

    # -- 4. Sample + de-normalize --
    Y_norm = generate_samples(model, args.M, d, device)
    Y_ = (Y_norm * trn_ds.std_data[0] + trn_ds.mean_data[0]).astype(np.float32)
    rY = np.linalg.norm(Y_, axis=1)
    print(f"[SHD] emitted {Y_.shape[0]} particles: ||y|| P50={np.median(rY):.2f} "
          f"P99={np.quantile(rY, 0.99):.2f} max={rY.max():.2e}", flush=True)

    # -- 5. Save --
    save_pretrain_pickle(Path(args.out), X_=x_all, Y_=Y_,
                         dataset=args.dataset, N=args.N, M=args.M, seed=args.seed)
    print(f"[SHD] pretrain pickle written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
