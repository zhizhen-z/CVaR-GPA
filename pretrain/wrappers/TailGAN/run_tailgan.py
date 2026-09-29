"""
Train Tail-GAN (Cont, Cucuringu, Xu & Zhang, Management Science 2025) on a
target sample set and emit a pickle in the pretrain schema so
the CVaR-GPA driver can fine-tune on top of it.

Settings with reference file:line citations; pretrain/settings/TailGAN.md has the full list
and the deviations ledger. Summary:

VERBATIM from reference (third_party/Tail-GAN/TailGAN.py @ bfbe950):
  - Generator MLP 1000->128->256->512->1024->d, LeakyReLU(0.2),
    BatchNorm1d(_, 0.8), output clamp [-1, 1]            (TailGAN.py:162-185)
  - Discriminator: sorted-PnL batch -> MLP batch->256->128->2,
    NeuralSort temp=0.01, projection W=10, project=True  (TailGAN.py:189-230)
  - Loss: S_quant elicitability score, W=10              (TailGAN.py:271-303)
  - Adam lr_G=1e-6, lr_D=1e-7, betas=(0.5, 0.999)        (TailGAN.py:36-40,324-325)
  - batch_size=1000, n_epochs=3000, n_critic 1/1, no early stop, no val split
  - latent z in R^1000 ~ iid Student-t df=5 ("t5")       (TailGAN.py:41,56,340-345)
  - alphas=[0.05]                                        (TailGAN.py:57)
  - Cap=10                                               (TailGAN.py:50)
  - Static portfolios: LShort, n_trans=50,
    sparse.random(d, max(d^2, 50), density=0.9, N(0,1)),
    column-normalized, position-scaled                   (gen_static_port.py:53-66)
  - Ensemble: numNN=10, keep members with final G-loss <= median
    (Screen_Ensemble intent; reference impl crashes)     (TailGAN.py:441-455)
  - Global seed=1; torch.manual_seed reset per member (shared init)
                                                         (TailGAN.py:28-30,312)

ADAPTATIONS (deviations ledger in pretrain/settings/TailGAN.md):
  1. n_cols=1 single-period reduction; strategies = per-asset buy&hold +
     50 static portfolios; MR/TF undefined without a time axis.
  2. Affine scaling into the clamp range: x_scaled = (x - mean)/(std * K),
     K chosen so max|x_scaled| = 0.9; inverted on emitted samples.
  3. N=5000 target samples (comparison convention), batch 1000 -> 5 batches.
     UPDATE BUDGET: the reference trains on len=50000 paths = 50 batches/epoch,
     so 3000 epochs = 150,000 G/D updates. At N=5000 the same 3000 epochs is
     only 15,000 updates (the original TailGAN_*/TailGANpar_* pickles). The
     150k-update rerun (TailGAN150k_*) uses n_epochs=30000 at
     N=5000 (150000 at raw N=1182, 1 batch/epoch) to match THEIR update count.
  2b. NOTE on adaptation 2: the reference applies NO scaling (its synthetic
     returns have per-step std ~2.5e-3 and sit inside the generator's
     clamp(-1,1) as-is). Our max-based K is a wrapper choice; on nu<=2 targets
     std and max|z| are set by the single largest draw (K~75 on 2D Cauchy),
     which places the target bulk ~1e-4 in clamp units. Disclosed, not tuned.
  4. Ensemble member m draws z from np.random.default_rng(1000 + m) instead
     of shared global numpy state (reproducibility; reference members differ
     only through numpy side-state anyway).
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse, stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "third_party" / "Tail-GAN"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "TTF"))  # for sample_target (byte-equal targets)

from util import deterministic_NeuralSort  # noqa: E402  (Tail-GAN repo)
from run_ttf import sample_target, DATASETS as TTF_DATASETS  # noqa: E402

GLOBAL_SEED = 1          # TailGAN.py:28
LATENT_DIM = 1000        # TailGAN.py:41
NOISE_DF = 5             # noise_name="t5", TailGAN.py:56,340-345
N_EPOCHS = 3000          # TailGAN.py:34
BATCH_SIZE = 1000        # TailGAN.py:35
LR_D, LR_G = 1e-7, 1e-6  # TailGAN.py:36-37
B1, B2 = 0.5, 0.999      # TailGAN.py:39-40
TEMP = 0.01              # TailGAN.py:38
ALPHAS = [0.05]          # TailGAN.py:57
W_CONST = 10.0           # TailGAN.py:58
CAP = 10                 # TailGAN.py:50
N_TRANS = 50             # TailGAN.py:49
NUM_NN = 10              # TailGAN.py:60
CLAMP_TARGET = 0.9       # adaptation 2: headroom inside the [-1,1] clamp

DATASET_DIMS = {k: v["dim"] for k, v in TTF_DATASETS.items()}


# ---------------------------------------------------------------------------
# Static portfolios — port of gen_static_port.Gen_StaticPort (LShort branch,
# lines 53-66), matrix built in-process instead of loaded from disk.
# ---------------------------------------------------------------------------
def gen_static_portfolios(d: int, n_ports: int, rng_seed: int) -> np.ndarray:
    np.random.seed(rng_seed)  # scipy.sparse.random uses global np RNG
    rvs = stats.norm(loc=0, scale=1).rvs
    unscale = sparse.random(d, max(int(d ** 2), n_ports), density=0.9,
                            data_rvs=rvs).toarray()
    price_start = np.ones(d)
    scale_mat = unscale / np.abs(unscale).sum(0)
    position = np.diag(1 / price_start.dot(np.abs(scale_mat)))
    trans_mat = np.dot(scale_mat, position)

    def is_stock(vec):
        return np.all((np.abs(vec) == 1.0) + (np.abs(vec) == 0.0))

    trans_mat_port = trans_mat[:, ~np.apply_along_axis(is_stock, 0, trans_mat)]
    # Reference asserts >= 10 (gen_static_port.py:67), then StaticPort slices
    # [:, :n_trans] which silently truncates when fewer survive (d=2 case).
    assert trans_mat_port.shape[1] >= 10, "not enough portfolio columns"
    return trans_mat_port[:, :n_ports]


# ---------------------------------------------------------------------------
# PnL — n_cols=1 reduction of Compute_PNL (TailGAN.py:127-158).
# BuyHold(Inc2Price(R), Cap) with a length-1 return path is Cap * r for the
# assets and Cap * (w . r) for a static portfolio; MR/TF undefined.
# ---------------------------------------------------------------------------
def compute_pnl(R: torch.Tensor, trans_mat: torch.Tensor) -> torch.Tensor:
    """R: (batch, d). Returns PnL (batch, d + n_trans)."""
    pnl_assets = CAP * R
    pnl_ports = CAP * (R @ trans_mat)
    return torch.cat([pnl_assets, pnl_ports], dim=1)


# ---------------------------------------------------------------------------
# Networks — verbatim ports (TailGAN.py:162-230) with R_shape=(d, 1) folded
# to vectors.
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, d: int):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(LATENT_DIM, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, d),
        )

    def forward(self, z):
        out = self.model(z)
        return torch.clamp(out, min=-1, max=1)


class Discriminator(nn.Module):
    def __init__(self, trans_mat: torch.Tensor):
        super().__init__()
        self.trans_mat = trans_mat
        self.model = nn.Sequential(
            nn.Linear(BATCH_SIZE, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(ALPHAS)),
        )

    def project_op(self, validity):
        # TailGAN.py:208-215
        for i, alpha in enumerate(ALPHAS):
            v = validity[:, 2 * i].clone()
            e = validity[:, 2 * i + 1].clone()
            indicator = torch.sign(torch.as_tensor(0.5 - alpha))
            validity[:, 2 * i] = indicator * (
                (W_CONST * v < e).float() * v
                + (W_CONST * v >= e).float() * (v + W_CONST * e) / (1 + W_CONST ** 2))
            validity[:, 2 * i + 1] = indicator * (
                (W_CONST * v < e).float() * e
                + (W_CONST * v >= e).float() * W_CONST * (v + W_CONST * e) / (1 + W_CONST ** 2))
        return validity

    def forward(self, R):
        # TailGAN.py:218-230
        PNL = compute_pnl(R, self.trans_mat)
        PNL_t = PNL.T
        PNL_s = PNL_t.reshape(*PNL_t.shape, 1)
        perm = deterministic_NeuralSort(PNL_s, TEMP)
        PNL_sort = torch.bmm(perm, PNL_s)
        validity = self.model(PNL_sort.reshape(*PNL_t.shape))
        validity = self.project_op(validity)
        return PNL, validity


# ---------------------------------------------------------------------------
# Score — verbatim S_quant (TailGAN.py:245-303)
# ---------------------------------------------------------------------------
def G1_quant(v, W=W_CONST):
    return -W * v ** 2 / 2


def G2_quant(e, alpha):
    return alpha * e


def G2in_quant(e, alpha):
    return alpha * e ** 2 / 2


def S_quant(v, e, X, alpha, W=W_CONST):
    if alpha < 0.5:
        rt = (((X <= v).float() - alpha) * (G1_quant(v, W) - G1_quant(X, W))
              + 1. / alpha * G2_quant(e, alpha) * (X <= v).float() * (v - X)
              + G2_quant(e, alpha) * (e - v) - G2in_quant(e, alpha))
    else:
        ai = 1 - alpha
        rt = (((X >= v).float() - ai) * (G1_quant(v, W) - G1_quant(X, W))
              + 1. / ai * G2_quant(-e, ai) * (X >= v).float() * (X - v)
              + G2_quant(-e, ai) * (v - e) - G2in_quant(-e, ai))
    return torch.mean(rt)


def score(validity, PNL):
    loss = 0
    for i, alpha in enumerate(ALPHAS):
        v = validity[:, [2 * i]]
        e = validity[:, [2 * i + 1]]
        loss = loss + S_quant(v, e, PNL.T, alpha)
    return loss


# ---------------------------------------------------------------------------
# Training — verbatim port of Train_Single (TailGAN.py:309-421) minus the
# per-epoch npy dumps; returns model + final-epoch G loss for screening.
# ---------------------------------------------------------------------------
def train_single(x_scaled: torch.Tensor, trans_mat: torch.Tensor,
                 member: int, device, label: str, n_epochs: int = N_EPOCHS):
    torch.manual_seed(GLOBAL_SEED)  # TailGAN.py:312 — members share init
    noise_rng = np.random.default_rng(1000 + member)  # adaptation 4

    d = x_scaled.shape[1]
    generator = Generator(d).to(device)
    discriminator = Discriminator(trans_mat.to(device)).to(device)

    opt_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(B1, B2))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(B1, B2))

    dataset = torch.utils.data.TensorDataset(x_scaled)
    # drop_last: the discriminator's input dimension IS the batch size (it
    # consumes the sorted per-strategy PnL across the batch), so a partial
    # trailing batch cannot be forwarded. No-op at N=5000 (5 exact batches);
    # required for the raw-FF25 diagnostic at N=1182 (one 1000-sample batch
    # per epoch, random 1000-of-1182 without replacement each epoch).
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, drop_last=True)

    g_losses_last_epoch = []
    for epoch in range(n_epochs):
        g_losses_last_epoch = []
        for (real_R,) in loader:
            real_R = real_R.to(device)
            z = torch.as_tensor(
                noise_rng.standard_t(NOISE_DF, (real_R.shape[0], LATENT_DIM)),
                dtype=torch.float32, device=device)
            gen_R = generator(z)

            # Discriminator step (TailGAN.py:354-369)
            opt_D.zero_grad()
            PNL, validity = discriminator(real_R)
            _, gen_validity = discriminator(gen_R)
            loss_D = score(validity, PNL) - score(gen_validity, PNL)
            loss_D.backward(retain_graph=True)
            opt_D.step()

            # Generator step (TailGAN.py:372-388)
            opt_G.zero_grad()
            _, gen_validity = discriminator(gen_R)
            loss_G = score(gen_validity, PNL)
            loss_G.backward()
            opt_G.step()
            g_losses_last_epoch.append(loss_G.item())

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"[TailGAN] {label} m{member} epoch={epoch} "
                  f"D={loss_D.item():.6f} G={np.mean(g_losses_last_epoch):.6f}",
                  flush=True)

    return generator, float(np.mean(g_losses_last_epoch))


GEN_SIZE = 1000  # TailGAN.py:330 — reference samples in batches of 1000


def sample_member(generator, n: int, member: int, device) -> np.ndarray:
    """Sample n particles in GEN_SIZE batches. NOTE: the reference never
    switches the generator to eval mode (no .eval() anywhere in TailGAN.py);
    its saved samples come from a train-mode model, so BatchNorm uses the
    generation batch's own statistics. We match that."""
    rng = np.random.default_rng(5000 + member)
    generator.train()
    chunks = []
    with torch.no_grad():
        remaining = n
        while remaining > 0:
            b = min(GEN_SIZE, remaining)
            z = torch.as_tensor(rng.standard_t(NOISE_DF, (b, LATENT_DIM)),
                                dtype=torch.float32, device=device)
            chunks.append(generator(z).cpu().numpy())
            remaining -= b
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Pickle schema — matches run_ttf.save_pretrain_pickle
# ---------------------------------------------------------------------------
def save_pretrain_pickle(out_path: Path, X_, Y_, dataset, N, M, seed):
    params = dict(
        N_samples_Q=N, N_samples_P=M, N_dim=X_.shape[1],
        dataset=dataset, random_seed=seed,
        generative_model="TailGAN", exp_no="pretrain",
        expname=out_path.stem, nu=None,
        X_=X_, Y_=Y_,
        L=None, lam=None, alpha=None, formulation=None,
        f="KL", Gamma="TailGAN", constraint=None, no_cvar=True,
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


def run_member(args, x_scaled_np, trans_mat, scale_K, std, mean, device):
    """Parallel mode: train ONE ensemble member, save its samples + G loss.
    Members are independent in the reference (same torch init seed, own
    numpy noise stream), so per-member jobs reproduce the sequential run."""
    x_t = torch.as_tensor(x_scaled_np, dtype=torch.float32)
    gen, g_loss = train_single(x_t, trans_mat, args.member, device,
                               label=args.dataset, n_epochs=args.n_epochs)
    # Store 3x GEN_SIZE so the merge can fill M=5000 even if fewer than 5
    # members survive screening (chunks stay reference-sized batches).
    Y_scaled = sample_member(gen, 3 * GEN_SIZE, args.member, device)
    out = Path(args.member_dir) / f"member_{args.member:02d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, Y_scaled=Y_scaled, g_loss=g_loss, K=scale_K, std=std, mean=mean)
    print(f"[TailGAN] member {args.member}: G loss {g_loss:.6f} -> {out}", flush=True)


def run_merge(args, x_all):
    """Screen members (final G loss <= median, Screen_Ensemble intent) and
    assemble the pretrain pickle: layered fill of GEN_SIZE blocks from kept
    members in index order — identical to the sequential path when 5 survive."""
    mdir = Path(args.member_dir)
    members = []
    for m in range(args.num_nn):
        f = mdir / f"member_{m:02d}.npz"
        assert f.exists(), f"missing member file: {f}"
        members.append(np.load(f))
    losses = np.array([float(d["g_loss"]) for d in members])
    thresh = np.percentile(losses, 50)
    kept = [m for m in range(args.num_nn) if losses[m] <= thresh]
    print(f"[TailGAN] merge: losses={np.round(losses, 6).tolist()}", flush=True)
    print(f"[TailGAN] merge: kept {kept} (<= {thresh:.6f})", flush=True)

    blocks = []
    max_layers = min(members[m]["Y_scaled"].shape[0] // GEN_SIZE for m in kept)
    for layer in range(max_layers):
        for m in kept:
            Y = members[m]["Y_scaled"]
            blocks.append(Y[layer * GEN_SIZE:(layer + 1) * GEN_SIZE])
        if sum(b.shape[0] for b in blocks) >= args.M:
            break
    total = sum(b.shape[0] for b in blocks)
    assert total >= args.M, f"only {total} samples from kept members, need {args.M}"
    Y_scaled = np.concatenate(blocks, axis=0)[:args.M]

    d0 = members[kept[0]]
    Y_ = (Y_scaled * float(d0["K"]) * d0["std"].astype(np.float32)
          + d0["mean"].astype(np.float32))
    rY = np.linalg.norm(Y_, axis=1)
    print(f"[TailGAN] merged {Y_.shape[0]} particles: ||y|| P50={np.median(rY):.2f} "
          f"P99={np.quantile(rY, 0.99):.2f} max={rY.max():.2e}", flush=True)
    save_pretrain_pickle(Path(args.out), X_=x_all.astype(np.float64), Y_=Y_,
                         dataset=args.dataset, N=args.N, M=args.M, seed=args.seed)
    print(f"[TailGAN] pretrain pickle written: {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DIMS.keys()))
    ap.add_argument("--seed", type=int, default=0, help="target-sample seed (comparison convention)")
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--num_nn", type=int, default=NUM_NN)
    ap.add_argument("--member", type=int, default=None,
                    help="parallel mode: train only this ensemble member")
    ap.add_argument("--member_dir", type=str, default=None,
                    help="directory for per-member npz files")
    ap.add_argument("--merge", action="store_true",
                    help="merge mode: screen + assemble pickle from member_dir")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    n_epochs, num_nn = args.n_epochs, args.num_nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = DATASET_DIMS[args.dataset]
    print(f"[TailGAN] dataset={args.dataset} d={d} N={args.N} M={args.M} "
          f"device={device} epochs={n_epochs} ensemble={num_nn}", flush=True)

    # -- 1. Target (byte-equal to Lip-KL / TTF comparisons) --
    x_all = sample_target(args.dataset, args.N, args.seed).astype(np.float64)
    r = np.linalg.norm(x_all, axis=1)
    print(f"[TailGAN] target: shape={x_all.shape}, ||x|| P50={np.median(r):.2f} "
          f"P99={np.quantile(r, 0.99):.2f} max={r.max():.2e}", flush=True)

    # -- 2. Scaling into the clamp range (adaptation 2) --
    mean = x_all.mean(axis=0)
    std = x_all.std(axis=0, ddof=1)
    z0 = (x_all - mean) / std
    K = np.abs(z0).max() / CLAMP_TARGET
    x_scaled = z0 / K
    print(f"[TailGAN] scaling: K={K:.4f}, max|x_scaled|={np.abs(x_scaled).max():.3f}",
          flush=True)

    if args.merge:
        assert args.member_dir and args.out, "--merge needs --member_dir and --out"
        run_merge(args, x_all)
        return

    # -- 3. Static portfolios (LShort, seeded with the global seed) --
    trans_mat_np = gen_static_portfolios(d, N_TRANS, rng_seed=GLOBAL_SEED)
    trans_mat = torch.as_tensor(trans_mat_np, dtype=torch.float32)
    print(f"[TailGAN] static portfolios: {trans_mat_np.shape}", flush=True)

    if args.member is not None:
        assert args.member_dir, "--member needs --member_dir"
        run_member(args, x_scaled, trans_mat, K, std, mean, device)
        return

    # -- 4. Train ensemble, screen, sample --
    assert args.out, "sequential mode needs --out"
    x_t = torch.as_tensor(x_scaled, dtype=torch.float32)
    members = []
    for m in range(num_nn):
        gen_m, g_loss = train_single(x_t, trans_mat, m, device,
                                     label=args.dataset, n_epochs=n_epochs)
        members.append((gen_m, g_loss))
        print(f"[TailGAN] member {m}: final G loss {g_loss:.6f}", flush=True)

    losses = np.array([gl for _, gl in members])
    thresh = np.percentile(losses, 50)  # Screen_Ensemble(thres_perc=50)
    kept = [i for i in range(num_nn) if losses[i] <= thresh]
    print(f"[TailGAN] screening: kept {kept} (loss<= {thresh:.6f})", flush=True)

    per = int(np.ceil(args.M / len(kept)))
    chunks = [sample_member(members[i][0], per, i, device) for i in kept]
    Y_scaled = np.concatenate(chunks, axis=0)[:args.M]

    # -- 5. De-scale, report, save --
    Y_ = (Y_scaled * K * std + mean).astype(np.float32)
    rY = np.linalg.norm(Y_, axis=1)
    print(f"[TailGAN] emitted {Y_.shape[0]} particles: ||y|| P50={np.median(rY):.2f} "
          f"P99={np.quantile(rY, 0.99):.2f} max={rY.max():.2e}", flush=True)

    save_pretrain_pickle(Path(args.out), X_=x_all, Y_=Y_,
                         dataset=args.dataset, N=args.N, M=args.M, seed=args.seed)
    print(f"[TailGAN] pretrain pickle written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
