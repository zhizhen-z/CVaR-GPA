"""
t-EDM (Pandey et al., "Heavy-Tailed Diffusion Models", ICLR 2025,
arXiv:2410.14171) — clean-room implementation for the baseline
comparison, written from the paper (no official code is available). pretrain/settings/tEDM.md
has the full specification, the paper equation citations and the deviations ledger.

Method summary (all cites = paper):
  train:  sigma ~ LogNormal(-1.2, 1.2); n = sigma * eps / sqrt(kappa),
          eps ~ N(0,I), scalar kappa ~ chi2(nu)/nu per sample (Sec 2.2, Alg 1);
          sigma_t = sigma * sqrt(nu/(nu-2)) (Alg 1 line 5); EDM preconditioning
          at sigma_t (App Eqs 92/100/101); loss = ||D - x0||^2 / c_out^2 (Eq 12).
  sample: Heun on dx/dt = (x - D(x,t))/t, Karras rho=7 grid, N=18 steps,
          sigma_max=80 -> sigma_min=0.002, t_N=0; init x ~ t_d(0, smax^2 I, nu)
          (Alg 2 — the only sampler delta vs EDM).
Data z-scored (sigma_data = 1.0), de-standardized on output.

Two additions (a scalar finite --nu runs the code path above unchanged):
  --nu inf        plain Gaussian EDM (kappa = 1, no rescaling): the "Gaussian Diffusion" column of the paper's Fig. 1.
  --nu 4,20       per-coordinate nu (the paper's toy setting, App. C.3: "we tune nu for each individual dimension",
                  x1 -> 20, x2 -> {4,7,10}). The paper leaves the per-dimension mechanics unspecified; we use the
                  simplest consistent reading: independent per-coordinate kappa_j ~ chi2(nu_j)/nu_j, so the noise is a
                  product of univariate Student-t's, sigma_t is per coordinate (sigma*sqrt(nu_j/(nu_j-2))), the EDM
                  coefficients c_in/c_skip/c_out and the loss weight are per coordinate, and the network's noise input
                  is the shared raw sigma, c_noise = ln(sigma)/4 (Table 7's convention). Sampling initialises x_j ~
                  t(0, sigma_max^2, nu_j) and conditions on t_k*sqrt(nu_j/(nu_j-2)) per coordinate ("consistent").
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "TTF"))  # byte-equal targets
from run_ttf import sample_target, DATASETS as TTF_DATASETS  # noqa: E402

# App Tables 5/8 (toy config)
SIGMA_DATA = 1.0
P_MEAN, P_STD = -1.2, 1.2
SIGMA_MAX, SIGMA_MIN, RHO, N_STEPS = 80.0, 0.002, 7.0, 18
LR = 1e-3            # "EDM training hyperparameters"
BATCH = 512
SAMPLES_SEEN = 3_000_000   # deviations ledger item 3
HIDDEN = {2: 64, 5: 128, 25: 256, 64: 256, 312: 256}  # d=2 = their toy verbatim; rest ours (64: same width as d=25)


def sample_t(shape_bk, nu, generator=None, device="cpu"):
    """t_d(0, I, nu) via x = z / sqrt(kappa), scalar kappa per row (Sec 2.2).
    nu = inf -> Gaussian (kappa = 1). nu = sequence of length d -> independent per-coordinate kappa_j."""
    B = shape_bk[0]
    z = torch.randn(*shape_bk, generator=generator, device=device)
    if isinstance(nu, (list, tuple, np.ndarray)):
        cols = []
        for j, nj in enumerate(nu):
            if np.isinf(nj):
                cols.append(torch.ones(B, 1, device=device))
            else:
                cols.append(torch.distributions.Chi2(torch.tensor(float(nj))).sample((B, 1)).to(device) / nj)
        kappa = torch.cat(cols, dim=1)                      # (B, d)
        return z / torch.sqrt(kappa)
    if np.isinf(nu):
        return z
    chi2 = torch.distributions.Chi2(torch.tensor(float(nu)))
    kappa = chi2.sample((B, 1)).to(device) / nu
    return z / torch.sqrt(kappa)


def nu_scale(nu):
    """sqrt(nu/(nu-2)) elementwise; 1 for nu = inf. Returns a float for scalar nu, a (d,) float64 array for a vector."""
    if isinstance(nu, (list, tuple, np.ndarray)):
        return np.array([1.0 if np.isinf(n) else np.sqrt(n / (n - 2)) for n in nu], dtype=np.float64)
    return 1.0 if np.isinf(nu) else float(np.sqrt(nu / (nu - 2)))


class Denoiser(nn.Module):
    """EDM-preconditioned MLP: D = c_skip*x + c_out*F(c_in*x, c_noise)."""

    def __init__(self, d: int, width: int):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(d + 1, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, d),
        )

    def forward(self, x, sigma_t, c_noise=None):
        """x: (B, d); sigma_t: (B, 1) — the rescaled sigma*sqrt(nu/(nu-2)).
        Per-coordinate nu: sigma_t is (B, d) and c_noise (B, 1) = ln(raw sigma)/4 must be given."""
        c_in = 1.0 / torch.sqrt(sigma_t ** 2 + SIGMA_DATA ** 2)      # Eq 92
        c_skip = SIGMA_DATA ** 2 / (sigma_t ** 2 + SIGMA_DATA ** 2)  # Eq 100
        c_out = sigma_t * SIGMA_DATA / torch.sqrt(sigma_t ** 2 + SIGMA_DATA ** 2)  # Eq 101
        if c_noise is None:
            c_noise = torch.log(sigma_t) / 4.0                       # Tables 5/8
        F = self.net(torch.cat([c_in * x, c_noise], dim=1))
        return c_skip * x + c_out * F


def train_tedm(x_data: torch.Tensor, nu: float, device, label: str,
               samples_seen: int = SAMPLES_SEEN):
    d = x_data.shape[1]
    model = Denoiser(d, HIDDEN[d]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = samples_seen // BATCH
    per_dim = isinstance(nu, (list, tuple, np.ndarray))
    scale = nu_scale(nu)
    if per_dim:
        assert len(nu) == d, (nu, d)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=device).view(1, d)

    for step in range(n_steps):
        idx = torch.randint(0, x_data.shape[0], (BATCH,), device=device)
        x0 = x_data[idx]
        sigma = torch.exp(P_MEAN + P_STD * torch.randn(BATCH, 1, device=device))
        n = sigma * sample_t((BATCH, d), nu, device=device)   # Alg 1 line 4
        sigma_t = sigma * scale                               # Alg 1 line 5 ((B,1) scalar nu; (B,d) per-dim nu)
        D = model(x0 + n, sigma_t, c_noise=(torch.log(sigma) / 4.0 if per_dim else None))
        c_out2 = (sigma_t * SIGMA_DATA) ** 2 / (sigma_t ** 2 + SIGMA_DATA ** 2)
        loss = (((D - x0) ** 2) / c_out2).sum(dim=1).mean()   # Eq 12
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == n_steps - 1:
            print(f"[tEDM] {label} nu={nu} step={step}/{n_steps} loss={loss.item():.4f}",
                  flush=True)
    return model


@torch.no_grad()
def sample_tedm(model, M: int, d: int, nu: float, device,
                sampler_cond: str = "consistent") -> np.ndarray:
    """Alg 2: deterministic Heun, Karras grid, t-distributed init.

    sampler_cond (deviations ledger item 8):
      "consistent" - the denoiser is conditioned on the ACTUAL std of the
          state, t_k*sqrt(nu/(nu-2)). This is what the paper's coefficients
          (Eq 92/100/101, Table 5: c_in = 1/sqrt(nu/(nu-2) sigma^2 + sigma_data^2)
          in raw sigma) imply, and it matches the training convention used
          here (Alg 1 line 5: condition on sigma*sqrt(nu/(nu-2)) = actual std).
      "literal" - the pre-fix behaviour: condition on the raw grid value t_k
          while the state has std t_k*sqrt(nu/(nu-2)); the network then
          under-denoises and the output tails inflate (2D ablation: P99 993
          vs 125 at nu=3, target 111). Kept only to reproduce old pickles.
    """
    i = torch.arange(N_STEPS, device=device, dtype=torch.float64)
    t = (SIGMA_MAX ** (1 / RHO)
         + i / (N_STEPS - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    t = torch.cat([t, torch.zeros(1, device=device)]).float()  # t_N = 0
    per_dim = isinstance(nu, (list, tuple, np.ndarray))
    if per_dim:
        assert sampler_cond == "consistent", "per-dimension nu is only defined with the consistent conditioning"
        cond = torch.as_tensor(nu_scale(nu), dtype=torch.float32, device=device).view(1, d)
    else:
        cond = nu_scale(nu) if sampler_cond == "consistent" else 1.0
    def _cn(tk):                                                # network noise input for the per-dim path: raw t_k
        return (torch.log(tk) / 4.0).expand(M, 1) if per_dim else None

    x = SIGMA_MAX * sample_t((M, d), nu, device=device)        # Alg 2 line 1
    for k in range(N_STEPS):
        tk, tk1 = t[k], t[k + 1]
        sig = (tk * cond).expand(M, d) if per_dim else (tk * cond).expand(M, 1)
        dk = (x - model(x, sig, c_noise=_cn(tk))) / tk
        x_next = x + (tk1 - tk) * dk
        if tk1 > 0:
            sig1 = (tk1 * cond).expand(M, d) if per_dim else (tk1 * cond).expand(M, 1)
            dk1 = (x_next - model(x_next, sig1, c_noise=_cn(tk1))) / tk1
            x_next = x + (tk1 - tk) * (dk / 2 + dk1 / 2)
        x = x_next
    return x.cpu().numpy()


def save_pretrain_pickle(out_path: Path, X_, Y_, dataset, N, M, seed, nu):
    params = dict(
        N_samples_Q=N, N_samples_P=M, N_dim=X_.shape[1],
        dataset=dataset, random_seed=seed,
        generative_model="tEDM", exp_no=f"pretrain_nu{nu}",
        expname=out_path.stem, nu=nu,
        X_=X_, Y_=Y_,
        L=None, lam=None, alpha=None, formulation=None,
        f="KL", Gamma="tEDM", constraint=None, no_cvar=True,
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
    ap.add_argument("--dataset", required=True, choices=list(TTF_DATASETS.keys()))
    ap.add_argument("--nu", type=str, required=True,
                    help="Student-t dof (>2); paper sweeps {3,5,7}. 'inf' = Gaussian EDM. Comma list = per-coordinate nu (toy: 4,20)")
    ap.add_argument("--seed", type=int, default=0, help="target-sample seed")
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--samples_seen", type=int, default=SAMPLES_SEEN)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--sampler_cond", choices=["consistent", "literal"], default="consistent",
                    help="denoiser conditioning at sampling; see sample_tedm docstring")
    args = ap.parse_args()

    _nus = [float(v) for v in args.nu.split(",")]
    nu = _nus if len(_nus) > 1 else _nus[0]
    nu_tag = args.nu.replace(",", "-")
    for _v in _nus:
        assert _v > 2, "nu must be > 2 (variance normalization, Sec 2.2)"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = TTF_DATASETS[args.dataset]["dim"]
    if isinstance(nu, list):
        assert len(nu) == d, f"per-coordinate nu needs {d} entries, got {len(nu)}"
    print(f"[tEDM] dataset={args.dataset} d={d} nu={args.nu} N={args.N} M={args.M} "
          f"device={device} samples_seen={args.samples_seen}", flush=True)

    x_all = sample_target(args.dataset, args.N, args.seed).astype(np.float64)
    r = np.linalg.norm(x_all, axis=1)
    print(f"[tEDM] target: ||x|| P50={np.median(r):.2f} P99={np.quantile(r, 0.99):.2f} "
          f"max={r.max():.2e}", flush=True)

    # z-score normalize (toy protocol, App C.3), full-sample stats
    mean, std = x_all.mean(0), x_all.std(0, ddof=1)
    x_norm = torch.as_tensor((x_all - mean) / std, dtype=torch.float32, device=device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = train_tedm(x_norm, nu, device, label=args.dataset,
                       samples_seen=args.samples_seen)

    Y_norm = sample_tedm(model, args.M, d, nu, device, sampler_cond=args.sampler_cond)
    Y_ = (Y_norm * std + mean).astype(np.float32)
    rY = np.linalg.norm(Y_, axis=1)
    print(f"[tEDM] emitted {Y_.shape[0]} particles: ||y|| P50={np.median(rY):.2f} "
          f"P99={np.quantile(rY, 0.99):.2f} max={rY.max():.2e}", flush=True)

    save_pretrain_pickle(Path(args.out), X_=x_all, Y_=Y_, dataset=args.dataset,
                         N=args.N, M=args.M, seed=args.seed, nu=nu_tag)
    print(f"[tEDM] pretrain pickle written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
