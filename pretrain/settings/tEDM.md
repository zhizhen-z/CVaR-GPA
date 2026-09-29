# t-EDM (Pandey et al., "Heavy-Tailed Diffusion Models", ICLR 2025, arXiv:2410.14171): implementation and deviations

No official code is available, so this model is implemented from the paper (`pretrain/wrappers/tEDM/run_tedm.py`);
citations are equation and algorithm numbers of arXiv:2410.14171v2.

## Method and settings

| | |
|---|---|
| Training (Alg. 1) | sigma ~ LogNormal(-1.2, 1.2); noise n = sigma eps / sqrt(kappa), kappa ~ chi2(nu)/nu, one scalar kappa per sample; EDM preconditioning evaluated at sigma sqrt(nu/(nu-2)) (Eqs. 92, 100, 101); loss = \|\|D(x0 + n) - x0\|\|^2 / c_out^2 (Eq. 12) |
| Sampler (Alg. 2) | deterministic Heun, Karras grid with rho = 7, 18 steps, sigma from 80 to 0.002; start x ~ t(0, 80^2 I, nu) |
| Other | sigma_data = 1, data z-scored and de-standardized on output, last checkpoint |
| Tail parameter | nu in {3, 5, 7}, their unconditional grid; the reported row is the nu with the smallest tail error of the pre-trained model. On Neal's funnel: their per-coordinate setting nu = (4, 20) (`--nu 4,20`) |
| File tag | `tEDM30Mc_nu<NU>` |

## Deviations ledger

1. Implemented from the paper; checked on the Student-t sampler and on nu to infinity recovering Gaussian EDM.
2. Architecture: the paper gives only its 2-d recipe (MLP, two hidden layers of width 64). Width 256 for d = 25 and d = 64 is our choice.
3. Optimizer: Adam lr 1e-3, batch 512. The runs of the paper use their budget of 30,000,000 samples seen (`--samples_seen 30000000`; the default of the wrapper is 3,000,000).
4. Tail parameter swept over their grid and reported as described above, which favors t-EDM.
5. Data z-scored with full-sample statistics; targets from `run_ttf.sample_target`.
6. Seeds fixed to 0; the paper states no seeding protocol.
7. c_noise = ln(sigma sqrt(nu/(nu-2))) / 4, following Alg. 1; their Table 5 prints ln(sigma)/4, which differs by a constant per nu.
8. Sampler conditioning (`--sampler_cond consistent`): the denoiser is conditioned on the actual standard deviation of the state, t_k sqrt(nu/(nu-2)), as its training is. A literal reading of Alg. 2 line 3 passes t_k and inflates the output tails. With per-coordinate nu, each coordinate has its own kappa, noise level and preconditioning, and the network receives the shared raw sigma.
