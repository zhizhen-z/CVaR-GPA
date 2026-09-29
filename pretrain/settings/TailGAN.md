# Tail-GAN (Cont, Cucuringu, Xu & Zhang, Management Science 2025): settings and deviations

Reference: `third_party/Tail-GAN/` from github.com/chaozhang-ox/Tail-GAN @ bfbe950. Wrapper: `pretrain/wrappers/TailGAN/run_tailgan.py`.

## Settings (code defaults of their synthetic experiment)

| Setting | Value | Source |
|---|---|---|
| Generator | MLP 1000-128-256-512-1024-d, LeakyReLU(0.2), BatchNorm(0.8), output clamped to [-1, 1] | TailGAN.py:162-185 |
| Latent noise | z in R^1000, iid Student-t with 5 degrees of freedom | TailGAN.py:41, 56, 340-345 |
| Discriminator | sorted-PnL batch, MLP batch-256-128-2, NeuralSort temperature 0.01, projection W = 10 | TailGAN.py:189-230 |
| Loss | elicitability score for (VaR, ES), W = 10, levels alphas = [0.05] | TailGAN.py:57, 271-303 |
| Static portfolios | 50, sparse random weights (density 0.9), column-normalized | gen_static_port.py:53-66 |
| Optimizer | Adam, lr_G 1e-6, lr_D 1e-7, betas (0.5, 0.999), batch 1000, 3000 epochs, no early stopping | TailGAN.py:34-46, 324-331 |
| Ensemble | 10 networks, keep those with final generator loss at or below the median | TailGAN.py:60, 441-455 |
| File tag | `TailGAN` (synthetic targets), `TailGANpar` (real targets, `--n_epochs 3000`) | |

## Deviations ledger

1. Single-period reduction: their data are price paths, ours are vectors. Strategies = one per asset plus the static portfolios (40 for d = 2, 50 otherwise); their mean-reversion and trend-following strategies need a time axis and are dropped.
2. Affine scaling of the data into the generator's clamp range, x_scaled = (x - mean) / (K std) with max |x_scaled| = 0.9, inverted on output. The reference applies no scaling; on targets with tail index at most 2 the scale K is set by the largest draw.
3. alphas = [0.05], the code default.
4. N = 5000 target samples instead of their 50000 scenarios. The published 3000 epochs then give 15000 updates instead of their 150000; `--n_epochs` can raise it.
5. Ensemble screening reimplemented (the reference line crashes); same selection rule.
6. Ensemble member m draws its noise from its own numpy stream; all members share the initialization (global seed 1), as in the reference.
7. Everything else verbatim.
