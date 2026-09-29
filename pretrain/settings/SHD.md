# SHD (Score-based Heavy-tailed Diffusion): settings and deviations

Reference: `third_party/heavy_tail_diffusion/` from github.com/huajianduzhuo-code/heavy_tail_diffusion @ 5f93587.
Paper: "Learning to Simulate from Heavy-tailed Distribution via Diffusion Model". Wrapper: `pretrain/wrappers/SHD/run_shd.py`.

## Settings

| | Synthetic targets (default protocol) | Real targets (`--protocol stock`) |
|---|---|---|
| Source | their `base_pareto.yaml`, paper App. D.1.3 | their `base_stock.yaml`, paper App. D.5.2 |
| Noise (training and start) | Student-t, nu = 3.5 | Student-t, nu = 3.5 |
| Noise levels | 20, geometric from 1.0 to 0.01 | 20, geometric from 0.8 to 0.01 |
| Non-homogeneous noise scale | on (max 2.0) | off |
| Langevin sampler | 50 steps per level, step 1e-4, anneal power 2 | 100 steps per level |
| Optimizer | Adam betas (0.5, 0.9), lr 1e-3 to 1e-5 exponential, batch 1024, 352 epochs = 1760 steps | lr 1e-2 to 1e-5, 300 epochs, full batch |
| Data normalization | reflect_normalize (random sign flips; exact for symmetric marginals) | normalize |
| File tag | `SHD` | `SHDstock` |

Common to both: CSDI score network (4 layers, 8 channels, 2 heads, embedding 32), structure condition_L = 0 and target_L = 1
(unconditional, single step), 90/10 sequential train/validation split, normalization statistics over the full array,
last-epoch weights. `base_stock.yaml` says Gaussian noise while the paper says Student-t with nu = 3.5; we follow the paper.

## Deviations ledger

1. Unconditional structure (their stock config) combined with the heavy-tail diffusion block of their pareto config; their stock config alone runs Gaussian diffusion.
2. 352 epochs x 5 batches instead of their 20 x 88: the same 1760 gradient steps and the same learning-rate endpoints, for our 4500 training rows.
3. N = 5000 target samples on the synthetic targets, identical across the models of the comparison.
4. Seeds applied (torch and numpy, 1); the reference scripts define `--seed` but never apply it.
5. One batch of M samples from `model.generate` with an all-zero condition mask, de-normalized with the statistics used for normalization.
6. No absolute value on the generated samples: the paper takes |x| because its Pareto data are nonnegative; our targets are two-sided.
7. Everything else verbatim.

Paper and released code disagree on the non-homogeneous noise scale: the paper uses 2 (1 + exp(-5 |x_d|))^-1 on the raw value, the
code uses 2 (1 + exp(-|x~_d| / 5))^-1 on the batch-standardized value, and applies it in training only. We follow the code.
