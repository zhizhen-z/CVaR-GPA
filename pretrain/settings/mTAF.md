# mTAF (Laszkiewicz, Lederer & Fischer, ICML 2022): settings and deviations

Reference: `third_party/mTAF_official/` from github.com/MikeLasz/marginalTailAdaptiveFlow @ cbcfeb4.
Wrapper: `pretrain/wrappers/mTAF/run_mtaf.py`. Their NSF class, base distribution, tail-aware permutation and LU layers,
training loop and tail-estimation script are called directly.

## Settings

| | Synthetic targets (`--variant "mTAF(fix)"`) | Real targets (`--variant mTAF --protocol real`) |
|---|---|---|
| Source | their synthetic pipeline (`run_nsf_df2.sh`, `main.py`, `utils/flows.py`) | their `real_world_experiments/weather.sh` and `main.py:73-85` |
| Degrees of freedom of the base | frozen at the tail estimates | learnable, lr_df 0.01 |
| Flow (NSF) | 5 layers, 3 bins, tail bound 2, hidden 30, 1 block, ReLU, linear tails | 5 layers, 3 bins, tail bound 2.5, hidden 100, 2 blocks, batch norm, LU linear |
| Optimizer | Adam lr 1e-4, no weight decay, no scheduler | Adam lr 1e-4, cosine annealing |
| Training | 10000 steps, batch 512, best-validation checkpoint every 250 steps | 20000 steps |
| File tag | `mTAF` | `mTAFreal` |

Common to both: tails estimated per marginal with their estimator on the validation data (an estimate above 10 counts as light);
base = Gaussian on light coordinates, Student-t on heavy ones; data z-scored with full-sample statistics; train:validation = 3:2.
Their estimator returns the power-law exponent nu + 1, and mTAF uses that value as the degrees of freedom. We run it unmodified.
The environment resolves `nflows` to the fork in `third_party/nflows` (torch 2 compatible); see item 3.

## Deviations ledger

1. Budget: 5000 target samples, split 3000 train / 2000 validation at their 3:2 ratio (theirs: 15000 / 10000). No test split; `data_test` is a copy of the validation set so that their final print runs.
2. Normalization by the full-sample mean and standard deviation, as in their code, inverted on output.
3. `min_bin_width` and `min_bin_height` = 1e-3 passed explicitly: the nflows fork changed these defaults to 1e-4.
4. All-heavy case (2-d Cauchy, Fama-French): their tail-aware permutation and LU layers reject zero light coordinates, so the plain `RandomPermutation` and `LULinear` of their own code are used with the all-Student-t base.
5. Tail estimation: the same script and flags, called with absolute paths instead of from inside their repository.
6. Variant per dataset, following their own split: frozen degrees of freedom on synthetic targets, learnable on real targets.
7. `wandb` and `openturns` are stubbed; neither is on our execution path.
8. Seeds set to 0 before the split, the permutation and the initialization; their pipeline seeds data generation only.
9. Tails estimated on the raw-scale validation data. After z-scoring an infinite-variance marginal almost no value passes the estimator's absolute threshold and it fails; the estimator is scale-invariant, so the estimate is unchanged.
