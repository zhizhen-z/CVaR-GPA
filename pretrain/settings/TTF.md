# TTF (Tail Transform Flow, Hickling & Prangle, ICML 2025): settings and deviations

Reference: `third_party/tailnflows/`, cloned from github.com/Tennessee-Wallaceh/tailnflows @ 61072f7.
Wrapper: `pretrain/wrappers/TTF/run_ttf.py`.

All optimizer settings, model construction and data splits match TTF's own reference implementation
for the corresponding experiment class. Two protocols apply, depending on the dataset.

## 1. Protocol A: TTF synthetic density estimation (student_t_2D, student_t_2D_nu1.2/1.5/1.8, Neal_funnel)

Source: `third_party/tailnflows/experiments/density_estimation_synthetic/run_synthetic_density_estimation.py`

| Setting | Value | Source |
|---|---|---|
| Model builder (ttf_m) | base_nsf_transform: num_bins=8, tail_bound=3.0, affine_autoreg_layer=True | lines 17-19, 33-43 |
| | build_ttf_m: fix_tails=False, pos/neg_tail_init = Uniform(0.05, 1.0) | lines 40-41 |
| Optimizer (opt_params for ttf_m) | lr 5e-3, num_epochs 5000, batch_size None (full batch), early_stop_patience 100 | lines 121-125 |
| Data split | trn/val/tst = n*0.4 / n*0.2 / n*0.4 | synth_de_data_generation.ipynb, cell 0 |

## 2. Protocol B: TTF real data (FF25_monthly, Streamflow_ohio; matches their fama5 experiment)

Source: `third_party/tailnflows/experiments/density_estimation_real_data/run_density_estimation.py`

| Setting | Value | Source |
|---|---|---|
| Model builder (ttf_rqs) | base_rqs_spec: num_bins=5, tail_bound=2.5, depth=1, affine_autoreg_layer=True, u_linear_layer=True | lines 18-27, 49-64 |
| | build_ttf_m: fix_tails=False, pos/neg_tail_init = Uniform(0.05, 1.0), final_rotation="lu" | line 65 |
| Optimizer (fama5) | lr 5e-4, num_steps 5000, batch_size 100 (mini-batch), early_stop_patience 500, eval_period 25, lr_scheduler None | lines 253-260 |
| Standardization (real data only) | per-dim mean/std from the trn+val samples (all but tst), applied to all splits; our samples are de-standardized back to data units before pickling, so CVaR-GPA and the evaluation see the original scale | generate_splits.py:29-33, run_density_estimation.py:174-179 |
| Split | the real-data path uses its own generate_splits.py; we still use 40/20/40 so that all TTF pre-trained models share one split convention | |

## 3. Deviations from the TTF reference (all deliberate)

1. **Target family per dataset.** 2-d isotropic Student-t and Neal's funnel (ours) instead of heavy_tailed_nuisance (their synthetic experiment); Fama-French monthly and daily streamflow (ours) instead of fama5 (their real-data experiment). Unavoidable for an evaluation on our targets.

   1b. **Target realization.** Synthetic targets are regenerated with the generator and the seed of the CVaR-GPA driver (`sample_target` in the wrapper): 2-d Cauchy (student_t_2D) = the first N rows of a draw of size 10000; 2-d Student-t with nu = 1.2, 1.5, 1.8 and Neal's funnel = the driver's own sample. For the real datasets we call the loaders of `cvar_gpa/data` with matching arguments, so the target is identical to the one used for fine-tuning and scoring.
2. **Best-validation checkpoint restore.** TTF's reference `data_fit.train` tracks best_val_loss but never snapshots or restores model weights. Our trainer mirrors `data_fit.train` step for step and also snapshots and restores the best-validation weights, so the sampled particles come from the checkpoint at which their paper's test log-likelihood is measured. Faithful to the protocol of the paper; it fixes a gap in the reference code.
3. **Seeds.** TTF varies the seed over 0-9. We fix seed 0, as for every model of the comparison.

Everything else matches the TTF reference verbatim, per protocol.
