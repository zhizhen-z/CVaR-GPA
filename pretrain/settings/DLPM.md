# DLPM (Shariatian, Simsekli & Durmus, ICLR 2025, arXiv:2407.18609): settings and deviations

Reference: `third_party/DLPM/`, cloned from github.com/darioShar/DLPM @ faecd3a.
Wrapper: `pretrain/wrappers/DLPM/run_dlpm.py`.

Every model, method and optimizer hyperparameter is taken verbatim from the authors' own 2-d data protocol
`dlpm/configs/2d_data.yml`, and the model, method and optimizer are their classes, imported directly (MLPModel,
GenerativeLevyProcess, the AdamW settings of init_default_optimizer). The training loop mirrors
`bem/TrainingManager.py:116-125` (`loss['loss'].mean().backward()`).

## 1. Their `2d_data.yml` settings used here

| Block | Values |
|---|---|
| method | dlpm, alpha=1.8, reverse_steps=100, mean_predict=EPSILON, var_predict=FIXED, scale='scale_preserving', rescale_timesteps=True, isotropic=True, input_scaling=False |
| model | nblocks=4, nunits=64, silu, LayerNorm, skip_connection, dropout 0, time_emb learnable size 32, no_a=True, learn_variance=False |
| training | batch 1024, EPS_LOSS, lploss=2.0, monte_carlo 1/1, mean aggregation, no grad clip, no EMA |
| optimizer | AdamW lr 5e-3, betas (0.9, 0.999), no schedule |
| sampling | reverse_steps=100, stochastic (deterministic=False), clip_denoised=False |

## 2. Adaptations ledger

1. **Gradient-step budget.** Ours = 125 epochs x 5 batches = 625 AdamW steps at N=5000 (held at 625 on the other targets by `--max_steps`). Their default run is 20 epochs x ceil(32000/1024) = 20 x 32 = 640 steps (DataLoader drop_last=False), so ours is 2% short of theirs.
2. **Data z-scored per dimension** (their 2-d configs run raw O(1)-scale mixtures; our targets span percent and Cauchy scales), inverted on output.
3. **Targets** built by `run_ttf.sample_target`, identical across the models of the comparison.
4. **alpha=1.8** (their config default; no sweep, following their settings).
