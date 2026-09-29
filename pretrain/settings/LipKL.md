# Lip-KL-GPA (Lipschitz-regularized KL generative particle algorithm): settings

Reference: Chen et al., "Robust generative learning with Lipschitz-regularized alpha-divergences allows minimal assumptions on target distributions" (Information and Inference, 2025), the citation of Lip-KL-GPA in the paper, and its code, github.com/HyeminGu/Proximal_generative_models; `cvar_gpa/lib/` and `cvar_gpa/util/` extend it.
Wrapper: `pretrain/wrappers/LipKL/run_lipkl.py`. It runs the hard driver of this repository without the CVaR term
(`cvar_gpa/cvar_gpa_hard.py`) and copies its output next to the other six pre-trained models.

| Setting | Value |
|---|---|
| Divergence | KL with Lipschitz constant L = 1, Donsker-Varadhan form |
| Step size, iterations | 0.5, 4000; forward Euler |
| Particles, target samples | M = N: 5000 (synthetic), 1182 (Fama-French), 7305 (streamflow); full batch |
| Seed | 0 |
| Config | `cvar_gpa/configs/Learning_<target>-GPA_NN.yaml` |

The CVaR term is switched off with `--no_cvar`. On Neal's funnel it is switched off through `--lam 1e10` instead
(the hard driver uses coef_CVaR = 1/lam), as in the run behind the reported numbers.
