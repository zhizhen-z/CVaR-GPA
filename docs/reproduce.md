# Reproducing the paper

This page describes the code behind the experiments of the paper and how to run them. For the use of CVaR-GPA on other samples see [usage.md](usage.md).

## Implementations

The repository contains two implementations of the algorithm.

| Implementation | Driver | Penalty | Used in the paper for |
|---|---|---|---|
| smooth | `cvar_gpa/cvar_gpa_smooth.py` | smoothed CVaR with sigmoid weights | all reported experiments |
| hard | `cvar_gpa/cvar_gpa_hard.py` | CVaR with an indicator gate | the appendix algorithms "CVaR-GPA using subgradients" and "Empirical tail statistics" |

Both start from the samples of a pre-trained model. The paper uses seven pre-trained models, all stored and launched in the same way.

| Pre-trained model | Wrapper in `pretrain/wrappers/` | Trained with |
|---|---|---|
| Lip-KL-GPA | `LipKL/run_lipkl.py` | `cvar_gpa/cvar_gpa_hard.py` without the CVaR term, which is the Lipschitz-regularized KL particle algorithm |
| TTF | `TTF/run_ttf.py` | the authors' implementation |
| mTAF | `mTAF/run_mtaf.py` | the authors' implementation |
| Tail-GAN | `TailGAN/run_tailgan.py` | the authors' implementation |
| SHD | `SHD/run_shd.py` | the authors' implementation |
| DLPM | `DLPM/run_dlpm.py` | the authors' implementation |
| t-EDM | `tEDM/run_tedm.py` | our implementation, following the authors' paper |

All seven wrappers take the same arguments: `--dataset <target> --seed 0 --N <target samples> --M <particles> --out <file>`.

## Layout

```
cvar_gpa/                     the algorithm
    cvar_gpa_smooth.py        smooth implementation
    cvar_gpa_hard.py          hard implementation; with --no_cvar it also does the pre-training
    lib/                      critic network, losses, critic training, particle transport
    util/                     argument parser, data generation, kinetic energy, plotting
    data/                     data loaders; ff25/ and streamflow/ hold the data files and their download scripts
    configs/                  two yaml files per target: Learning_<target> (pre-training) and CVaR_<target> (fine-tuning)
pretrain/                     the seven pre-trained models
    wrappers/                 one folder per model: LipKL, TTF, mTAF, TailGAN, SHD, DLPM, tEDM
    settings/                 one note per model: the settings used, where they come from, and any deviations from the authors' code
    pretrained/               outputs: <target>/<model>/KL=02.00-<tag>_<N>_<M>_00_<target>.pickle (not in the repository)
launch/                       SLURM launchers
    common.sh                 paths, environments and the settings of the smooth implementation
    pretrain/                 pre-training of the seven models (lipkl*.sbatch for Lip-KL-GPA, the rest for the six wrappers)
    smooth/                   fine-tuning with the smooth implementation; submit_all.sh submits all experiments in the paper
    hard/                     fine-tuning with the hard implementation
analysis/                     scoring and figure scripts, plus the outputs of our runs:
                              metrics/ (CSV files), tables/ (LaTeX tables of the paper), figures/
environment/                  pinned package lists of the two environments
requirements.txt              the packages that CVaR-GPA itself needs
examples/                     quickstart.py and animation.py
docs/                         usage.md, this page, and the animation of the front page
setup_third_party.sh          clones the baseline repositories at the commits used for the paper
```

## Setup

```bash
export CVAR_GPA_ROOT=/path/to/this/repository      # read by every launcher
./setup_third_party.sh                             # clones the external models into pretrain/third_party/
```

| Environment | Used for | Versions |
|---|---|---|
| `cvar_gpa_tf` | the two drivers and `analysis/` | Python 3.10, TensorFlow 2.20, NumPy 2.2, SciPy 1.15, Matplotlib 3.10, PyYAML 6.0 |
| `${TAILNFLOWS_ENV}` (path to the environment) | the six external wrappers in `pretrain/wrappers/`; the Lip-KL-GPA wrapper uses `cvar_gpa_tf` | Python 3.9, PyTorch 2.0.1, NumPy 1.26, and the requirements of each baseline repository |

Full package lists are in `environment/`. The baseline environment needs two of the cloned forks installed from source: `pip install -e pretrain/third_party/nflows` and `pip install pretrain/third_party/marginalTailAdaptiveFlow`.

The launchers are SLURM scripts. Replace `YOUR_SLURM_ACCOUNT` and adjust the partition, constraint and GPU lines for your cluster. Each run writes its log to `logs/`; SLURM's own `slurm-<jobid>.out` goes to the directory you submit from. Every launcher wraps a single `python` command, which can also be run by hand.

## Settings of the reported experiments (smooth implementation)

The same values are used for every dataset and every pre-trained model. They are set once in `launch/common.sh`; the level and the sample sizes depend on `N` and are set in the fine-tuning launchers.

| Paper | Code | Value |
|---|---|---|
| level `alpha` | `--beta_level` | `1 - 3/N` (0.9994, 0.9974619289, 0.9995893224 for N = 5000, 1182, 7305) |
| smoothing `h` | `--ramp_h` | 0.5 |
| Lipschitz constant `L` | `-L` | 0.125 |
| step size `Delta t` | `--lr_P` | 25 |
| weight `lambda` | `--lam` | `--lam = lambda / (1 - alpha) = 4e-3`, so `lambda = (12/N) * 1e-3` |
| maximum number of iterations | `--epochs` | 20000 |
| `M = N` | `--N_samples_P`, `--N_samples_Q` | 5000 (synthetic), 1182 (Fama-French), 7305 (streamflow) |

Stopping rule. A run stops when two medians over the last 1000 iterations both fall below 1e-10: the kinetic energy `K_k = (1/2M) sum ||v_i||^2`, and the kinetic energy of the critic velocity on the 3 particles with the largest radius. Otherwise it stops after 20000 iterations.

Configs. Each target has two configs in `cvar_gpa/configs/`: `Learning_<target>` for Lip-KL-GPA pre-training (`L = 1`, step size 0.5) and `CVaR_<target>` for fine-tuning (`L = 0.125`, step size 25), where `<target>` is `student_t_nu<NU>`, `Neal_funnel`, `FF25_monthly` or `Streamflow_ohio` (and `custom`, for the samples of the user). The launchers pass the same values on the command line.

## Reproducing the experiments

1. Pre-train the seven models on each target with the launchers in `launch/pretrain/`. The settings of each model are listed in `pretrain/settings/`.
   - Lip-KL-GPA: `lipkl.sbatch` with `DS=student_t_2D`, `DS=student_t_2D_nu1.2` (or 1.5, 1.8), `DS=Neal_funnel` or `DS=FF25_monthly N=1182`, and `lipkl_streamflow.sbatch` for streamflow.
   - Synthetic targets: `synthetic.sbatch` with `DS=student_t_2D` (the 2-d Cauchy target), `DS=student_t_2D_nu1.2`, `DS=student_t_2D_nu1.5`, `DS=student_t_2D_nu1.8` or `DS=Neal_funnel`.
   - Real targets: `real.sbatch` with `DS=Streamflow_ohio N=7305` or `DS=FF25_monthly N=1182`.
   - Tail-GAN: `tailgan_member.sbatch`, then `tailgan_merge.sbatch`.
   - For t-EDM the tail parameter is swept over {3, 5, 7}, as its authors do. The reported value is the one whose pre-trained model has the smallest tail error averaged over the marginals (`marginal_mean_tail` in `metrics/tedm_sweep_metrics.csv`). On Neal's funnel it uses the authors' per-coordinate setting (4, 20).
2. Fine-tune. For the smooth implementation, `launch/smooth/submit_all.sh` submits all experiments reported in the paper, using the same launcher for each of the seven pre-trained models (`MODEL=LipKL`, `TTF`, and so on). For the hard implementation, use `launch/hard/finetune_lipkl_student_t.sbatch`.
3. Score the runs:
   ```bash
   cd $CVAR_GPA_ROOT/analysis
   python score_runs.py            # run outputs -> metrics/*.csv
   ```
4. Make the figures with the `plot_*.py` scripts, which write to `analysis/figures/`.

The outputs of our runs are included, so nothing needs to be re-run to see the results: `analysis/metrics/` holds the CSV files that every number in the paper comes from, `analysis/tables/` the LaTeX tables of the paper, and `analysis/figures/` the figures.

| File in `analysis/` | Content |
|---|---|
| `metrics.py` | the two error metrics of the paper, `global_l1_error` and `tail_error`, and `hill_tail_index` |
| `paths.py` | file names of the run outputs: `pretrained_file(target, model)`, `finetuned_file(target, model)` |
| `score_runs.py` | computes both errors for every run and writes the five CSV files |
| `plot_student_t.py` | 2-d Student-t: both errors as a function of the tail index (figure in the paper) |
| `plot_alpha_levels.py` | 2-d Cauchy: the effect of the level `alpha` (figure in the paper) |
| `plot_neal_funnel.py` | Neal's funnel: each of the seven models before and after fine-tuning, one file per panel (figures in the paper) |
| `plot_real_data.py ff25` / `streamflow` | Fama-French and streamflow: both errors of every marginal for Lip-KL-GPA |
| `metrics/lipkl_metrics.csv` | Lip-KL-GPA before and after fine-tuning on 2-d Student-t, Neal's funnel, Fama-French and streamflow, plus the study of the level `alpha` |
| `metrics/seven_models_metrics.csv` | all seven models before and after fine-tuning on every target: joint error and the error of every marginal |
| `metrics/tedm_sweep_metrics.csv` | pre-trained t-EDM for each tail parameter, and which one is reported |
| `metrics/streamflow_gauges.csv` | site number, drainage area and Hill tail index of the 64 gauges (computed by `hill_alpha` in `cvar_gpa/data/streamflow/usgs_nwis.py`) |
| `metrics/ff25_portfolios.csv` | name and Hill tail index of the 25 portfolios (from the absolute values of the centered monthly returns) |

## Determinism

On a given machine both drivers are deterministic. Across CPU models the float32 arithmetic differs in the last digit, and the difference grows along the trajectory, so particles computed on another machine will not be bitwise equal to ours. The reported runs used seed 0 on a single CPU model (`--constraint=amd7543`); the streamflow runs used GPU nodes. `score_runs.py` and the figure scripts are deterministic on any machine.

## Data

- The synthetic targets (2-d Student-t, Neal's funnel) are generated by `cvar_gpa/util/generate_data.py` with seed 0. The wrappers rebuild the same sample with `sample_target` in `pretrain/wrappers/TTF/run_ttf.py`.
- Fama-French 25 portfolios, monthly returns: `cvar_gpa/data/ff25/FF25_monthly.csv`, downloaded from the Kenneth R. French data library by `download_ff25.py`.
- Daily streamflow in the Ohio River basin, 2005 to 2024, from the USGS National Water Information System, in `cvar_gpa/data/streamflow/`. `pull_ohio_all_complete.py` downloads every gauge of the basin whose record covers the period and keeps the 312 with a complete daily record (all 7305 days), and `select_smallest_64.py` keeps the 64 of them with the smallest drainage area (`ohio_smallest64_dv_2005_2024.npz`, the dataset used in the paper). The loader converts discharge to mm/day.

## What is not included

- Run outputs.
- The launcher used to pre-train the six external models on the Fama-French target was not kept. `launch/pretrain/real.sbatch` applies the streamflow commands (the real-data protocols, with tags `TTFstd`, `SHDstock`, `mTAFreal` and `TailGANpar`) to that target.
- The third-party baseline repositories. `setup_third_party.sh` clones them at the commits used for the paper; each has its own license.

## Third-party code

| Baseline | Repository | Commit |
|---|---|---|
| TTF | github.com/Tennessee-Wallaceh/tailnflows (with its forks of nflows and marginalTailAdaptiveFlow) | 61072f7 (nflows f90af9f, marginalTailAdaptiveFlow cc6268e) |
| mTAF | github.com/MikeLasz/marginalTailAdaptiveFlow | cbcfeb4 |
| Tail-GAN | github.com/chaozhang-ox/Tail-GAN | bfbe950 |
| SHD | github.com/huajianduzhuo-code/heavy_tail_diffusion | 5f93587 |
| DLPM | github.com/darioShar/DLPM | faecd3a |
| t-EDM | our implementation in `pretrain/wrappers/tEDM/run_tedm.py`, following the authors' paper | |
| Lip-KL-GPA | github.com/HyeminGu/Proximal_generative_models (written by Hyemin Gu), not cloned; `cvar_gpa/lib/` and `cvar_gpa/util/` build on it | |
