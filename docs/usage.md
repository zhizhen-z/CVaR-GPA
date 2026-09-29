# Using CVaR-GPA

CVaR-GPA takes two sets of samples, those of a pre-trained model and those of the target, and moves the first towards the second. This page covers the use on your own samples, the two drivers and the file formats. To reproduce the paper see [reproduce.md](reproduce.md).

## Your own samples and your own target

1. Store the target samples as a `.npy` array of shape `(N, d)`, and the samples of your model in the layout of [File formats](#file-formats):

   ```python
   import pickle
   import numpy as np

   np.save("my_target.npy", target)                     # shape (N, d)
   with open("my_model.pickle", "wb") as fh:            # samples of your model, shape (M, d), in the units of the target
       pickle.dump([{}, {"trajectories": [samples.astype(np.float32)]}], fh)
   ```

2. Fine-tune. `CVaR_custom` is the dataset that reads `--target_file`.

   ```bash
   cd cvar_gpa
   python cvar_gpa_smooth.py --dataset CVaR_custom \
       --target_file /path/to/my_target.npy --init_P_file /path/to/my_model.pickle \
       --N_samples_P <M> --beta_level <alpha> \
       --f KL --formulation DV -L 0.125 --lr_P 25 --epochs 20000 --save_iter 200 \
       --lam 4e-3 --lam_rule fixed --activation_ftn leaky_relu_001 --cvar_gate ramp --ramp_h 0.5 \
       --stop_tail_ke_median 1e-10 --stop_ke_median 1e-10 --stop_median_window 1000 \
       --random_seed 0 --exp_no my_run
   ```

   The paper uses `M = N` and `alpha = 1 - 3/N`; the other values above are those of the paper.

3. Read the fine-tuned samples from the output, `assets/CVaR_custom/KL-Lipschitz_0.1250_ramp_<N>_<M>_00_my_run.pickle`:

   ```python
   param, result = pickle.load(open(output, "rb"))
   fine_tuned = result["trajectories"][-1]              # shape (M, d)
   ```

   `analysis/metrics.py` has the two error metrics of the paper, `global_l1_error` and `tail_error`, and `examples/animation.py` animates a run on a 2-d target.

Without a pre-trained model, `Learning_custom` pre-trains one with Lip-KL-GPA on the same target file:

```bash
python cvar_gpa_hard.py --dataset Learning_custom --target_file /path/to/my_target.npy \
    --N_samples_P <M> --no_cvar --f KL --formulation DV -L 1.0 --lr_P 0.5 --epochs 4000 --save_iter 50 \
    --random_seed 0 --exp_no pretrain
```

Its output, `assets/Learning_custom/KL-Lipschitz_1.0000_<N>_<M>_00_pretrain.pickle`, can be passed to `--init_P_file`.

## Arguments

| Argument | Meaning | Paper |
|---|---|---|
| `--dataset` | the target; it also selects the config `cvar_gpa/configs/<dataset>-GPA_NN.yaml` | |
| `--target_file` | target samples for `CVaR_custom` and `Learning_custom` | |
| `--init_P_file` | samples of the pre-trained model | |
| `--N_samples_Q`, `--N_samples_P` | number of target samples `N` and of particles `M` | `M = N` |
| `--beta_level` | level `alpha` of the CVaR | `1 - 3/N` |
| `--lam` | weight of the CVaR term | 4e-3 |
| `--ramp_h` | smoothing `h` of the CVaR | 0.5 |
| `-L` | Lipschitz constant | 0.125 |
| `--lr_P` | step size | 25 |
| `--epochs` | maximum number of iterations | 20000 |
| `--save_iter` | iterations between two saved snapshots | 200 |
| `--stop_tail_ke_median`, `--stop_ke_median`, `--stop_median_window` | stopping rule: thresholds and window | 1e-10, 1e-10, 1000 |
| `--random_seed` | seed | 0 |
| `--exp_no` | tag at the end of the output name | |

An argument that is not given takes its value from the config file.

## Running the drivers directly

Both drivers run from `cvar_gpa/`. They read `configs/<dataset>-GPA_NN.yaml` and write `../assets/<dataset>/<name>.pickle`, a list `[param, result]` where `result['trajectories'][-1]` holds the particles and `param['X_']` the target sample.

Smooth implementation:

```bash
cd cvar_gpa
python cvar_gpa_smooth.py --dataset CVaR_student_t_nu1.0 -nu 1.0 \
    --N_samples_Q 5000 --N_samples_P 5000 --beta_level 0.9994 \
    --init_P_file <pickle of the pre-trained model> \
    --f KL --formulation DV -L 0.125 --lr_P 25 --epochs 20000 --save_iter 200 \
    --lam 4e-3 --lam_rule fixed --activation_ftn leaky_relu_001 --cvar_gate ramp --ramp_h 0.5 \
    --stop_tail_ke_median 1e-10 --stop_ke_median 1e-10 --stop_median_window 1000 \
    --random_seed 0 --exp_no my_run
```

Hard implementation:

```bash
cd cvar_gpa
python cvar_gpa_hard.py --dataset CVaR_student_t_nu1.0 -nu 1.0 --random_seed 0 \
    --N_samples_Q 5000 --N_samples_P 5000 --beta_level 0.999 --ab_weight 0 --lam 250000 \
    --init_P_file <pickle of the pre-trained model> \
    --f KL --formulation DV -L 0.125 --lr_P 25 --epochs 20000 --save_iter 200 --exp_no my_hard_run
```

Without `--init_P_file` the particles start from the default source distribution of the dataset.

## File formats

All samples are stored as Python pickle files with the same layout, a list of two dictionaries:

```python
param, result = pickle.load(open(file, "rb"))
```

**Pre-trained samples** (`pretrain/pretrained/<target>/<model>/KL=02.00-<tag>_<N>_<M>_00_<target>.pickle`, written by the seven wrappers, read by `--init_P_file`)

| Entry | Content |
|---|---|
| `result['trajectories'][-1]` | the samples of the pre-trained model: array of shape `(M, d)`, `float32`, in the units of the data (not standardized) |
| `param['X_']` | the target sample the model was trained on: array of shape `(N, d)` |
| `param['Y_']` | the same samples as `result['trajectories'][-1]` |
| `param['dataset']`, `param['N_samples_Q']`, `param['N_samples_P']`, `param['random_seed']` | target name, `N`, `M` and seed |
| other entries | empty or `None`; they are there so that the file has the fields of a driver output |

The drivers read only `result['trajectories'][-1]`. If it has more than `--N_samples_P` rows, the first `--N_samples_P` are used. Samples of any other model can therefore be fine-tuned after storing them in this layout:

```python
import pickle
import numpy as np

samples = np.load("my_samples.npy")          # shape (M, d), in the units of the target
with open("my_model.pickle", "wb") as fh:
    pickle.dump([{}, {"trajectories": [samples.astype(np.float32)]}], fh)
```

and passing `--init_P_file my_model.pickle` to either driver. The target is chosen with `--dataset`: one of the targets of the paper, or `CVaR_custom` with `--target_file`.

**Outputs of the two drivers** (`assets/<dataset>/<name>.pickle`; the name contains `L`, the tail index for the Student-t targets, `N`, `M`, the seed and `--exp_no`)

| Entry | Content |
|---|---|
| `result['trajectories']` | list of snapshots of the particles, one every `--save_iter` iterations, each of shape `(M, d)`; the last one is the fine-tuned sample |
| `param['X_']` | the target sample, shape `(N, d)` |
| `param['Y_']` | the default initial sample of the dataset (not the pre-trained sample; that one is in the file named by `param['init_P_file']`) |
| `param` | all settings of the run: the config file with the command-line arguments applied |
| `result['KE_Ps']`, `result['divergences']`, `result['cvar_signed_diff_history']` | per iteration: kinetic energy, value of the Lipschitz-regularized KL objective, and CVaR of the particles minus CVaR of the target |
| `result['vectorfields']`, `result['kl_vf_snapshots']`, `result['cvar_vf_snapshots']` | per snapshot: the velocity of every particle, and its two components |

Pickle files can execute code when loaded; load only files that you trust.
