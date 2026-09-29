"""Quick start: pre-train a model on the 2-d Cauchy target, fine-tune it with CVaR-GPA, and compare the two.

    python examples/quickstart.py                      # 2000 fine-tuning iterations
    python examples/quickstart.py --iterations 20000   # the setting of the paper

Runs on a CPU. No SLURM, no GPU and none of the external models are needed.

  1. Lip-KL-GPA pre-training, 4000 iterations, with the settings of pretrain/wrappers/LipKL/run_lipkl.py.
  2. CVaR-GPA fine-tuning (cvar_gpa/cvar_gpa_smooth.py) with the settings of the paper (launch/common.sh).
  3. Both error metrics of the paper before and after fine-tuning, and an animation of the run.

Outputs are written to assets/quickstart/. A step whose output exists is skipped.
"""
import argparse
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "cvar_gpa"
OUT = ROOT / "assets" / "quickstart"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

N = 5000                                   # target samples = particles
ALPHA = 1 - 3 / N                          # level of the CVaR: 0.9994

PRETRAIN = ["cvar_gpa_hard.py", "--dataset", "Learning_student_t_nu1.0", "-nu", "1.0", "--no_cvar",
            "--f", "KL", "--formulation", "DV", "-L", "1.0", "--lr_P", "0.5", "--epochs", "4000", "--save_iter", "50",
            "--N_samples_Q", str(N), "--N_samples_P", str(N), "--random_seed", "0", "--exp_no", "pretrain"]
FINETUNE = ["cvar_gpa_smooth.py", "--dataset", "CVaR_student_t_nu1.0", "-nu", "1.0",
            "--f", "KL", "--formulation", "DV", "-L", "0.125", "--lr_P", "25",
            "--lam", "4e-3", "--lam_rule", "fixed", "--cvar_gate", "ramp", "--ramp_h", "0.5", "--beta_level", str(ALPHA),
            "--activation_ftn", "leaky_relu_001",
            "--stop_tail_ke_median", "1e-10", "--stop_ke_median", "1e-10", "--stop_median_window", "1000",
            "--N_samples_Q", str(N), "--N_samples_P", str(N), "--random_seed", "0", "--exp_no", "quickstart"]
NO_PLOTS = ["--plot_result", ""]           # the drivers' own figures are not needed here


def run_driver(arguments, written, kept):
    """Run a driver from cvar_gpa/ (it reads configs/ from its working directory) and keep its output as `kept`."""
    if kept.exists():
        print(f"[skip] {kept} exists")
        return
    start = time.time()
    subprocess.run([sys.executable, "-u"] + arguments + NO_PLOTS, cwd=CODE, check=True)
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(written, kept)
    print(f"[done] {kept}  ({(time.time() - start) / 60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=2000, help="fine-tuning iterations (the paper uses at most 20000)")
    args = parser.parse_args()

    pretrained = OUT / "pretrained.pickle"
    finetuned = OUT / f"finetuned_{args.iterations}.pickle"
    run_driver(PRETRAIN, ROOT / "assets" / "Learning_student_t_nu1.0" / f"KL-Lipschitz_1.0000_1.00_{N}_{N}_00_pretrain.pickle", pretrained)
    run_driver(FINETUNE + ["--init_P_file", str(pretrained), "--epochs", str(args.iterations),
                           "--save_iter", str(max(1, args.iterations // 100))],
               ROOT / "assets" / "CVaR_student_t_nu1.0" / f"KL-Lipschitz_0.1250_1.00_ramp_{N}_{N}_00_quickstart.pickle", finetuned)

    from metrics import global_l1_error, radius, tail_error
    with open(finetuned, "rb") as fh:
        param, result = pickle.load(fh)
    with open(pretrained, "rb") as fh:
        before = np.asarray(pickle.load(fh)[1]["trajectories"][-1], float)
    target, after = np.asarray(param["X_"], float), np.asarray(result["trajectories"][-1], float)
    print(f"\n2-d Cauchy target, {N} samples, {len(result['KE_Ps'])} fine-tuning iterations")
    print(f"{'':22s}{'pre-trained':>14s}{'fine-tuned':>14s}")
    for name, error in (("global L1 error", global_l1_error), ("tail error", tail_error)):
        print(f"{name:22s}{error(radius(before), radius(target)):14.4f}{error(radius(after), radius(target)):14.4f}")
    print(f"{'largest radius':22s}{radius(before).max():14.0f}{radius(after).max():14.0f}     (target: {radius(target).max():.0f})\n")

    from animation import make_animation
    make_animation(finetuned, pretrained, OUT / f"finetuning_{args.iterations}.gif",
                   title="Fine-tuning a pre-trained model on a 2-d Cauchy target")


if __name__ == "__main__":
    main()
