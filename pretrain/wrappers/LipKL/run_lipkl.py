"""
Train Lip-KL-GPA, the Lipschitz-regularized KL generative particle algorithm, on a target sample set and
emit a pickle in the pretrain schema of the CVaR-GPA driver. Same command line as the other six wrappers.

Lip-KL-GPA is the hard driver of this repository without the CVaR term (cvar_gpa/cvar_gpa_hard.py), so this
wrapper only maps the target to the driver's dataset and settings, runs the driver, and copies its output to --out.
pretrain/settings/LipKL.md lists the settings.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "cvar_gpa"

COMMON = ["--f", "KL", "--formulation", "DV", "-L", "1.0", "--lr_P", "0.5"]


def driver_call(dataset, N, M, seed):
    """-> (driver dataset, arguments, name of the pickle the driver writes)"""
    if dataset == "student_t_2D" or dataset.startswith("student_t_2D_nu"):
        nu = "1.0" if dataset == "student_t_2D" else dataset[len("student_t_2D_nu"):]
        return (f"Learning_student_t_nu{nu}",
                ["-nu", nu, "--random_seed", str(seed), "--N_samples_Q", str(N), "--N_samples_P", str(M), "--save_iter", "50", "--no_cvar"],
                f"KL-Lipschitz_1.0000_{float(nu):.2f}_{N}_{M}_{seed:02d}_pretrain.pickle")
    if dataset == "Neal_funnel":
        return ("Learning_Neal_funnel",
                ["--N_samples_Q", str(N), "--N_samples_P", str(M), "--save_iter", "50", "--lam", "1e10"],
                f"KL-Lipschitz_1.0000_{N}_{M}_{seed:02d}_pretrain.pickle")
    if dataset == "FF25_monthly":      # the loader shrinks N_samples_Q to the 1182 months
        return ("Learning_FF25_monthly",
                ["--random_seed", str(seed), "--N_samples_Q", "5000", "--N_samples_P", str(M), "--save_iter", "100", "--no_cvar"],
                f"KL-Lipschitz_1.0000_{N}_{M}_{seed:02d}_pretrain.pickle")
    if dataset == "Streamflow_ohio":
        return ("Learning_Streamflow_ohio",
                ["--random_seed", str(seed), "--N_samples_Q", str(N), "--N_samples_P", str(M), "--mb_size_P", str(M), "--save_iter", "100", "--no_cvar"],
                f"KL-Lipschitz_1.0000_{N}_{M}_{seed:02d}_pretrain.pickle")
    raise ValueError(f"unknown dataset: {dataset!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["student_t_2D", "student_t_2D_nu1.2", "student_t_2D_nu1.5", "student_t_2D_nu1.8", "Neal_funnel", "FF25_monthly", "Streamflow_ohio"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=5000, help="number of target samples")
    ap.add_argument("--M", type=int, default=5000, help="number of particles")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    if args.dataset == "Neal_funnel" and args.seed != 0:
        raise SystemExit("Neal_funnel: the run behind the reported numbers uses the seed of the config (0)")

    ds, extra, name = driver_call(args.dataset, args.N, args.M, args.seed)
    cmd = [sys.executable, "-u", "cvar_gpa_hard.py", "--dataset", ds] + COMMON + ["--epochs", str(args.epochs)] + extra + ["--exp_no", "pretrain"]
    print("[Lip-KL-GPA]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CODE, check=True)          # the driver reads configs/ and data/ from its working directory
    src = ROOT / "assets" / ds / name
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    print(f"[Lip-KL-GPA] pre-trained model: {out}", flush=True)


if __name__ == "__main__":
    main()
