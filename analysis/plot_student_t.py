"""2-d isotropic Student-t, nu in {1.0, 1.2, 1.5, 1.8}: global L1 error and tail error (log scale) of the
pre-trained Lip-KL-GPA model (grey) and of the CVaR-GPA fine-tuned model (red), both measured against the
target sample of the fine-tuned run. Output: analysis/figures/fig_student_t.{pdf,png,csv}."""
import csv, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths
from metrics import radius, global_l1_error, tail_error

OUT = paths.FIGURES_DIR; OUT.mkdir(parents=True, exist_ok=True)
NUS = ["1.0", "1.2", "1.5", "1.8"]

rows = []
for nu in NUS:
    ft = paths.finetuned_file(f"student_t_nu{nu}", "Lip-KL"); pre = paths.pretrained_file(f"student_t_nu{nu}", "Lip-KL")
    p, res = pickle.load(open(ft, "rb")); _, res_pre = pickle.load(open(pre, "rb"))
    X = np.asarray(p["X_"], dtype=np.float64); D = np.asarray(res["trajectories"][-1], dtype=np.float64)
    Y = np.asarray(res_pre["trajectories"][-1], dtype=np.float64)[:D.shape[0]]
    rQ, rY, rD = radius(X), radius(Y), radius(D)
    r = {"pre_l1": global_l1_error(rY, rQ), "pre_tail": tail_error(rY, rQ), "ft_l1": global_l1_error(rD, rQ), "ft_tail": tail_error(rD, rQ), "N": X.shape[0]}
    r["nu"] = nu; r["lam"] = float(p["lam"]); r["stop_iter"] = int(p.get("stop_snapshot_iter", 0) or 0) or None
    rows.append(r)
    print(nu, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

with open(OUT / "fig_student_t.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["nu", "lam", "stop_iter", "pre_l1", "pre_tail", "ft_l1", "ft_tail", "N"])
    w.writeheader(); w.writerows(rows)

# ----------------------------------------------------------------------------- style
_S = (16.0) / 5.5   # type size: points on this canvas = target size * canvas width / 5.5 in (text width), so labels print at 9 pt and ticks at 8 pt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9 * _S,
                     "axes.titlesize": 9 * _S, "axes.labelsize": 9 * _S, "xtick.labelsize": 8 * _S, "ytick.labelsize": 8 * _S})
GREY, RED = "#4d4d4d", "#d62728"
LAB_PRE, LAB_FT = "Pre-trained (Lip-KL-GPA)", "CVaR-GPA"

def fmt(v):
    return f"{v:.2f}" if v >= 0.1 else f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"

def make(stem: str):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
    series = [(LAB_PRE, GREY, "pre"), (LAB_FT, RED, "ft")]
    n = len(series); width = 0.8 / n; x = np.arange(2)
    ymin, ymax = 1e-2, 200.0   # headroom for the rotated value labels under the panel titles
    for ax, r in zip(axes, rows):
        for j, (lab, col, key) in enumerate(series):
            vals = [r[f"{key}_l1"], r[f"{key}_tail"]]
            pos = x - 0.4 + width * (j + 0.5)
            ax.bar(pos, vals, width=width * 0.95, color=col, edgecolor="black", linewidth=0.8, label=lab)
            for xp, v in zip(pos, vals):
                ax.text(xp, v * 1.25, fmt(v), ha="center", va="bottom", rotation=90, fontsize=5 * _S, fontweight="bold")
        ax.set_yscale("log"); ax.set_ylim(ymin, ymax)
        ax.set_xticks(x); ax.set_xticklabels([r"$\mathcal{E}_{L^1}$", r"$\mathcal{E}_{\mathrm{tail}}$"])
        ax.set_title(rf"$\nu = {r['nu']}$")
        ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    axes[0].set_ylabel("error")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=n, frameon=False, fontsize=9 * _S, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{stem}.pdf")

make("fig_student_t")
