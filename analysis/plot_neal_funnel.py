#!/usr/bin/env python3
"""Neal's funnel, one file per panel: generated particles (teal) over the analytic log-density, x2 marginal on top (log y-axis),
x1 marginal on the right, red = the N = 5000 target sample. One panel per pre-trained model and per fine-tuned model, plus two
ground-truth panels that carry the legend (sample and histogram keys inside the panel, colorbar of the log-density on its right).
Output: analysis/figures/neal_funnel/neal_{ground_truth,ground_truth_density,<model>_{pretrain,ft}}.{pdf,png}
"""
import os, pickle
import paths
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

OUTDIR = str(paths.FIGURES_DIR / "neal_funnel"); os.makedirs(OUTDIR, exist_ok=True)
XLIM, YLIM = (-150, 150), (-13, 13)
TEAL, RED = "#5b8fa8", "#e8384f"
SIDE = 3.2   # canvas side in inches; type below is in points on this canvas
LEG_W = 0.62 # extra canvas width for the colorbar of the ground-truth figure; the panel itself keeps the geometry of the other files
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})

MODELS = [  # (file stem, title, model, t-EDM tail parameter): the seven pre-trained models
    ("lipkl",   "Lip-KL-GPA",          "Lip-KL",   None),
    ("ttf",     "TTF",                 "TTF",      None),
    ("tailgan", "Tail-GAN",            "Tail-GAN", None),
    ("tedm",    r"t-EDM ($\nu=4,20$)", "t-EDM",    "4-20"),
    ("shd",     "SHD",                 "SHD",      None),
    ("dlpm",    "DLPM",                "DLPM",     None),
    ("mtaf",    "mTAF",                "mTAF",     None),
]

def log_density(x1, x2):
    return (-0.5 * (x1 / 3.0) ** 2 - np.log(3.0 * np.sqrt(2 * np.pi)) - 0.5 * x2 ** 2 * np.exp(-x1) - 0.5 * x1 - 0.5 * np.log(2 * np.pi))
g2 = np.linspace(*XLIM, 900); g1 = np.linspace(*YLIM, 600); G2, G1 = np.meshgrid(g2, g1)
LD = np.clip(log_density(G1, G2), -20, 0)

def save(fig, stem):
    for ext in ("pdf", "png"):
        fp = os.path.join(OUTDIR, f"{stem}.{ext}"); fig.savefig(fp, dpi=300, bbox_inches="tight"); print("wrote", fp)
    plt.close(fig)

def panel(stem, Y, X, title, color=TEAL, legend=False):
    """One panel. Y = particles drawn over the density (None: density only), X = target sample (red histograms).
    legend=True puts the keys inside the panel and widens the canvas by LEG_W for a slim vertical colorbar."""
    W = SIDE + (LEG_W if legend else 0.0); f = SIDE / W   # f: panel share of the canvas width
    fig = plt.figure(figsize=(W, SIDE))
    inner = gridspec.GridSpec(2, 2, width_ratios=[4, 1.1], height_ratios=[1.1, 4], wspace=0.05, hspace=0.05, left=0.16 * f, right=0.98 * f, top=0.93, bottom=0.14)
    ax = fig.add_subplot(inner[1, 0]); axt = fig.add_subplot(inner[0, 0], sharex=ax); axr = fig.add_subplot(inner[1, 1], sharey=ax)
    ax.imshow(LD, origin="lower", extent=[*XLIM, *YLIM], aspect="auto", cmap="magma", vmin=-20, vmax=0)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_xticks([-100, 0, 100]); ax.set_yticks(range(-10, 11, 5)); ax.tick_params(labelsize=8)
    ax.set_xlabel(r"$x_2$", fontsize=10, labelpad=1); ax.set_ylabel(r"$x_1$", fontsize=10, labelpad=0)
    if Y is not None:
        ax.scatter(Y[:, 1], Y[:, 0], s=1.8, color=color, alpha=0.7, linewidths=0, rasterized=True)
    b2 = np.linspace(*XLIM, 121); b1 = np.linspace(*YLIM, 61)
    axt.hist(X[:, 1], bins=b2, density=True, histtype="step", color=RED, lw=0.9)
    axr.hist(X[:, 0], bins=b1, density=True, histtype="step", color=RED, lw=0.9, orientation="horizontal")
    if Y is not None and color == TEAL:
        axt.hist(Y[:, 1], bins=b2, density=True, histtype="step", color=TEAL, lw=0.9)
        axr.hist(Y[:, 0], bins=b1, density=True, histtype="step", color=TEAL, lw=0.9, orientation="horizontal")
    axt.set_yscale("log"); axt.set_ylim(1e-5, 3e-1); axt.set_yticks([1e-4, 1e-2]); axt.tick_params(labelbottom=False, labelsize=7)
    axr.set_xlim(0, 0.15); axr.set_xticks([0.1]); axr.tick_params(labelleft=False, labelsize=7)   # no 0 label: it would sit against the main axis's "100"
    for a in (axt, axr):
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle(title, fontsize=10, x=0.5 * f, y=0.995)
    if legend:   # keys inside the empty black corners, slim vertical colorbar on the right
        kw = dict(frameon=False, fontsize=6.2, labelcolor="w", labelspacing=0.5, handlelength=0.9, handletextpad=0.4, borderaxespad=0.5, borderpad=0)
        pts = [Line2D([], [], color=TEAL, marker="o", ms=3.5, lw=0, label="generated\nsamples")]
        if Y is not None: pts.append(Line2D([], [], color=RED, marker="o", ms=3.5, lw=0, label="target samples\n($N=5000$)"))
        ax.add_artist(ax.legend(handles=pts, loc="lower left", **kw))
        ax.legend(handles=[Line2D([], [], color=TEAL, lw=1.4, label="generated\nmarginal density"), Line2D([], [], color=RED, lw=1.4, label="target\nmarginal density")], loc="lower right", **kw)
        pos = ax.get_position()
        cax = fig.add_axes([f + 0.03 / W, pos.y0, 0.09 / W, pos.height])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-20, 0), cmap="magma"), cax=cax, orientation="vertical", extend="min")
        cb.set_ticks([-20, -15, -10, -5, 0]); cb.ax.tick_params(labelsize=7, pad=2)
        cb.set_label(r"target log-density $\log p(x_1, x_2)$", fontsize=8, labelpad=3)
    save(fig, stem)

def load_traj(fp):
    if not os.path.exists(fp): return None
    p, r = pickle.load(open(fp, "rb"))
    return np.asarray(r["trajectories"][-1], float)[:5000]

X = np.asarray(pickle.load(open(paths.finetuned_file("neal_funnel", "Lip-KL"), "rb"))[0]["X_"], float)

# ground truth: the N=5000 target sample over its analytic log-density
panel("neal_ground_truth", X, X, "Ground truth", color=RED, legend=True)
panel("neal_ground_truth_density", None, X, "Ground truth", legend=True)   # density only

for stem, lab, model, nu in MODELS:
    for stage, fp, title in (("pretrain", paths.pretrained_file("neal_funnel", model, nu), lab), ("ft", paths.finetuned_file("neal_funnel", model, nu), lab + " + CVaR-GPA")):
        Y = load_traj(fp)
        if Y is None: print("  [skip] missing", fp); continue
        panel(f"neal_{stem}_{stage}", Y, X, title)
