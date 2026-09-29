#!/usr/bin/env python3
"""Study of the level alpha on the 2-d Cauchy target (nu = 1): alpha in {0.85, 0.95, 0.9994}, i.e. (1 - alpha) M in {750, 250, 3}.
(a) The 1000 largest radii g(x) = ||x|| of the target (black), the pre-trained model (red) and the fine-tuned models (blue,
    shade increasing with alpha); dashed lines: VaR_alpha of the target (labels on top) and of the pre-trained model (labels below).
(b) Smoothed CVaR gaps Delta = CVaR_target - CVaR_current (same h) before and after fine-tuning at the training level, and after
    fine-tuning evaluated at alpha = 0.9994.
CLI: plot_alpha_levels.py [nu] [h]     Defaults: nu=1.0, h=0.5.     Output: analysis/figures/fig_alpha_levels_nu<NU>_h<H>.{pdf,png}
"""
import os, pickle, sys
import paths
import numpy as np
from scipy.special import expit, log1p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

def _softplus(u): return np.maximum(u, 0.0) + log1p(np.exp(-np.abs(u)))
def _smoothed_quantile(g, h, alpha, n_bisect=80):
    tail = 1.0 - alpha; lo = float(g.min()) - 30 * h; hi = float(g.max()) + 30 * h
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if float(expit((g - mid) / h).mean()) > tail: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
def smoothed_cvar_delta(P, X, h, alpha):
    gP = np.linalg.norm(P, axis=1); gQ = np.linalg.norm(X, axis=1)
    qP = _smoothed_quantile(gP, h, alpha); qQ = _smoothed_quantile(gQ, h, alpha)
    cP = qP + (1 / (1 - alpha)) * float((h * _softplus((gP - qP) / h)).mean())
    cQ = qQ + (1 / (1 - alpha)) * float((h * _softplus((gQ - qQ) / h)).mean())
    return cQ - cP

OUT = str(paths.FIGURES_DIR); os.makedirs(OUT, exist_ok=True)
NU  = sys.argv[1] if len(sys.argv) > 1 else "1.0"
H   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
ALPHAS = [0.85, 0.95, 0.9994]
REF_A  = 0.9994                 # reference level for the cross-level column: (1 - alpha) M = 3 at M = 5000
TOPK = 1000

_S = 9.8 / 5.5   # type size: points on this canvas = target size * canvas width / 5.5 in (text width); W_FIG = 9.8
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9 * _S,
                     "axes.labelsize": 9 * _S, "xtick.labelsize": 8 * _S, "ytick.labelsize": 8 * _S,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, RED, INK, GREY = "#1f6fb5", "#c8372d", "#3c3c3c", "#8c8c8c"
BLACK = "#141414"        # target; fine-tuned rows are BLUE
GREYS = plt.get_cmap("Greys"); REDS = plt.get_cmap("Reds"); BLUES = plt.get_cmap("Blues")
TCOL = {a: GREYS(x) for a, x in zip(ALPHAS, (0.68, 0.85, 1.00))}   # target VaR_alpha guides
PCOL = {a: REDS(x)  for a, x in zip(ALPHAS, (0.62, 0.80, 0.97))}   # pretrain VaR_alpha guides
FCOL = {a: BLUES(x) for a, x in zip(ALPHAS, (0.66, 0.82, 0.97))}   # fine-tuned rows: same light-to-dark ramp as the guides

def ft_path(a):
    return str(paths.finetuned_file(f"student_t_nu{NU}", "Lip-KL", alpha=None if a == 0.9994 else str(a)))
def load(fp):
    param, res = pickle.load(open(fp, "rb"))
    return param, np.asarray(res["trajectories"][-1], float)
def radii(P): return np.sort(np.linalg.norm(P, axis=1))

pre_fp = str(paths.pretrained_file(f"student_t_nu{NU}", "Lip-KL"))
_, P_init = load(pre_fp)
rows, deltas, KPUSH = [], {}, {}
X = None
for a in ALPHAS:
    param, P = load(ft_path(a))
    if X is None: X = np.asarray(param["X_"], float)
    KPUSH[a] = int(round((1.0 - a) * P.shape[0]))
    rows.append(("fine-tuned\n" + rf"$\alpha={a}$", radii(P), FCOL[a]))
    deltas[a] = (smoothed_cvar_delta(P_init, X, H, a), smoothed_cvar_delta(P, X, H, a), smoothed_cvar_delta(P, X, H, 0.9994))
rT, rP = radii(X), radii(P_init)
rows = [("target", rT, BLACK), ("pretrain", rP, RED)] + rows
ROW_ALPHA = [None, None] + ALPHAS          # row index -> its CVaR level
n = len(rows)

def sg(v):
    """Three significant figures, always written out in full: 5330, 137, 47.1, 0.224, 0.0124.
    Plain numerals instead of a x10^k form, which keeps the gap table narrow."""
    if v == 0: return "0"
    d = 2 - int(np.floor(np.log10(abs(v))))          # decimals that give 3 significant figures
    return f"{round(v, d):+.{max(d, 0)}f}"

rng = np.random.default_rng(0)
# --- explicit layout (inches): a fixed axes box means the gap-table columns can be placed in
# points, and the whole figure keeps a known width so the type size at \linewidth is predictable.
W_FIG, H_FIG = 9.8, 5.0
L_IN, B_IN, T_IN = 1.20, 0.95, 0.50          # left labels / x axis / panel label + top VaR labels
GAP_IN, TAB_IN = 0.30, 2.55                  # gutter between the panels / panel (b)
AXW_IN, AXH_IN = W_FIG - L_IN - GAP_IN - TAB_IN, H_FIG - B_IN - T_IN
fig = plt.figure(figsize=(W_FIG, H_FIG))
ax  = fig.add_axes([L_IN / W_FIG, B_IN / H_FIG, AXW_IN / W_FIG, AXH_IN / H_FIG])
axT = fig.add_axes([(L_IN + AXW_IN + GAP_IN) / W_FIG, (B_IN + 0.17 * AXH_IN) / H_FIG,
                    TAB_IN / W_FIG, 0.58 * AXH_IN / H_FIG])   # 0.17 centres the (now shorter) table on panel (a)
axT.axis("off")
def dark(c, f=0.72):                          # darken a row colour so small type stays readable
    r, g, b = matplotlib.colors.to_rgb(c); return (r * f, g * f, b * f)
for _j, a in enumerate(ALPHAS):
    q = float(np.quantile(rT, a))
    ax.axvline(q, color=TCOL[a], ls="--", lw=1.2, alpha=0.9, zorder=1)
    ax.text(q, n - 0.35 + 0.34 * (_j % 2), rf"$\alpha={a}$", color=TCOL[a], fontsize=7 * _S, ha="center", va="bottom", clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=6)
    qp = float(np.quantile(rP, a))
    ax.axvline(qp, color=PCOL[a], ls="--", lw=1.2, alpha=0.9, zorder=1)
    ax.text(qp, -0.60 - 0.30 * (_j % 2), rf"$\alpha={a}$", color=PCOL[a], fontsize=7 * _S, ha="center", va="top", clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=6)

for i, (name, r, col) in enumerate(rows):
    r = r[-TOPK:]
    ax.scatter(r, i + rng.uniform(-0.22, 0.22, len(r)), s=9, color=col, alpha=0.35, linewidths=0, rasterized=True)

# ---- panel (b): the CVaR gaps, as a small table of its own
def cell(x, y, t, col="0.15", fs=7 * _S, ha="center", weight=None):
    axT.text(x, y, t, transform=axT.transAxes, ha=ha, va="center",
             fontsize=fs, color=col, fontweight=weight)
def rule(x0, x1, y, col="0.6", lw=0.7):
    axT.plot([x0, x1], [y] * 2, transform=axT.transAxes, color=col, lw=lw, clip_on=False)

cx = [0.08, 0.37, 0.67, 0.98]          # column centres (headers)
rx = [0.16, 0.50, 0.80, 1.12]          # right edges of the numeric columns (values right-aligned, paper style)
cell(0.0, 1.17, "(b)", "0.15", 9 * _S, ha="left", weight="bold")
rule(0.22, 0.81, 1.00); rule(0.85, 1.12, 1.00)
cell(0.505, 1.05, r"at training $\alpha$", "0.35", 6.5 * _S)
cell(0.985, 1.05, r"at $\alpha{=}0.9994$", "0.35", 6.5 * _S)
cell(cx[0], 0.89, r"train $\alpha$", "0.35", 6.5 * _S)
cell(cx[1], 0.89, r"$\Delta_{\mathrm{init}}$", dark(RED), 7 * _S)   # the pretrain's gap, so red
cell(cx[2], 0.89, r"$\Delta_{\mathrm{final}}$", "0.35", 7 * _S)
cell(cx[3], 0.89, r"$\Delta_{\mathrm{final}}$", "0.35", 7 * _S)
rule(0.0, 1.12, 0.78, "0.4", 0.8)
for _k, a in enumerate(ALPHAS):
    y = 0.64 - 0.19 * _k
    di, df, d9 = deltas[a]
    cell(rx[0], y, rf"${a}$", dark(FCOL[a]), 7 * _S, ha="right")
    cell(rx[1], y, rf"${sg(di)}$", dark(PCOL[a]), 7 * _S, ha="right")
    cell(rx[2], y, rf"${sg(df)}$", "0.55", 7 * _S, ha="right")
    cell(rx[3], y, rf"${sg(d9)}$", dark(FCOL[a]), 7 * _S, ha="right", weight="bold" if a == REF_A else None)
rule(0.0, 1.12, 0.14, "0.4", 0.8)
cell(0.0, 0.00, r"$\Delta=\mathrm{CVaR}_\alpha(\mathrm{target})-\mathrm{CVaR}_\alpha(\mathrm{current})$",
     "black", 6.5 * _S, ha="left")

ax.set_xscale("log")
_ref  = min(r[-TOPK:].min() for _, r, _ in rows[:2])          # target / pretrain set the range
_rmin = min(r[-TOPK:].min() for _, r, _ in rows)              # but never clip a plotted particle
_rmax = max(r.max() for _, r, _ in rows)
ax.set_xlim(max(0.85 * _rmin, 0.25 * _ref), 1.35 * _rmax)     # the floor keeps a collapsed cell from stretching the axis
ax.set_yticks(range(n)); ax.set_yticklabels([nm for nm, _, _ in rows], linespacing=1.5)
for lab, (_, _, col) in zip(ax.get_yticklabels(), rows):
    lab.set_color(col)
ax.set_ylim(-1.46, n - 0.38)   # room for the staggered pretrain VaR labels above the x axis
ax.set_xlabel(r"radial risk $g(x)=\|x\|$  (log axis)", labelpad=12)
ax.grid(True, axis="x", which="major", color="0.92", lw=0.7)
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=99))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis="x", which="minor", length=2.5, color="0.6")
ax.tick_params(axis="y", length=3.0, color="0.6")
for _sp in ax.spines.values():                  # subdued border for panel (a)
    _sp.set_color("0.72"); _sp.set_linewidth(0.7)
ax.text(-0.17, 1.06, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontsize=9 * _S,
        fontweight="bold", color="0.15", clip_on=False)
for ext in ("pdf", "png"):
    out = os.path.join(OUT, f"fig_alpha_levels_nu{NU}_h{H:g}.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight"); print("wrote", out)
