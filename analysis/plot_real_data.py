#!/usr/bin/env python3
"""Real datasets: global L1 error and tail error of every marginal |x_i| (log scale), for the pre-trained Lip-KL-GPA model (grey)
and its CVaR-GPA fine-tuned model (red). Streamflow gauges are ordered by drainage area; with more than 30 marginals each error
is split into two rows.
CLI: plot_real_data.py ff25 | streamflow          Output: analysis/figures/fig_<dataset>_marginals.{pdf,png,csv}
"""
import sys, os, csv, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paths
from metrics import global_l1_error, tail_error

DS = sys.argv[1] if len(sys.argv) > 1 else ""
if DS not in ("ff25", "streamflow"): raise SystemExit("usage: plot_real_data.py ff25 | streamflow")
XLAB = "Portfolio index $i$" if DS == "ff25" else "Gauge index $i$ (sorted by drainage area)"
p, r = pickle.load(open(paths.finetuned_file(DS, "Lip-KL"), "rb")); pp, rp = pickle.load(open(paths.pretrained_file(DS, "Lip-KL"), "rb"))
X = np.asarray(p['X_'], float); Y = np.asarray(rp['trajectories'][-1], float)[:X.shape[0]]; D = np.asarray(r['trajectories'][-1], float)
d = X.shape[1]; order = np.arange(d); ids = [str(k) for k in range(d)]
if DS == "streamflow":
    z = np.load(paths.STREAMFLOW_NPZ, allow_pickle=True)
    order = np.argsort(np.asarray(z["drain_area_km2"], float)); ids = [str(s) for s in z["site_id"]]
errors = lambda A, k: (global_l1_error(np.abs(A[:, k]), np.abs(X[:, k])), tail_error(np.abs(A[:, k]), np.abs(X[:, k])))
pre = np.array([errors(Y, k) for k in order])
ft = np.array([errors(D, k) for k in order])
OUT = str(paths.FIGURES_DIR); os.makedirs(OUT, exist_ok=True); STEM = f"fig_{DS}_marginals"
with open(f"{OUT}/{STEM}.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["marginal", "id", "pre_l1", "pre_tail", "ft_l1", "ft_tail"])
    for i, k in enumerate(order): w.writerow([i, ids[k], pre[i, 0], pre[i, 1], ft[i, 0], ft[i, 1]])
GREY,RED="#4d4d4d","#d62728"
def fmt(v): return f"{v:.2f}" if v>=0.1 else f"{v:.3f}" if v>=0.01 else f"{v:.4f}"
# > 30 marginals: split each metric into two rows of ceil(d/2) marginals so the figure fits a portrait page
nsplit=1 if d<=30 else 2; per=int(np.ceil(d/nsplit)); _W=max(10,0.42*per+3)
_S=_W/5.5   # type size: points on this canvas = target size * canvas width / 5.5 in (text width)
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":9*_S,"axes.titlesize":9*_S,"axes.labelsize":9*_S,"xtick.labelsize":(6.5 if d>30 else 8)*_S,"ytick.labelsize":8*_S})
fig,axes=plt.subplots(2*nsplit,1,figsize=(_W,3.2*2*nsplit))
axes=np.atleast_1d(axes); width=0.4
for col_i,title in ((0,r"$\mathcal{E}_{L^1,i}$"),(1,r"$\mathcal{E}_{\mathrm{tail},i}$")):
    lo=min(pre[:,col_i].min(),ft[:,col_i].min()); hi=max(pre[:,col_i].max(),ft[:,col_i].max())
    for s_ in range(nsplit):
        ax=axes[col_i*nsplit+s_]; idx=np.arange(s_*per,min(d,(s_+1)*per)); x=np.arange(len(idx))
        for j,(lab,c,A) in enumerate([("Pre-trained (Lip-KL-GPA)",GREY,pre),("CVaR-GPA",RED,ft)]):
            pos=x-0.2+width*j; vals=A[idx,col_i]
            ax.bar(pos,vals,width=width*0.95,color=c,edgecolor="black",linewidth=0.6,label=lab)
        ax.set_yscale("log"); ax.set_ylim(lo/2.5, hi*3); ax.set_xticks(x); ax.set_xticklabels([str(i) for i in idx])
        ax.set_xlim(-0.7,per-0.3); ax.set_ylabel(title); ax.set_xlabel(XLAB)
        if s_==0: ax.set_title(title)
        ax.grid(True,axis="y",ls=":",lw=0.6,alpha=0.7); ax.set_axisbelow(True)
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=2,frameon=False,fontsize=9*_S,bbox_to_anchor=(0.5,1.03))
fig.tight_layout(rect=(0,0,1,0.95))
for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/{STEM}.{ext}",dpi=200,bbox_inches="tight"); print("wrote",f"{OUT}/{STEM}.{ext}")
print(f"{DS}: per-marginal mean pre {pre[:,0].mean():.4f}/{pre[:,1].mean():.4f} ft {ft[:,0].mean():.4f}/{ft[:,1].mean():.4f}")
