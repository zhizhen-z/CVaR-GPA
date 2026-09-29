"""Animation of a fine-tuning run on a 2-d target: the particles (colored by their radius) and the complementary CDF (CCDF)
of the radius, for every saved snapshot of the run.

    python examples/animation.py <fine-tuned run .pickle> <pre-trained model .pickle> <output .gif>

Also writes the last frame next to the animation, as <output>.png.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from PIL import Image

BACKGROUND, TEXT, TEXT_SECONDARY, GRID, AXIS = "#0d1117", "#f0f3f6", "#9da7b3", "#222933", "#3a424d"
TARGET, PRETRAINED, FINETUNED = "#6e7681", "#f0883e", "#5ee6ff"
PARTICLES = LinearSegmentedColormap.from_list("particles", plt.get_cmap("plasma")(np.linspace(0.18, 1.0, 256)))


def radius(samples):
    return np.linalg.norm(samples, axis=1)


def ccdf(samples):
    """-> (sorted radii, empirical CCDF at each of them)"""
    r = np.sort(radius(samples))
    return r, 1.0 - np.arange(len(r)) / len(r)


def style(ax):
    ax.set_facecolor(BACKGROUND)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_SECONDARY, which="both", length=3)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def draw_frame(particles, target, pretrained, iteration, title, r_max):
    """One frame, as a PIL image."""
    decades = int(np.ceil(np.log10(r_max)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.3), facecolor=BACKGROUND, gridspec_kw=dict(width_ratios=[1, 1.25]))
    style(left); style(right)

    order = np.argsort(radius(particles))                       # largest radius drawn last
    r = np.clip(radius(particles)[order], 1.0, 10.0 ** decades)
    left.scatter(target[:, 0], target[:, 1], s=7, color=TARGET, alpha=0.5, linewidths=0, label="target samples")
    left.scatter(particles[order, 0], particles[order, 1], s=5 + 9 * np.log10(r), c=r, cmap=PARTICLES,
                 norm=LogNorm(1.0, 10.0 ** decades), alpha=0.8, linewidths=0)
    left.scatter([], [], s=30, color=PARTICLES(0.75), label="generated samples (color: radius)")
    left.set_xscale("symlog", linthresh=10); left.set_yscale("symlog", linthresh=10)
    lim = 2 * 10.0 ** decades
    left.set_xlim(-lim, lim); left.set_ylim(-lim, lim); left.set_aspect("equal")
    ticks = [-10.0 ** decades, -100, 0, 100, 10.0 ** decades] if decades > 2 else [-10.0 ** decades, 0, 10.0 ** decades]
    left.set_xticks(ticks); left.set_yticks(ticks)
    left.set_xlabel("$x_1$", color=TEXT_SECONDARY); left.set_ylabel("$x_2$", color=TEXT_SECONDARY)
    left.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, fontsize=8.5, markerscale=1.6,
                handletextpad=0.1, columnspacing=0.8, labelcolor=TEXT)

    for samples, color, width, label in ((target, TARGET, 3.2, "target"), (pretrained, PRETRAINED, 1.6, "pre-trained model"),
                                         (particles, FINETUNED, 2.2, "CVaR-GPA")):
        right.step(*ccdf(samples), where="post", color=color, lw=width, label=label)
    right.set_xscale("log"); right.set_yscale("log")
    right.set_xlim(1.0, lim); right.set_ylim(0.75 / len(target), 1.3)
    right.set_xlabel("radius  ||x||", color=TEXT_SECONDARY)
    right.set_ylabel("CCDF of the radius", color=TEXT_SECONDARY)
    right.legend(frameon=False, loc="lower left", fontsize=9.5, labelcolor=TEXT)

    fig.suptitle(title, x=0.02, ha="left", color=TEXT, fontsize=12.5)
    fig.text(0.98, 0.955, f"iteration {iteration:>6,d}", ha="right", va="center", color=TEXT_SECONDARY, fontsize=11,
             family="DejaVu Sans Mono")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=110, facecolor=BACKGROUND)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_animation(finetuned_pickle, pretrained_pickle, out_gif, title="Fine-tuning a pre-trained model with CVaR-GPA", max_frames=60):
    with open(finetuned_pickle, "rb") as fh:
        param, result = pickle.load(fh)
    with open(pretrained_pickle, "rb") as fh:
        pretrained = np.asarray(pickle.load(fh)[1]["trajectories"][-1], float)
    target = np.asarray(param["X_"], float)
    snapshots = [np.asarray(s, float) for s in result["trajectories"]]
    assert target.shape[1] == 2, "the animation is for 2-d targets"
    pretrained = pretrained[:snapshots[0].shape[0]]
    n_iterations, every = len(result["KE_Ps"]), int(param["save_iter"])
    iterations = [min((k + 1) * every, n_iterations) for k in range(len(snapshots))]

    # frames: the pre-trained model, then snapshots spaced geometrically (the tail moves fastest at the start)
    picked = sorted(set(np.round(np.geomspace(1, len(snapshots), min(max_frames, len(snapshots)))).astype(int) - 1))
    frames = [(0, pretrained)] + [(iterations[k], snapshots[k]) for k in picked]
    r_max = max(radius(target).max(), max(radius(s).max() for _, s in frames))

    images = [draw_frame(particles, target, pretrained, it, title, r_max) for it, particles in frames]
    palette = images[-1].quantize(colors=128, method=Image.MEDIANCUT)
    quantized = [image.quantize(palette=palette, dither=Image.NONE) for image in images]
    durations = [1500] + [150] * (len(quantized) - 2) + [3000]      # milliseconds; first and last frame are held
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(out_gif, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=True)
    images[-1].save(out_gif.with_suffix(".png"))
    print(f"wrote {out_gif} ({len(images)} frames) and {out_gif.with_suffix('.png')}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    make_animation(*sys.argv[1:4])
