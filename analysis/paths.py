"""Where the run outputs are and how their files are named. Used by the scoring, table and figure scripts.

  pretrain/pretrained/<folder>/<model>/KL=02.00-<tag>_<N>_<M>_00_<folder>.pickle     one of the seven pre-trained models
  assets/<driver dataset>/<stem><tag>_k3_h0.5.pickle                                  its CVaR-GPA fine-tuned model

Every pickle is a list [param, result]: result['trajectories'][-1] holds the particles, param['X_'] the target sample.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
PRETRAINED_DIR = ROOT / "pretrain" / "pretrained"
METRICS_DIR = Path(__file__).resolve().parent / "metrics"    # metric CSVs (shipped with the repository)
FIGURES_DIR = Path(__file__).resolve().parent / "figures"    # figures of the paper
STREAMFLOW_NPZ = ROOT / "cvar_gpa" / "data" / "streamflow" / "ohio_smallest64_dv_2005_2024.npz"
FF25_CSV = ROOT / "cvar_gpa" / "data" / "ff25" / "FF25_monthly.csv"

# The seven pre-trained models, in the row order of every table.
MODELS = ["Lip-KL", "TTF", "Tail-GAN", "t-EDM", "SHD", "DLPM", "mTAF"]
MODEL_FOLDER = {"Lip-KL": "LipKL", "TTF": "TTF", "Tail-GAN": "TailGAN", "t-EDM": "tEDM", "SHD": "SHD", "DLPM": "DLPM", "mTAF": "mTAF"}
# File tag of a model. The real targets use the real-data settings of TTF, Tail-GAN, SHD and mTAF, hence other tags.
TAG_SYNTHETIC = {"Lip-KL": "LipKL", "TTF": "TTF", "Tail-GAN": "TailGAN", "SHD": "SHD", "DLPM": "DLPM", "mTAF": "mTAF"}
TAG_REAL = {"Lip-KL": "LipKL", "TTF": "TTFstd", "Tail-GAN": "TailGANpar", "SHD": "SHDstock", "DLPM": "DLPM", "mTAF": "mTAFreal"}

FINETUNE_TAG = "k3_h0.5"   # alpha = 1 - 3/N, h = 0.5


def _student_t(nu):
    folder = "student_t_2D" if nu == "1.0" else f"student_t_2D_nu{nu}"
    return dict(real=False, particles=5000, pretrained_folder=folder,
                pretrained_suffix=f"5000_5000_00_{folder}" + ("_df1.0" if nu == "1.0" else ""),
                finetuned_folder=f"CVaR_student_t_nu{nu}", finetuned_stem=f"KL-Lipschitz_0.1250_{float(nu):.2f}_ramp_5000_5000_00_")


TARGETS = {
    "student_t_nu1.0": _student_t("1.0"),      # 2-d Student-t with tail index 1: the 2-d Cauchy target
    "student_t_nu1.2": _student_t("1.2"),
    "student_t_nu1.5": _student_t("1.5"),
    "student_t_nu1.8": _student_t("1.8"),
    "neal_funnel": dict(real=False, particles=5000, pretrained_folder="Neal_funnel", pretrained_suffix="5000_5000_00_Neal_funnel",
                        finetuned_folder="CVaR_Neal_funnel", finetuned_stem="KL-Lipschitz_0.1250_ramp_5000_5000_00_"),
    "ff25": dict(real=True, particles=1182, pretrained_folder="FF25_monthly", pretrained_suffix="1182_1182_00_FF25_monthly",
                 finetuned_folder="CVaR_FF25_monthly", finetuned_stem="KL-Lipschitz_0.1250_ramp_1182_1182_00_"),
    "streamflow": dict(real=True, particles=7305, pretrained_folder="Streamflow_ohio", pretrained_suffix="7305_7305_00_Streamflow_ohio",
                       finetuned_folder="CVaR_Streamflow_ohio", finetuned_stem="KL-Lipschitz_0.1250_ramp_7305_7305_00_"),
}
SEVEN_MODEL_TARGETS = ["student_t_nu1.0", "student_t_nu1.2", "student_t_nu1.5", "student_t_nu1.8",
                       "neal_funnel", "ff25", "streamflow"]   # targets of the seven-model comparison
TEDM_NU_NEAL = "4-20"   # on Neal's funnel t-EDM uses the per-coordinate tail parameter (4, 20) of its authors


def _file_tags(target, model, tedm_nu):
    """-> (tag in the file name of the pre-trained model, tag in the file name of the fine-tuned model)"""
    if model == "t-EDM":
        assert tedm_nu is not None, "t-EDM needs its tail parameter: 3, 5, 7 or '4-20'"
        return f"tEDM30Mc_nu{tedm_nu}", f"tEDMc_nu{tedm_nu}"
    tag = (TAG_REAL if TARGETS[target]["real"] else TAG_SYNTHETIC)[model]
    return tag, tag


def pretrained_file(target, model, tedm_nu=None):
    """Pickle of a pre-trained model."""
    t = TARGETS[target]; tag, _ = _file_tags(target, model, tedm_nu)
    return PRETRAINED_DIR / t["pretrained_folder"] / MODEL_FOLDER[model] / f"KL=02.00-{tag}_{t['pretrained_suffix']}.pickle"


def finetuned_file(target, model, tedm_nu=None, alpha=None):
    """Pickle of a fine-tuned model. alpha=None: the level 1 - 3/N of the paper; alpha='0.85' or '0.95': the study of the level."""
    t = TARGETS[target]; _, tag = _file_tags(target, model, tedm_nu)
    run = FINETUNE_TAG if alpha is None else f"a{alpha}_h0.5"
    return ASSETS_DIR / t["finetuned_folder"] / f"{t['finetuned_stem']}{tag}_{run}.pickle"
