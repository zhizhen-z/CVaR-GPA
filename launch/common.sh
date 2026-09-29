# Sourced by every launcher. Set CVAR_GPA_ROOT to the root of this repository before submitting.
: "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}"
CODE="${CVAR_GPA_ROOT}/cvar_gpa"                       # the drivers run from here: configs/ and data/ are read from the working directory
ASSETS="${CVAR_GPA_ROOT}/assets"                       # outputs of the two drivers (all fine-tuned models)
PRETRAINED="${CVAR_GPA_ROOT}/pretrain/pretrained"      # the seven pre-trained models: <target>/<model>/KL=02.00-<tag>_<N>_<M>_00_<target>.pickle
WRAPPERS="${CVAR_GPA_ROOT}/pretrain/wrappers"
LOGS="${CVAR_GPA_ROOT}/logs"; mkdir -p "${LOGS}"          # run logs; SLURM's own slurm-<jobid>.out goes to the folder you submit from

activate_tf() {        # environment of the two CVaR-GPA drivers (environment/cvar_gpa_tf_requirements.txt)
  module load conda/latest 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "${CVAR_GPA_ENV:-cvar_gpa_tf}"; }
BASELINE_PY() {        # python of the baseline environment (environment/baselines_requirements.txt)
  "${TAILNFLOWS_ENV:?set TAILNFLOWS_ENV to the prefix of the baseline conda environment}/bin/python" -u "$@"; }

# Smooth version (Algorithms 1 and 2): the settings of every reported experiment.
#   --lam = lambda/(1-alpha) = 4e-3, h = 0.5, L = 0.125, step size 25, at most 20000 iterations, leaky-ReLU(0.01) critic, seed 0;
#   stop when the two medians over the last 1000 iterations fall below 1e-10.
SMOOTH=(--f KL --formulation DV -L 0.125 --lr_P 25 --epochs 20000 --save_iter 200
        --lam 4e-3 --lam_rule fixed --activation_ftn leaky_relu_001 --cvar_gate ramp --ramp_h 0.5
        --stop_tail_ke_median 1e-10 --stop_ke_median 1e-10 --stop_median_window 1000
        --random_seed 0)
TAG="k3_h0.5"          # experiment tag of the fine-tuned models: alpha = 1 - 3/N, h = 0.5

# File name of a pre-trained model:  pretrained_file <target> <model folder> <tag> <N>
pretrained_file() { local suf="$4_$4_00_$1"; [ "$1" = student_t_2D ] && suf="${suf}_df1.0"; echo "${PRETRAINED}/$1/$2/KL=02.00-$3_${suf}.pickle"; }
