#!/usr/bin/env bash
#SBATCH -J smooth_ft_streamflow
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
# Smooth CVaR-GPA on one of the seven pre-trained models of the streamflow target (d = 64, M = N = 7305; alpha = 1 - 3/7305). Ran on GPU nodes.
# Variables: INIT = pickle of the pre-trained model; MODEL = its tag (LipKL, TTFstd, TailGANpar, SHDstock, DLPM, mTAFreal, tEDMc_nu3, ...).
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
: "${INIT:?} ${MODEL:?}"
[ -f "${INIT}" ] || { echo "missing pre-trained model ${INIT}"; exit 1; }
EXP="${MODEL}_${TAG}"
activate_tf; cd "${CODE}"
python -u cvar_gpa_smooth.py --dataset CVaR_Streamflow_ohio \
    --N_samples_Q 7305 --N_samples_P 7305 --mb_size_P 7305 --beta_level 0.9995893224 --ab_weight 0 \
    --init_P_file "${INIT}" "${SMOOTH[@]}" --exp_no "${EXP}" 2>&1 | tee "${LOGS}/smooth_streamflow_${EXP}.log"
