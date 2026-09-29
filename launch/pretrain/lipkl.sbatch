#!/usr/bin/env bash
#SBATCH -J pre_lipkl
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --constraint=amd7543
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 03:00:00
# Pre-trained Lip-KL-GPA model (pretrain/wrappers/LipKL/run_lipkl.py), seed 0.
#   DS=student_t_2D (2-d Cauchy; default) | student_t_2D_nu1.2 | student_t_2D_nu1.5 | student_t_2D_nu1.8 | Neal_funnel     N=5000
#   DS=FF25_monthly N=1182
# The streamflow target has its own launcher (lipkl_streamflow.sbatch, GPU).
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
DS="${DS:-student_t_2D}"; N="${N:-5000}"
OUT="$(pretrained_file "${DS}" LipKL LipKL "${N}")"
[ -f "${OUT}" ] && { echo "[skip] ${OUT}"; exit 0; }
activate_tf
python -u "${WRAPPERS}/LipKL/run_lipkl.py" --dataset ${DS} --seed 0 --N ${N} --M ${N} --out "${OUT}" 2>&1 | tee "${LOGS}/pretrain_LipKL_${DS}.log"
