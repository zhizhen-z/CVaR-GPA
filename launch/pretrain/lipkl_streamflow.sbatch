#!/usr/bin/env bash
#SBATCH -J pre_lipkl_streamflow
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
# Pre-trained Lip-KL-GPA model on the streamflow target (64 gauges); M = N = 7305, seed 0. Ran on a GPU node.
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
DS=Streamflow_ohio; N=7305
OUT="$(pretrained_file "${DS}" LipKL LipKL "${N}")"
[ -f "${OUT}" ] && { echo "[skip] ${OUT}"; exit 0; }
activate_tf
python -u "${WRAPPERS}/LipKL/run_lipkl.py" --dataset ${DS} --seed 0 --N ${N} --M ${N} --out "${OUT}" 2>&1 | tee "${LOGS}/pretrain_LipKL_${DS}.log"
