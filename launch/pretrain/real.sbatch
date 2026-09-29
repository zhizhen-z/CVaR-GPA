#!/usr/bin/env bash
#SBATCH -J pre_base_real
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --array=0-6
# Pre-trained baselines on a real target, each at the real-data settings of its authors; M = N, seed 0.
#   DS=Streamflow_ohio N=7305 (default)   or   DS=FF25_monthly N=1182
#   tasks 0 DLPM, 1 SHD (stock protocol), 2 TTF (real-data protocol), 3 mTAF (real-data protocol), 4-6 t-EDM nu = 3, 5, 7
# These are the commands of the streamflow runs. The launcher of the Fama-French runs was not kept; its models carry
# the same tags, and this launcher applies the same commands to that target.
# Tail-GAN is trained as an ensemble: tailgan_member.sbatch, then tailgan_merge.sbatch.
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
export OMP_NUM_THREADS=4
DS="${DS:-Streamflow_ohio}"; N="${N:-7305}"; M=${N}; SEED=0; T=${SLURM_ARRAY_TASK_ID}
OUT_DIR="${PRETRAINED}/${DS}"; SUF="${N}_${M}_00_${DS}"
run() { mkdir -p "$(dirname "$1")"; [ -f "$1" ] && { echo "[skip] $1"; exit 0; }; }
case $T in
  0) OUT="${OUT_DIR}/DLPM/KL=02.00-DLPM_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/DLPM/run_dlpm.py" --dataset ${DS} --seed ${SEED} --N ${N} --M ${M} --epochs 100000 --max_steps 625 --out "${OUT}" ;;
  1) OUT="${OUT_DIR}/SHD/KL=02.00-SHDstock_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/SHD/run_shd.py" --dataset ${DS} --protocol stock --seed ${SEED} --N ${N} --M ${M} --out "${OUT}" ;;
  2) OUT="${OUT_DIR}/TTF/KL=02.00-TTFstd_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/TTF/run_ttf.py" --dataset ${DS} --seed ${SEED} --N ${N} --M ${M} --out "${OUT}" ;;
  3) OUT="${OUT_DIR}/mTAF/KL=02.00-mTAFreal_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/mTAF/run_mtaf.py" --dataset ${DS} --variant mTAF --protocol real --seed ${SEED} --N ${N} --M ${M} --out "${OUT}" ;;
  4|5|6) NUS=(3 5 7); NU=${NUS[$((T-4))]}
     OUT="${OUT_DIR}/tEDM/KL=02.00-tEDM30Mc_nu${NU}_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/tEDM/run_tedm.py" --dataset ${DS} --nu ${NU} --seed ${SEED} --N ${N} --M ${M} \
         --samples_seen 30000000 --sampler_cond consistent --out "${OUT}" ;;
esac
