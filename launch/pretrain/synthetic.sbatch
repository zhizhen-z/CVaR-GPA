#!/usr/bin/env bash
#SBATCH -J pre_base_synth
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --array=0-6
# Pre-trained baselines on a synthetic target, each at the synthetic-data settings of its authors; N = M = 5000, seed 0.
#   DS=student_t_2D (2-d Cauchy; default)   tasks 0 TTF, 1 SHD, 2 DLPM, 3 mTAF(fix), 4-6 t-EDM nu = 3, 5, 7
#   DS=student_t_2D_nu1.2 | _nu1.5 | _nu1.8    2-d Student-t with that tail index; same tasks as the Cauchy target
#   DS=Neal_funnel                             tasks 0-3 as above, 4 t-EDM with the per-coordinate setting nu = (4, 20)
# Tail-GAN is trained as an ensemble: tailgan_member.sbatch, then tailgan_merge.sbatch.
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
export OMP_NUM_THREADS=4
DS="${DS:-student_t_2D}"; N=5000; M=5000; SEED=0; T=${SLURM_ARRAY_TASK_ID}
case "${DS}" in student_t_2D) SUF="${N}_${M}_00_student_t_2D_df1.0" ;; Neal_funnel|student_t_2D_nu1.2|student_t_2D_nu1.5|student_t_2D_nu1.8) SUF="${N}_${M}_00_${DS}" ;; *) echo "DS must be student_t_2D, student_t_2D_nu1.2, student_t_2D_nu1.5, student_t_2D_nu1.8 or Neal_funnel"; exit 1 ;; esac
OUT_DIR="${PRETRAINED}/${DS}"
run() { mkdir -p "$(dirname "$1")"; [ -f "$1" ] && { echo "[skip] $1"; exit 0; }; }
case $T in
  0) OUT="${OUT_DIR}/TTF/KL=02.00-TTF_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/TTF/run_ttf.py" --dataset ${DS} --seed ${SEED} --N ${N} --M ${M} --out "${OUT}" ;;
  1) OUT="${OUT_DIR}/SHD/KL=02.00-SHD_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/SHD/run_shd.py" --dataset ${DS} --seed ${SEED} --N ${N} --M ${M} --out "${OUT}" ;;
  2) OUT="${OUT_DIR}/DLPM/KL=02.00-DLPM_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/DLPM/run_dlpm.py" --dataset ${DS} --seed ${SEED} --N ${N} --M ${M} --out "${OUT}" ;;
  3) OUT="${OUT_DIR}/mTAF/KL=02.00-mTAF_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/mTAF/run_mtaf.py" --dataset ${DS} --variant "mTAF(fix)" --seed ${SEED} --N ${N} --M ${M} --out "${OUT}" ;;
  4|5|6)
     if [ "${DS}" = Neal_funnel ]; then [ $T = 4 ] || exit 0; NU="4,20"; else NUS=(3 5 7); NU=${NUS[$((T-4))]}; fi
     OUT="${OUT_DIR}/tEDM/KL=02.00-tEDM30Mc_nu${NU//,/-}_${SUF}.pickle"; run "${OUT}"
     BASELINE_PY "${WRAPPERS}/tEDM/run_tedm.py" --dataset ${DS} --nu "${NU}" --seed ${SEED} --N ${N} --M ${M} \
         --samples_seen 30000000 --sampler_cond consistent --out "${OUT}" ;;
esac
