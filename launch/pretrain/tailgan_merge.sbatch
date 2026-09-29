#!/usr/bin/env bash
#SBATCH -J pre_tailgan_merge
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
# Tail-GAN ensemble: screen the members and write the pre-trained model. Same variables as tailgan_member.sbatch.
# File tag: TailGAN on the synthetic targets, TailGANpar on the real targets.
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
export OMP_NUM_THREADS=4
DS="${DS:-student_t_2D}"; N="${N:-5000}"; EXTRA="${EXTRA:-}"
case "${DS}" in
  student_t_2D) NAME="TailGAN_${N}_${N}_00_student_t_2D_df1.0" ;;
  Neal_funnel|student_t_2D_nu*) NAME="TailGAN_${N}_${N}_00_${DS}" ;;
  *)               NAME="TailGANpar_${N}_${N}_00_${DS}" ;;
esac
TG="${PRETRAINED}/${DS}/TailGAN"; OUT="${TG}/KL=02.00-${NAME}.pickle"
[ -f "${OUT}" ] && { echo "[skip] ${OUT}"; exit 0; }
BASELINE_PY "${WRAPPERS}/TailGAN/run_tailgan.py" --dataset ${DS} --seed 0 --N ${N} --M ${N} ${EXTRA} \
    --merge --member_dir "${TG}/members" --out "${OUT}"
