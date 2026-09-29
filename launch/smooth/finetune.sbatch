#!/usr/bin/env bash
#SBATCH -J smooth_ft
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --constraint=amd7543
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 03:00:00
# Smooth CVaR-GPA on one of the seven pre-trained models. Variables (sbatch --export=ALL,TARGET=...,INIT=...,MODEL=...):
#   TARGET  2d | student_t (with NU=1.2, 1.5 or 1.8) | neal | ff25
#   INIT    pickle of the pre-trained model (pretrain/pretrained/<target>/<model>/...)
#   MODEL   tag of the pre-trained model: LipKL, TTF, TailGAN, SHD, DLPM, mTAF, tEDMc_nu3, ... (real targets: TTFstd, TailGANpar, SHDstock, mTAFreal)
#   ALPHA   optional, TARGET=2d only: level alpha of the study of the level (default 0.9994 = 1 - 3/5000); the tag becomes <MODEL>_a<ALPHA>_h0.5
# The streamflow target has its own launcher (finetune_streamflow.sbatch, GPU). submit_all.sh lists every call of the paper.
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
: "${TARGET:?} ${INIT:?} ${MODEL:?}"; ALPHA="${ALPHA:-}"
[ -f "${INIT}" ] || { echo "missing pre-trained model ${INIT}"; exit 1; }
EXP="${MODEL}_${TAG}"; LEVEL=0.9994
if [ -n "${ALPHA}" ]; then [ "${TARGET}" = 2d ] || { echo "ALPHA is for TARGET=2d"; exit 1; }; EXP="${MODEL}_a${ALPHA}_h0.5"; LEVEL="${ALPHA}"; fi
activate_tf; cd "${CODE}"
case "${TARGET}" in
  2d)   python -u cvar_gpa_smooth.py --dataset CVaR_student_t_nu1.0 -nu 1.0 \
          --N_samples_Q 5000 --N_samples_P 5000 --beta_level ${LEVEL} \
          --init_P_file "${INIT}" "${SMOOTH[@]}" --exp_no "${EXP}" 2>&1 | tee "${LOGS}/smooth_2d_${EXP}.log" ;;
  student_t) : "${NU:?set NU to 1.2, 1.5 or 1.8}"
        python -u cvar_gpa_smooth.py --dataset CVaR_student_t_nu${NU} -nu ${NU} \
          --N_samples_Q 5000 --N_samples_P 5000 --beta_level 0.9994 \
          --init_P_file "${INIT}" "${SMOOTH[@]}" --exp_no "${EXP}" 2>&1 | tee "${LOGS}/smooth_student_t_nu${NU}_${EXP}.log" ;;
  neal) python -u cvar_gpa_smooth.py --dataset CVaR_Neal_funnel \
          --N_samples_Q 5000 --N_samples_P 5000 --beta_level 0.9994 --ab_weight 0 \
          --init_P_file "${INIT}" "${SMOOTH[@]}" --exp_no "${EXP}" 2>&1 | tee "${LOGS}/smooth_neal_${EXP}.log" ;;
  ff25) python -u cvar_gpa_smooth.py --dataset CVaR_FF25_monthly \
          --N_samples_Q 5000 --N_samples_P 1182 --beta_level 0.9974619289 --ab_weight 0 \
          --init_P_file "${INIT}" "${SMOOTH[@]}" --exp_no "${EXP}" 2>&1 | tee "${LOGS}/smooth_ff25_${EXP}.log" ;;
  *) echo "TARGET must be 2d, student_t, neal or ff25"; exit 1 ;;
esac
