#!/usr/bin/env bash
# Submits every fine-tuning run of the paper (smooth version). Run it after the seven pre-trained models of each target exist.
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
cd "${CVAR_GPA_ROOT}"; HERE="launch/smooth"
# ft <TARGET> <target folder> <N> <model folder> <pre-trained tag> <fine-tune tag> [NU] [ALPHA]
ft() { local init; init="$(pretrained_file "$2" "$4" "$5" "$3")"
       if [ "$1" = streamflow ]; then sbatch -J "smooth_$2_$6" --export=ALL,INIT="${init}",MODEL="$6" "${HERE}/finetune_streamflow.sbatch"
       else sbatch -J "smooth_$2_$6${8:+_a$8}" --export=ALL,TARGET="$1",INIT="${init}",MODEL="$6",NU="${7:-}",ALPHA="${8:-}" "${HERE}/finetune.sbatch"; fi; }

# The seven pre-trained models. Synthetic targets use the synthetic-data settings of each baseline, real targets the real-data settings.
SYN=(LipKL:LipKL TTF:TTF TailGAN:TailGAN SHD:SHD DLPM:DLPM mTAF:mTAF)
REAL=(LipKL:LipKL TTF:TTFstd TailGAN:TailGANpar SHD:SHDstock DLPM:DLPM mTAF:mTAFreal)
for M in "${SYN[@]}"; do F="${M%%:*}"; T="${M##*:}"
  ft 2d   student_t_2D 5000 "${F}" "${T}" "${T}"
  ft neal Neal_funnel  5000 "${F}" "${T}" "${T}"
  for NU in 1.2 1.5 1.8; do ft student_t "student_t_2D_nu${NU}" 5000 "${F}" "${T}" "${T}" "${NU}"; done
done
for M in "${REAL[@]}"; do F="${M%%:*}"; T="${M##*:}"
  ft ff25       FF25_monthly         1182 "${F}" "${T}" "${T}"
  ft streamflow Streamflow_ohio 7305 "${F}" "${T}" "${T}"
done
# t-EDM: tail parameter swept over {3, 5, 7}; on Neal's funnel the per-coordinate setting (4, 20) of its authors
for TNU in 3 5 7; do
  ft 2d         student_t_2D             5000 tEDM "tEDM30Mc_nu${TNU}" "tEDMc_nu${TNU}"
  ft ff25       FF25_monthly         1182 tEDM "tEDM30Mc_nu${TNU}" "tEDMc_nu${TNU}"
  ft streamflow Streamflow_ohio 7305 tEDM "tEDM30Mc_nu${TNU}" "tEDMc_nu${TNU}"
  for NU in 1.2 1.5 1.8; do ft student_t "student_t_2D_nu${NU}" 5000 tEDM "tEDM30Mc_nu${TNU}" "tEDMc_nu${TNU}" "${NU}"; done
done
ft neal Neal_funnel 5000 tEDM "tEDM30Mc_nu4-20" "tEDMc_nu4-20"
# Study of the level alpha on the 2-d Cauchy target (Lip-KL-GPA); alpha = 0.9994 is the run above
for A in 0.85 0.95; do ft 2d student_t_2D 5000 LipKL LipKL LipKL "" "${A}"; done
