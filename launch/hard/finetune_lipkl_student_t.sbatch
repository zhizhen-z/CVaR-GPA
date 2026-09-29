#!/usr/bin/env bash
#SBATCH -J hard_student_t
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --constraint=amd7543
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH --array=0-3
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
ALPHA="${ALPHA:-0.999}"; LAM="${LAM:-250000}"
NUS=(1.0 1.2 1.5 1.8); NU=${NUS[$SLURM_ARRAY_TASK_ID]}
DS=student_t_2D; [ "${NU}" = 1.0 ] || DS="student_t_2D_nu${NU}"
PRE="$(pretrained_file "${DS}" LipKL LipKL 5000)"
[ -f "${PRE}" ] || { echo "missing pre-trained model ${PRE}"; exit 1; }
activate_tf; cd "${CODE}"
python -u cvar_gpa_hard.py --dataset CVaR_student_t_nu${NU} -nu ${NU} --random_seed 0 \
    --N_samples_Q 5000 --N_samples_P 5000 --f KL --formulation DV -L 0.125 --lr_P 25 --epochs 20000 --save_iter 200 \
    --beta_level ${ALPHA} --ab_weight 0 --lam ${LAM} \
    --init_P_file "${PRE}" --exp_no "LipKL_hard_a${ALPHA}_lam${LAM}" 2>&1 | tee "${LOGS}/hard_student_t_nu${NU}.log"
