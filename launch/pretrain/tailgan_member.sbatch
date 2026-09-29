#!/usr/bin/env bash
#SBATCH -J pre_tailgan_member
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --array=0-9
# Tail-GAN ensemble: one array task trains one of the 10 members. Then run tailgan_merge.sbatch with the same variables.
#   DS=student_t_2D | student_t_2D_nu1.2 | student_t_2D_nu1.5 | student_t_2D_nu1.8 | Neal_funnel     N=5000   (default settings of the wrapper)
#   DS=Streamflow_ohio N=7305 | DS=FF25_monthly N=1182       EXTRA="--n_epochs 3000"
set -euo pipefail
source "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}/launch/common.sh"
export OMP_NUM_THREADS=4
DS="${DS:-student_t_2D}"; N="${N:-5000}"; EXTRA="${EXTRA:-}"
MEMBER_DIR="${PRETRAINED}/${DS}/TailGAN/members"; mkdir -p "${MEMBER_DIR}"
BASELINE_PY "${WRAPPERS}/TailGAN/run_tailgan.py" --dataset ${DS} --seed 0 --N ${N} --M ${N} ${EXTRA} \
    --member "${SLURM_ARRAY_TASK_ID}" --member_dir "${MEMBER_DIR}"
