#!/bin/bash
# Single-GPU sbatch launcher: build the MDP dataset (if not already built) then
# train IQL via d3rlpy on it. d3rlpy training is lightweight relative to the
# IsaacGym rollout jobs this project usually submits -- one GPU is plenty.
#
# Required env vars (--export=ALL,VAR=val,... at submission):
#   MANIFEST  path to an episode manifest JSON (generate_episode_manifest.py)
#   ARM       p_only | p_gt_tac
#   OUT_DIR   output directory (dataset .h5 + IQL checkpoint + logs go here)
# Optional:
#   N_STEPS         IQL training steps (default 100000)
#   WANDB_PROJECT   forwarded to train_iql.py --wandb_project if set
#   WANDB_NAME      forwarded to train_iql.py --wandb_name if set
#
# Example:
#   sbatch --export=ALL,MANIFEST=.../shadow_hand_pen_manifest.json,ARM=p_gt_tac,\
#          OUT_DIR=/scratch/project/prj-02-phai-lab/yqq/runs/offline_rl/pen_p_gt_tac \
#          offline_rl_iql.sbatch
#
#SBATCH --job-name=iql_offline
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

YQQ="/scratch/project/prj-02-phai-lab/yqq"
source "${YQQ}/env.sh"   # modules + caches->yqq + HF/GH/WANDB tokens (see rules/vision-workflow.md)

# numpy's bundled OpenBLAS auto-detects the NODE's full core count (not this job's
# cgroup-limited --cpus-per-task), then crashes with RLIMIT_NPROC pthread_create
# failures trying to spawn that many threads -- cap explicitly to what was requested.
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

PYTHON_BIN="${PYTHON_BIN:-${YQQ}/envs/d3rlpy_offline/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: ${PYTHON_BIN} not found -- run setup_d3rlpy_env.sh once to create the " \
       "d3rlpy_offline env before submitting this job." >&2
  exit 1
fi

: "${MANIFEST:?set MANIFEST=/path/to/<task>_manifest.json}"
: "${ARM:?set ARM=p_only or p_gt_tac}"
: "${OUT_DIR:?set OUT_DIR=/path/to/output/dir}"
if [[ "${ARM}" != "p_only" && "${ARM}" != "p_gt_tac" ]]; then
  echo "ERROR: ARM must be p_only or p_gt_tac, got '${ARM}'" >&2
  exit 1
fi
N_STEPS="${N_STEPS:-100000}"

mkdir -p "${OUT_DIR}"

# NOT $(dirname "${BASH_SOURCE[0]}") -- this cluster's SLURM copies the submitted
# script into a per-job spool dir (/cm/local/apps/slurm/var/spool/job<id>/) before
# running it, so BASH_SOURCE[0] resolves there, not to this script's real repo
# location (confirmed failing 2026-09-06: "can't open file '.../spool/job511112/
# build_mdp_dataset.py'"). Hardcode the known repo path instead.
SCRIPT_DIR="${YQQ}/DexterousHands/bidexhands/tactile_collection/offline_rl"
DATASET_PATH="${OUT_DIR}/dataset_${ARM}.h5"

echo "[offline_rl_iql] MANIFEST=${MANIFEST} ARM=${ARM} OUT_DIR=${OUT_DIR} N_STEPS=${N_STEPS}"

if [[ -f "${DATASET_PATH}" ]]; then
  echo "[offline_rl_iql] dataset already built at ${DATASET_PATH}, skipping build_mdp_dataset.py"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_mdp_dataset.py" \
    --manifest "${MANIFEST}" --arm "${ARM}" --out "${DATASET_PATH}"
fi

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS+=(--wandb_project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
  WANDB_ARGS+=(--wandb_name "${WANDB_NAME}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/train_iql.py" \
  --dataset "${DATASET_PATH}" --out "${OUT_DIR}/iql_${ARM}" \
  --n_steps "${N_STEPS}" \
  "${WANDB_ARGS[@]}"
