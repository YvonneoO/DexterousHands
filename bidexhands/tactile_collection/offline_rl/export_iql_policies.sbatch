#!/bin/bash
# Stage 1 of the two-stage IQL sweep-eval: export every model_<step>.d3
# checkpoint under CKPT_DIR to a portable TorchScript .pt (export_iql_policies.py),
# via d3rlpy_offline (python 3.11). CPU-only -- no IsaacGym, no sim, just a
# trace+save pass over already-trained networks. See export_iql_policies.py's
# module docstring for WHY this stage is split from the live-rollout eval
# (offline_rl_iql_sweep_eval.sbatch, Stage 2, python 3.8 bidexhands_isaacgym_py38 --
# the two envs cannot coexist in one process on this cluster).
#
# Required env vars (--export=ALL,VAR=val,... at submission):
#   CKPT_DIR  an offline_rl_iql.sbatch OUT_DIR/iql_<arm> dir (model_<step>.d3 files)
# Optional:
#   OUT_SUBDIR  forwarded to export_iql_policies.py --out_subdir (default exported_policies)
#
# Example:
#   sbatch --export=ALL,CKPT_DIR=.../pen_p_gt_tac/iql_p_gt_tac/iql_p_gt_tac_20260907002040 \
#          export_iql_policies.sbatch
#
# Chain Stage 2 after this succeeds (afterok -- Stage 2 needs the real .pt files,
# not just a completed-but-failed Stage 1):
#   sbatch --dependency=afterok:<this_jobid> \
#          --export=ALL,CKPT_DIR=...,TASK=...,ARM=...,MANIFEST=... \
#          offline_rl_iql_sweep_eval.sbatch
#
#SBATCH --job-name=iql_export
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

YQQ="/scratch/project/prj-02-phai-lab/yqq"
source "${YQQ}/env.sh"   # modules + caches->yqq + HF/GH/WANDB tokens (see rules/vision-workflow.md)

# Same OpenBLAS/OMP/MKL thread-count fix as offline_rl_iql.sbatch -- numpy's
# bundled OpenBLAS auto-detects the NODE's full core count (not this job's
# cgroup-limited --cpus-per-task) and crashes with RLIMIT_NPROC otherwise.
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON_BIN="${PYTHON_BIN:-${YQQ}/envs/d3rlpy_offline/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: ${PYTHON_BIN} not found -- run setup_d3rlpy_env.sh once to create the " \
       "d3rlpy_offline env before submitting this job." >&2
  exit 1
fi

: "${CKPT_DIR:?set CKPT_DIR=/path/to/iql_<arm>/iql_<arm>_<timestamp> (model_<step>.d3 files)}"
OUT_SUBDIR="${OUT_SUBDIR:-exported_policies}"

# NOT $(dirname "${BASH_SOURCE[0]}") -- SLURM copies the submitted script into a
# per-job spool dir before running it, so BASH_SOURCE[0] resolves there, not to
# this script's real repo location (see offline_rl_iql.sbatch's own note on this,
# confirmed failing 2026-09-06). Hardcode the known repo path instead.
SCRIPT_DIR="${YQQ}/DexterousHands/bidexhands/tactile_collection/offline_rl"

echo "[iql_export] CKPT_DIR=${CKPT_DIR} OUT_SUBDIR=${OUT_SUBDIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_iql_policies.py" \
  --ckpt_dir "${CKPT_DIR}" --out_subdir "${OUT_SUBDIR}" --device cpu:0
