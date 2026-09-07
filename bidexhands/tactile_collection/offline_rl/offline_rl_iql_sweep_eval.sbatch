#!/bin/bash
# Stage 2 of the two-stage IQL sweep-eval. Single-GPU sbatch launcher: sweep
# every policy_step<N>.pt EXPORTED by export_iql_policies.py (Stage 1) through
# sweep_eval_iql.py's live-rollout success-rate eval, logging a
# success-rate-vs-step curve to wandb. Runs in `bidexhands_isaacgym_py38`
# (python 3.8), NOT `d3rlpy_offline` anymore -- see sweep_eval_iql.py's module
# docstring for why (IsaacGym's py3.8-only .so bindings can't coexist with
# d3rlpy's py3.11 env in one process). IsaacGym rollout is the expensive part
# here, but it's still single-env (--num_envs=1, see sweep_eval_iql.py's
# build_eval_context call) -- one GPU is plenty, same as offline_rl_iql.sbatch.
#
# Prerequisite: export_iql_policies.sbatch (Stage 1) must have already written
# CKPT_DIR/exported_policies/policy_step<N>.pt -- chain this job after it:
#   sbatch --dependency=afterok:<stage1_jobid> \
#          --export=ALL,CKPT_DIR=...,TASK=...,ARM=...,MANIFEST=... \
#          offline_rl_iql_sweep_eval.sbatch
#
# Required env vars (--export=ALL,VAR=val,... at submission):
#   CKPT_DIR  an offline_rl_iql.sbatch OUT_DIR/iql_<arm> dir -- must contain
#             CKPT_DIR/exported_policies/policy_step<N>.pt (Stage 1's output),
#             NOT bare model_<step>.d3 files
#   TASK      bidexhands --task, e.g. ShadowHandPen or ShadowHandScissors
#   ARM       p_only | p_gt_tac | p_pred_tac
#   MANIFEST  path to the SAME episode manifest JSON that built CKPT_DIR's dataset
# Optional:
#   EPISODES         episodes per checkpoint (default 30 -- quick/biased, see
#                     sweep_eval_iql.py's own module docstring)
#   EXPORTED_SUBDIR  forwarded to sweep_eval_iql.py --exported_subdir (default
#                     exported_policies -- must match Stage 1's --out_subdir)
#   CFG_ENV          bidexhands --cfg_env path (default derived from TASK below)
#   WANDB_PROJECT    forwarded to sweep_eval_iql.py --wandb_project if set
#   WANDB_NAME       forwarded to sweep_eval_iql.py --wandb_name if set
#
# Example:
#   sbatch --export=ALL,CKPT_DIR=.../pen_p_gt_tac/iql_p_gt_tac/iql_p_gt_tac_20260907002040,\
#          TASK=ShadowHandPen,ARM=p_gt_tac,MANIFEST=.../shadow_hand_pen_manifest.json \
#          offline_rl_iql_sweep_eval.sbatch
#
#SBATCH --job-name=iql_sweep_eval
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

# Same OpenBLAS/OMP/MKL thread-count fix as offline_rl_iql.sbatch -- numpy's
# bundled OpenBLAS auto-detects the NODE's full core count (not this job's
# cgroup-limited --cpus-per-task) and crashes with RLIMIT_NPROC otherwise.
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# bidexhands_isaacgym_py38 -- the proven working IsaacGym env (confirmed via a
# real rollout smoke test, job 512871/verify_isaacgym_env.sh), NOT d3rlpy_offline
# anymore (see sweep_eval_iql.py's module docstring: the two envs can't coexist
# in one process on this cluster). This script no longer imports d3rlpy at all.
ENV_ROOT="${ENV_ROOT:-${YQQ}/envs/bidexhands_isaacgym_py38}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_ROOT}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: ${PYTHON_BIN} not found -- bidexhands_isaacgym_py38 should already exist " \
       "(it's the env verify_isaacgym_env.sh smoke-tested this session, job 512871); this " \
       "script does not create it." >&2
  exit 1
fi

# Exact env-activation pattern proven in verify_isaacgym_env.sh -- a plain `conda
# activate` does NOT set LD_LIBRARY_PATH reliably enough for isaacgym's precompiled
# .so bindings to find libpython3.8.so.1.0, and ninja (needed to JIT-compile
# isaacgym's gymtorch C++ extension) isn't put on PATH by anything else here.
export LD_LIBRARY_PATH="${ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${ENV_ROOT}/bin:${PATH}"

# NOTE (unverified on this machine -- no VISION access, see sweep_eval_iql.py's
# module docstring): bidexhands_isaacgym_py38 already needs torch for the sim/RL
# stack and already imports PIL unconditionally via rollout_eval_core.py (used
# elsewhere in this env for RGB capture), so no Pillow install workaround should
# be needed here (unlike the old d3rlpy_offline launcher, which DID need one --
# see git history). Double-check `torch.jit.load` works against this env's
# installed torch version on the FIRST real run of this stage.

: "${CKPT_DIR:?set CKPT_DIR=/path/to/iql_<arm>/iql_<arm>_<timestamp> (must contain exported_policies/policy_step<N>.pt from Stage 1)}"
: "${TASK:?set TASK=ShadowHandPen or ShadowHandScissors}"
: "${ARM:?set ARM=p_only, p_gt_tac, or p_pred_tac}"
: "${MANIFEST:?set MANIFEST=/path/to/<task>_manifest.json, the same one used to build the CKPT_DIR dataset}"
if [[ "${ARM}" != "p_only" && "${ARM}" != "p_gt_tac" && "${ARM}" != "p_pred_tac" ]]; then
  echo "ERROR: ARM must be p_only, p_gt_tac, or p_pred_tac, got '${ARM}'" >&2
  exit 1
fi
EPISODES="${EPISODES:-30}"
EXPORTED_SUBDIR="${EXPORTED_SUBDIR:-exported_policies}"

# cfg/<Task>ProprioGTTac.yaml / cfg/<Task>ProprioOnly.yaml are this project's
# established naming convention (confirmed on-disk: e.g. cfg/ShadowHandPenProprioGTTac.yaml,
# cfg/ShadowHandPenProprioOnly.yaml) -- p_only doesn't need the tactile sensors the
# GTTac configs wire up, so default to the matching one; override CFG_ENV explicitly
# if a given ARM/TASK combo needs something else (e.g. p_pred_tac's manifest was
# collected under a specific config that isn't just "GTTac").
if [[ "${ARM}" == "p_only" ]]; then
  CFG_ENV="${CFG_ENV:-cfg/${TASK}ProprioOnly.yaml}"
else
  CFG_ENV="${CFG_ENV:-cfg/${TASK}ProprioGTTac.yaml}"
fi

# NOT $(dirname "${BASH_SOURCE[0]}") -- SLURM copies the submitted script into a
# per-job spool dir before running it, so BASH_SOURCE[0] resolves there, not to
# this script's real repo location (see offline_rl_iql.sbatch's own note on this,
# confirmed failing 2026-09-06). Hardcode the known repo path instead.
REPO_ROOT="${YQQ}/DexterousHands"

echo "[iql_sweep_eval] CKPT_DIR=${CKPT_DIR} TASK=${TASK} ARM=${ARM} MANIFEST=${MANIFEST} EPISODES=${EPISODES} CFG_ENV=${CFG_ENV} EXPORTED_SUBDIR=${EXPORTED_SUBDIR}"

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS+=(--wandb_project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
  WANDB_ARGS+=(--wandb_name "${WANDB_NAME}")
fi

# sweep_eval_iql.py must run as `python -m tactile_collection.offline_rl.sweep_eval_iql`
# from the bidexhands/ root (same convention as eval_ppo_sweep.py / sweep_eval_bc_student.py),
# with bidexhands' own args after a literal `--`.
cd "${REPO_ROOT}/bidexhands"
"${PYTHON_BIN}" -m tactile_collection.offline_rl.sweep_eval_iql \
  --ckpt_dir "${CKPT_DIR}" --exported_subdir "${EXPORTED_SUBDIR}" --arm "${ARM}" --manifest "${MANIFEST}" \
  --episodes "${EPISODES}" \
  "${WANDB_ARGS[@]}" \
  -- --task "${TASK}" --algo ppo --cfg_env "${CFG_ENV}" --num_envs 1 --headless --seed 1234
