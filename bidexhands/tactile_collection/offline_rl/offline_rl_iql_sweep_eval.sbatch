#!/bin/bash
# Single-GPU sbatch launcher: sweep every model_<step>.d3 checkpoint from an
# offline_rl_iql.sbatch training run through sweep_eval_iql.py's live-rollout
# success-rate eval, logging a success-rate-vs-step curve to wandb. IsaacGym
# rollout (not d3rlpy training itself) is the expensive part here, but it's
# still single-env (--num_envs=1, see sweep_eval_iql.py's build_eval_context
# call) -- one GPU is plenty, same as offline_rl_iql.sbatch.
#
# Required env vars (--export=ALL,VAR=val,... at submission):
#   CKPT_DIR  an offline_rl_iql.sbatch OUT_DIR/iql_<arm> dir (model_<step>.d3 files)
#   TASK      bidexhands --task, e.g. ShadowHandPen or ShadowHandScissors
#   ARM       p_only | p_gt_tac | p_pred_tac
#   MANIFEST  path to the SAME episode manifest JSON that built CKPT_DIR's dataset
# Optional:
#   EPISODES        episodes per checkpoint (default 30 -- quick/biased, see
#                    sweep_eval_iql.py's own module docstring)
#   CFG_ENV         bidexhands --cfg_env path (default derived from TASK below)
#   WANDB_PROJECT   forwarded to sweep_eval_iql.py --wandb_project if set
#   WANDB_NAME      forwarded to sweep_eval_iql.py --wandb_name if set
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

PYTHON_BIN="${PYTHON_BIN:-${YQQ}/envs/d3rlpy_offline/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: ${PYTHON_BIN} not found -- run setup_d3rlpy_env.sh once to create the " \
       "d3rlpy_offline env before submitting this job." >&2
  exit 1
fi

# rollout_eval_core.py (imported for its build_eval_context env-setup helper only --
# see sweep_eval_iql.py's own module docstring) unconditionally imports PIL at
# module load time even though this script never uses it (no RGB capture here).
# d3rlpy_offline was built for d3rlpy training only, so Pillow was never installed
# in it -- confirmed failing 2026-09-07 (ModuleNotFoundError: No module named 'PIL',
# jobs 513786-513789).
if ! "${PYTHON_BIN}" -c "import PIL" 2>/dev/null; then
  echo "[iql_sweep_eval] Pillow not present in d3rlpy_offline env, installing (small, no known conflicts)"
  "${PYTHON_BIN}" -m pip install -q pillow
fi

: "${CKPT_DIR:?set CKPT_DIR=/path/to/iql_<arm>/iql_<arm>_<timestamp> (model_<step>.d3 files)}"
: "${TASK:?set TASK=ShadowHandPen or ShadowHandScissors}"
: "${ARM:?set ARM=p_only, p_gt_tac, or p_pred_tac}"
: "${MANIFEST:?set MANIFEST=/path/to/<task>_manifest.json, the same one used to build the CKPT_DIR dataset}"
if [[ "${ARM}" != "p_only" && "${ARM}" != "p_gt_tac" && "${ARM}" != "p_pred_tac" ]]; then
  echo "ERROR: ARM must be p_only, p_gt_tac, or p_pred_tac, got '${ARM}'" >&2
  exit 1
fi
EPISODES="${EPISODES:-30}"

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

echo "[iql_sweep_eval] CKPT_DIR=${CKPT_DIR} TASK=${TASK} ARM=${ARM} MANIFEST=${MANIFEST} EPISODES=${EPISODES} CFG_ENV=${CFG_ENV}"

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
  --ckpt_dir "${CKPT_DIR}" --arm "${ARM}" --manifest "${MANIFEST}" \
  --episodes "${EPISODES}" \
  "${WANDB_ARGS[@]}" \
  -- --task "${TASK}" --algo ppo --cfg_env "${CFG_ENV}" --num_envs 1 --headless --seed 1234
