#!/usr/bin/env bash
# Reproduce the five-success ShadowHandPen validation rollout used by
# multi_success_wilor_view_raw_rigid.
set -euo pipefail

# The current Brev container mounts DexterousHands at
# /lp-dev/qianqian/DexterousHands. The /workspace variants are retained for
# portability. Override DEXTEROUSHANDS_ROOT when needed.
if [[ -n "${DEXTEROUSHANDS_ROOT:-}" ]]; then
  dex_root="${DEXTEROUSHANDS_ROOT}"
elif [[ -d /lp-dev/qianqian/DexterousHands/bidexhands ]]; then
  dex_root=/lp-dev/qianqian/DexterousHands
elif [[ -d /workspace/DexterousHands/bidexhands ]]; then
  dex_root=/workspace/DexterousHands
elif [[ -d /workspace/bidexhands ]]; then
  dex_root=/workspace
else
  echo "Cannot find the DexterousHands/bidexhands mount." >&2
  echo "Set DEXTEROUSHANDS_ROOT to the DexterousHands repository root." >&2
  exit 2
fi

bidex_root="${dex_root}/bidexhands"
python_bin="${PYTHON_BIN:-python}"
python_env_root="${PYTHON_ENV_ROOT:-$(cd "$(dirname "${python_bin}")/.." && pwd)}"
gpu_id="${GPU_ID:-0}"
task_name="${TASK_NAME:-ShadowHandPen}"
tactile_task_title="${TACTILE_TASK_TITLE:-${task_name}}"
checkpoint="${CHECKPOINT:-${dex_root}/runs/task_pipeline/gpu2/shadow_hand_pen/full_2048_seed32/model_6500.pt}"
output_dir="${OUTPUT_DIR:-${dex_root}/runs/tactile_validation/shadow_hand_pen/multi_success_wilor_view_raw_rigid}"
cfg_train="${CFG_TRAIN:-${bidex_root}/cfg/ppo/config.yaml}"

for required in \
  "${bidex_root}/tactile_collection/rollout_tactile_rgb_chest.py" \
  "${checkpoint}" \
  "${cfg_train}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file is missing: ${required}" >&2
    exit 3
  fi
done

if [[ -e "${output_dir}/summary.json" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
  echo "Refusing to overwrite an existing validated run: ${output_dir}" >&2
  echo "Set OUTPUT_DIR to a new directory (recommended), or ALLOW_EXISTING_OUTPUT=1 explicitly." >&2
  exit 4
fi

cd "${bidex_root}"
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHON="${python_bin}"
export PYTHONPATH="${bidex_root}:${dex_root}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${python_env_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

# Collection termination and synchronized output.
export BIDEX_TACTILE_DIR="${output_dir}"
export BIDEX_TACTILE_STEPS="${MAX_STEPS:-5000}"
export BIDEX_STOP_AFTER_SUCCESSES="${TARGET_SUCCESSES:-5}"
export BIDEX_SUCCESS_ONLY_BUFFER="${SUCCESS_ONLY_BUFFER:-1}"
export BIDEX_FRAME_STRIDE=2
export BIDEX_VIDEO_FPS=30
export BIDEX_VIDEO_WIDTH=960
export BIDEX_VIDEO_HEIGHT=720

# Frozen WiLoR/SAM3 probe view. QA on the archived five episodes measured
# camera_eye - camera_target = (0.32, 0.00, 0.80) m in every episode.
export BIDEX_CAMERA_MODE=chest
export BIDEX_CHEST_TARGET_MODE=workspace
export BIDEX_CHEST_TARGET_CENTER=bbox
export BIDEX_CHEST_TARGET_SMOOTHING=0.0
export BIDEX_CHEST_EYE_OFFSET="${CAMERA_EYE_OFFSET:-0.32,0.0,0.80}"
export BIDEX_CHEST_TARGET_OFFSET="${CAMERA_TARGET_OFFSET:-0.0,0.0,0.08}"
export BIDEX_CROP_RGB=0
export BIDEX_RGB_CROP_BOX="0.00,0.00,1.00,1.00"
export BIDEX_HAND_COLOR_SAME=1
export BIDEX_HAND_COLOR_RGB="0.42,0.52,0.56"
export BIDEX_TACTILE_TASK="${tactile_task_title}"

# Handover creates a static goal_object actor for task visualization. Hide it
# from RGB capture by default so the RGB->tactile model sees only the real
# manipulated object. This is a render-only collector toggle; observations,
# rewards, success, and policy checkpoint stay unchanged.
if [[ "${TASK_NAME:-}" == "ShadowHandOver" && -z "${BIDEX_HIDE_GOAL_OBJECT_VISUAL:-}" ]]; then
  export BIDEX_HIDE_GOAL_OBJECT_VISUAL=1
fi

# Raw rigid contacts are projected with Gaussian weights onto the EgoTouch
# 21x21 layout (217 valid taxels per hand). No temporal EMA or normalization.
export BIDEX_CONTACT_PROJECTION=rigid_contacts
export BIDEX_NORMALIZATION=none
# Fail strict coverage QA when less than 95% of positive hand-object normal
# force lands on rigid bodies represented by the 217-taxel chart.
export BIDEX_MIN_MAPPED_FORCE_FRACTION="${MIN_MAPPED_FORCE_FRACTION:-0.95}"

# One independent synchronized review video and two NPZ files per success.
export BIDEX_WRITE_COMBINED_VIDEO=0
export BIDEX_WRITE_SUCCESS_SIDE_BY_SIDE="${WRITE_SUCCESS_SIDE_BY_SIDE:-1}"
export BIDEX_KEEP_COMPONENT_VIDEOS=0

exec "${python_bin}" -m tactile_collection.rollout_tactile_rgb_chest \
  --task "${task_name}" \
  --algo ppo \
  --model_dir "${checkpoint}" \
  --cfg_train "${cfg_train}" \
  --num_envs 1 \
  --headless \
  --test \
  --seed "${ROLLOUT_SEED:-3204}" \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --graphics_device_id "${GRAPHICS_DEVICE_ID:-${gpu_id}}" \
  --pipeline cpu
