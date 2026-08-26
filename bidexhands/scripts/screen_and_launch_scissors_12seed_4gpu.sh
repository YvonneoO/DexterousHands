#!/usr/bin/env bash
# Screen seed stability first, then launch 3k formal Scissors collection.
set -euo pipefail

dex_root="${DEXTEROUSHANDS_ROOT:-/lp-dev/qianqian/DexterousHands}"
bidex_root="${dex_root}/bidexhands"
collector="${COLLECTOR_SCRIPT:-${bidex_root}/scripts/collect_wilor_raw_rigid.sh}"
qa_script="${QA_SCRIPT:-${bidex_root}/scripts/qa_success_npz.py}"
python_bin="${PYTHON_BIN:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
checkpoint="${SCISSORS_CHECKPOINT:-${dex_root}/runs/task_pipeline/gpu6/shadow_hand_scissors/full_2048_seed60/model_6500.pt}"
formal_launcher="${FORMAL_LAUNCHER:-${bidex_root}/scripts/launch_scissors_bulk_12seed_4gpu.sh}"
screen_root="${SCREEN_ROOT:-${dex_root}/runs/tactile_validation/shadow_hand_scissors/seed_screen_5100_5123_20260826}"
cpu_threads="${COLLECTOR_CPU_THREADS:-8}"

mkdir -p "${screen_root}"

screen_seed() {
  local gpu="$1" seed="$2"
  local output="${screen_root}/seed_${seed}"
  mkdir -p "${output}"
  if [[ ! -f "${output}/PASS" && ! -f "${output}/FAIL" ]]; then
    set +e
    DEXTEROUSHANDS_ROOT="${dex_root}" \
    PYTHON_BIN="${python_bin}" \
    OMP_NUM_THREADS="${cpu_threads}" MKL_NUM_THREADS="${cpu_threads}" \
    GPU_ID="${gpu}" GRAPHICS_DEVICE_ID="${gpu}" \
    TASK_NAME=ShadowHandScissors TACTILE_TASK_TITLE=ShadowHandScissors \
    CHECKPOINT="${checkpoint}" \
    CAMERA_EYE_OFFSET="0.32,0.0,0.80" CAMERA_TARGET_OFFSET="0.0,0.0,0.08" \
    ROLLOUT_SEED="${seed}" TARGET_SUCCESSES=2 MAX_STEPS=1500 \
    SUCCESS_ONLY_BUFFER=0 WRITE_SUCCESS_SIDE_BY_SIDE=0 \
    MIN_MAPPED_FORCE_FRACTION=0.95 OUTPUT_DIR="${output}" \
    bash "${collector}" >"${output}/collector.log" 2>&1
    rc=$?
    set -e
    successes=0
    if [[ -d "${output}/successful_episodes" ]]; then
      successes="$(find "${output}/successful_episodes" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    fi
    printf 'seed=%s\ngpu=%s\ncollector_exit=%s\nsuccesses=%s\n' \
      "${seed}" "${gpu}" "${rc}" "${successes}" >"${output}/screen_result.txt"
    if (( successes >= 2 )); then
      "${python_bin}" "${qa_script}" \
        --require-coverage --min-mapped-force-fraction 0.95 \
        "${output}" >"${output}/coverage_qa_report.json"
      printf '%s\n' "${seed}" >"${output}/PASS"
    else
      printf '%s\n' "${seed}" >"${output}/FAIL"
      # Failed screening images are not dataset payload; retain only logs/results.
      rm -rf "${output}/rgb_frames" "${output}/successful_episodes"
      rm -f "${output}/trajectory_env0.npz" "${output}/pressure_grids.npz" \
        "${output}/rgb.mp4" "${output}/tactile.mp4" \
        "${output}/rgb_tactile_side_by_side.mp4"
    fi
  fi
}

screen_stream() {
  local gpu="$1"
  shift
  local seed
  for seed in "$@"; do screen_seed "${gpu}" "${seed}"; done
}

screen_stream 0 5100 5104 5108 5112 5116 5120 >"${screen_root}/gpu0.log" 2>&1 & p0=$!
screen_stream 1 5101 5105 5109 5113 5117 5121 >"${screen_root}/gpu1.log" 2>&1 & p1=$!
screen_stream 2 5102 5106 5110 5114 5118 5122 >"${screen_root}/gpu2.log" 2>&1 & p2=$!
screen_stream 3 5103 5107 5111 5115 5119 5123 >"${screen_root}/gpu3.log" 2>&1 & p3=$!

status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
wait "${p2}" || status=1
wait "${p3}" || status=1
(( status == 0 )) || exit "${status}"

selected="${screen_root}/selected_12_seeds.txt"
find "${screen_root}" -mindepth 2 -maxdepth 2 -name PASS -type f -exec cat {} \; \
  | sort -n | head -n 12 >"${selected}"
selected_count="$(wc -l <"${selected}")"
if (( selected_count < 12 )); then
  echo "Only ${selected_count} stable seeds passed; refusing formal launch." >&2
  exit 7
fi

SCISSORS_SEEDS_FILE="${selected}" exec "${formal_launcher}"
