#!/usr/bin/env bash
# Collect 3,000 successful trajectories for Handover and Scissors concurrently.
# Each task uses two GPUs, 12 rollout seeds, and 250 accepted successes per seed.
set -euo pipefail

dex_root="${DEXTEROUSHANDS_ROOT:-/lp-dev/qianqian/DexterousHands}"
bidex_root="${dex_root}/bidexhands"
collector="${COLLECTOR_SCRIPT:-${bidex_root}/scripts/collect_wilor_raw_rigid.sh}"
qa_script="${QA_SCRIPT:-${bidex_root}/scripts/qa_success_npz.py}"
python_bin="${PYTHON_BIN:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
dataset_root="${DATASET_ROOT:-${dex_root}/runs/tactile_dataset}"
shard_size="${SHARD_SIZE:-250}"
allow_busy_host="${ALLOW_BUSY_HOST:-0}"
cpu_threads="${COLLECTOR_CPU_THREADS:-8}"

if [[ ! -x "${collector}" || ! -f "${qa_script}" ]]; then
  echo "Missing collector or QA script: ${collector} ${qa_script}" >&2
  exit 2
fi

zombie_count="$(ps -eo stat= | awk '$1 ~ /^Z/ {n++} END {print n+0}')"
load_one="$(awk '{print $1}' /proc/loadavg)"
cpu_count="$(nproc)"
if [[ "${allow_busy_host}" != "1" ]]; then
  if (( zombie_count > 0 )); then
    echo "Refusing launch: ${zombie_count} zombie processes are present." >&2
    echo "Inspect/reap their parents or set ALLOW_BUSY_HOST=1 after explicit approval." >&2
    exit 3
  fi
  if awk -v load="${load_one}" -v cpus="${cpu_count}" 'BEGIN {exit !(load > 2.0 * cpus)}'; then
    echo "Refusing launch: load1=${load_one} exceeds 2x ${cpu_count} CPUs." >&2
    echo "Wait for host load to recover or set ALLOW_BUSY_HOST=1 after explicit approval." >&2
    exit 4
  fi
fi

run_shard() {
  local task="$1" title="$2" checkpoint="$3" camera_eye="$4"
  local gpu="$5" seed="$6" max_steps="$7" output="$8"
  mkdir -p "${output}"
  DEXTEROUSHANDS_ROOT="${dex_root}" \
  PYTHON_BIN="${python_bin}" \
  OMP_NUM_THREADS="${cpu_threads}" MKL_NUM_THREADS="${cpu_threads}" \
  GPU_ID="${gpu}" GRAPHICS_DEVICE_ID="${gpu}" \
  TASK_NAME="${task}" TACTILE_TASK_TITLE="${title}" \
  CHECKPOINT="${checkpoint}" \
  CAMERA_EYE_OFFSET="${camera_eye}" CAMERA_TARGET_OFFSET="0.0,0.0,0.08" \
  ROLLOUT_SEED="${seed}" TARGET_SUCCESSES="${shard_size}" MAX_STEPS="${max_steps}" \
  SUCCESS_ONLY_BUFFER=0 WRITE_SUCCESS_SIDE_BY_SIDE=0 \
  MIN_MAPPED_FORCE_FRACTION=0.95 \
  OUTPUT_DIR="${output}" \
  bash "${collector}" >"${output}/collector.log" 2>&1

  "${python_bin}" "${qa_script}" \
    --require-coverage --min-mapped-force-fraction 0.95 \
    "${output}" >"${output}/coverage_qa_report.json"
}

run_seed_stream() {
  local task="$1" title="$2" checkpoint="$3" camera_eye="$4"
  local task_slug="$5" gpu="$6" max_steps="$7"
  shift 7
  local seed output
  for seed in "$@"; do
    output="${dataset_root}/${task_slug}/seed_${seed}_n${shard_size}"
    run_shard \
      "${task}" "${title}" "${checkpoint}" "${camera_eye}" \
      "${gpu}" "${seed}" "${max_steps}" "${output}"
  done
}

handover_checkpoint="${dex_root}/runs/formal_2048/over_seed0/model_6500.pt"
scissors_checkpoint="${dex_root}/runs/task_pipeline/gpu6/shadow_hand_scissors/full_2048_seed60/model_6500.pt"

mkdir -p \
  "${dataset_root}/shadow_hand_over" \
  "${dataset_root}/shadow_hand_scissors"

run_seed_stream \
  ShadowHandOver ShadowHandOver "${handover_checkpoint}" "0.16,0.0,0.60" \
  shadow_hand_over 0 30000 \
  6300 6302 6304 6306 6308 6310 \
  >"${dataset_root}/shadow_hand_over/gpu0_supervisor.log" 2>&1 &
pid_h0=$!

run_seed_stream \
  ShadowHandOver ShadowHandOver "${handover_checkpoint}" "0.16,0.0,0.60" \
  shadow_hand_over 1 30000 \
  6301 6303 6305 6307 6309 6311 \
  >"${dataset_root}/shadow_hand_over/gpu1_supervisor.log" 2>&1 &
pid_h1=$!

run_seed_stream \
  ShadowHandScissors ShadowHandScissors "${scissors_checkpoint}" "0.32,0.0,0.80" \
  shadow_hand_scissors 2 60000 \
  6400 6402 6404 6406 6408 6410 \
  >"${dataset_root}/shadow_hand_scissors/gpu2_supervisor.log" 2>&1 &
pid_s2=$!

run_seed_stream \
  ShadowHandScissors ShadowHandScissors "${scissors_checkpoint}" "0.32,0.0,0.80" \
  shadow_hand_scissors 3 60000 \
  6401 6403 6405 6407 6409 6411 \
  >"${dataset_root}/shadow_hand_scissors/gpu3_supervisor.log" 2>&1 &
pid_s3=$!

{
  printf 'handover_gpu0_pid=%s\n' "${pid_h0}"
  printf 'handover_gpu1_pid=%s\n' "${pid_h1}"
  printf 'scissors_gpu2_pid=%s\n' "${pid_s2}"
  printf 'scissors_gpu3_pid=%s\n' "${pid_s3}"
} >"${dataset_root}/handover_scissors_12seed_4gpu.pids"

status=0
wait "${pid_h0}" || status=1
wait "${pid_h1}" || status=1
wait "${pid_s2}" || status=1
wait "${pid_s3}" || status=1
exit "${status}"
