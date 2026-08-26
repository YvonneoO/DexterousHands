#!/usr/bin/env bash
# Collect 3,000 successful ShadowHandScissors trajectories on GPUs 0-3.
# Four GPU streams each run three seeds serially; each seed keeps 250 successes.
set -euo pipefail

dex_root="${DEXTEROUSHANDS_ROOT:-/lp-dev/qianqian/DexterousHands}"
bidex_root="${dex_root}/bidexhands"
collector="${COLLECTOR_SCRIPT:-${bidex_root}/scripts/collect_wilor_raw_rigid.sh}"
qa_script="${QA_SCRIPT:-${bidex_root}/scripts/qa_success_npz.py}"
python_bin="${PYTHON_BIN:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
dataset_root="${DATASET_ROOT:-${dex_root}/runs/tactile_dataset}"
checkpoint="${SCISSORS_CHECKPOINT:-${dex_root}/runs/task_pipeline/gpu6/shadow_hand_scissors/full_2048_seed60/model_6500.pt}"
shard_size="${SHARD_SIZE:-250}"
cpu_threads="${COLLECTOR_CPU_THREADS:-8}"
task_root="${dataset_root}/shadow_hand_scissors"
seeds_file="${SCISSORS_SEEDS_FILE:-}"

if [[ ! -x "${collector}" || ! -f "${qa_script}" || ! -f "${checkpoint}" ]]; then
  echo "Missing collector, QA script, or checkpoint." >&2
  exit 2
fi

run_shard() {
  local gpu="$1" seed="$2"
  local output="${task_root}/seed_${seed}_n${shard_size}"
  local existing=0
  mkdir -p "${output}"
  if [[ -d "${output}/successful_episodes" ]]; then
    existing="$(find "${output}/successful_episodes" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  fi
  if (( existing >= shard_size )); then
    echo "seed=${seed} already has ${existing} episodes; re-running strict QA"
  elif (( existing > 0 )); then
    echo "Refusing to overwrite partial seed=${seed} shard with ${existing} episodes: ${output}" >&2
    exit 5
  else
    DEXTEROUSHANDS_ROOT="${dex_root}" \
    PYTHON_BIN="${python_bin}" \
    OMP_NUM_THREADS="${cpu_threads}" MKL_NUM_THREADS="${cpu_threads}" \
    GPU_ID="${gpu}" GRAPHICS_DEVICE_ID="${gpu}" \
    TASK_NAME=ShadowHandScissors TACTILE_TASK_TITLE=ShadowHandScissors \
    CHECKPOINT="${checkpoint}" \
    CAMERA_EYE_OFFSET="0.32,0.0,0.80" CAMERA_TARGET_OFFSET="0.0,0.0,0.08" \
    ROLLOUT_SEED="${seed}" TARGET_SUCCESSES="${shard_size}" MAX_STEPS=60000 \
    SUCCESS_ONLY_BUFFER=0 WRITE_SUCCESS_SIDE_BY_SIDE=0 \
    MIN_MAPPED_FORCE_FRACTION=0.95 \
    OUTPUT_DIR="${output}" \
    bash "${collector}" >"${output}/collector.log" 2>&1
  fi

  "${python_bin}" "${qa_script}" \
    --require-coverage --min-mapped-force-fraction 0.95 \
    "${output}" >"${output}/coverage_qa_report.json"
}

run_stream() {
  local gpu="$1"
  shift
  local seed
  for seed in "$@"; do
    run_shard "${gpu}" "${seed}"
  done
}

mkdir -p "${task_root}"
if [[ -n "${seeds_file}" ]]; then
  mapfile -t all_seeds < <(awk 'NF {print $1}' "${seeds_file}")
else
  all_seeds=(6400 6401 6402 6403 6404 6405 6406 6407 6408 6409 6410 6411)
fi
if (( ${#all_seeds[@]} != 12 )); then
  echo "Expected exactly 12 Scissors seeds, got ${#all_seeds[@]}" >&2
  exit 6
fi

gpu0_seeds=() gpu1_seeds=() gpu2_seeds=() gpu3_seeds=()
for i in "${!all_seeds[@]}"; do
  case $((i % 4)) in
    0) gpu0_seeds+=("${all_seeds[$i]}") ;;
    1) gpu1_seeds+=("${all_seeds[$i]}") ;;
    2) gpu2_seeds+=("${all_seeds[$i]}") ;;
    3) gpu3_seeds+=("${all_seeds[$i]}") ;;
  esac
done

printf '%s\n' "${all_seeds[@]}" >"${task_root}/selected_seeds.txt"
run_stream 0 "${gpu0_seeds[@]}" >"${task_root}/gpu0_supervisor.log" 2>&1 & p0=$!
run_stream 1 "${gpu1_seeds[@]}" >"${task_root}/gpu1_supervisor.log" 2>&1 & p1=$!
run_stream 2 "${gpu2_seeds[@]}" >"${task_root}/gpu2_supervisor.log" 2>&1 & p2=$!
run_stream 3 "${gpu3_seeds[@]}" >"${task_root}/gpu3_supervisor.log" 2>&1 & p3=$!

printf 'gpu0_pid=%s\ngpu1_pid=%s\ngpu2_pid=%s\ngpu3_pid=%s\n' \
  "${p0}" "${p1}" "${p2}" "${p3}" >"${task_root}/scissors_12seed_4gpu.pids"

status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
wait "${p2}" || status=1
wait "${p3}" || status=1
exit "${status}"
