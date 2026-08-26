#!/usr/bin/env bash
# Complete the validated ShadowHandPen dataset to exactly 5,000 usable
# successful episodes. Run inside bidexhands-rl. At most two collectors run.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
collector_script="${COLLECTOR_SCRIPT:-${script_dir}/collect_shadow_hand_pen_5_success_wilor_raw_rigid.sh}"
dex_root="${DEXTEROUSHANDS_ROOT:-/lp-dev/qianqian/DexterousHands}"
base="${DATASET_ROOT:-${dex_root}/runs/tactile_dataset/shadow_hand_pen}"
python_bin="${PYTHON_BIN:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
qa_script="${QA_SCRIPT:-${dex_root}/bidexhands/scripts/qa_success_npz.py}"

if [[ ! -x "${collector_script}" || ! -f "${qa_script}" ]]; then
  echo "Collector launcher or QA script is missing: ${collector_script} ${qa_script}" >&2
  exit 2
fi

run_shard() {
  local gpu="$1" seed="$2" target="$3" max_steps="$4" output="$5"
  mkdir -p "${output}"
  DEXTEROUSHANDS_ROOT="${dex_root}" \
  PYTHON_BIN="${python_bin}" \
  GPU_ID="${gpu}" GRAPHICS_DEVICE_ID="${gpu}" \
  ROLLOUT_SEED="${seed}" TARGET_SUCCESSES="${target}" MAX_STEPS="${max_steps}" \
  SUCCESS_ONLY_BUFFER=0 WRITE_SUCCESS_SIDE_BY_SIDE=0 \
  OUTPUT_DIR="${output}" \
  bash "${collector_script}" >"${output}/collector.log" 2>&1
  "${python_bin}" "${qa_script}" \
    --require-coverage --min-mapped-force-fraction 0.95 \
    "${output}" >"${output}/coverage_qa_report.json"
}

out_a="${base}/wilor_view_raw_rigid_data_4200_4600"
out_b="${base}/wilor_view_raw_rigid_data_4600_5000"
out_repair="${base}/wilor_view_raw_rigid_data_3200_3242_repair"

# Isaac Gym's headless renderer always opens a small context on physical GPU 0.
# Therefore GPU 0 runs 400 and then the repair, while GPU 1 runs 400. This
# keeps the process set on exactly two physical GPUs (0 and 1).
# The result is 4,158 existing + 400 + 400 + 42 = 5,000 usable episodes.
(
  run_shard 0 5004 400 60000 "${out_a}"
  run_shard 0 5006 42 10000 "${out_repair}"
) &
pid_a=$!

run_shard 1 5005 400 60000 "${out_b}" &
pid_b=$!

printf 'gpu0_supervisor_pid=%s\ngpu1_collector_pid=%s\n' "${pid_a}" "${pid_b}" | tee "${base}/bulk_2gpu_20260826.pids"
wait "${pid_a}"
wait "${pid_b}"
echo "ShadowHandPen bulk completion sequence finished."
