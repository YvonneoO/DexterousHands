#!/usr/bin/env bash
set -uo pipefail

wait_pid="${1:-33080}"
while [[ -r "/proc/${wait_pid}/stat" ]] && [[ "$(awk '{print $3}' "/proc/${wait_pid}/stat")" != "Z" ]]; do
  sleep 30
done

repo=/lp-dev/qianqian/DexterousHands
python_bin=/lp-dev/qianqian/envs/rlgpu/bin/python
status="${repo}/runs/task_pipeline/paired_ego_collection/status.tsv"
cd "${repo}/bidexhands"
export CUDA_VISIBLE_DEVICES=0
export PYTHON="${python_bin}"
export LD_LIBRARY_PATH=/lp-dev/qianqian/envs/rlgpu/lib
export PYTHONUNBUFFERED=1
export BIDEX_STOP_AFTER_SUCCESSES=1
export BIDEX_SUCCESS_ONLY_BUFFER=1
export BIDEX_FRAME_STRIDE=2
export BIDEX_VIDEO_FPS=30
export BIDEX_VIDEO_WIDTH=640
export BIDEX_VIDEO_HEIGHT=480

run_task() {
  local task="$1" out="$2" checkpoint="$3" train_cfg="$4" budget="$5" seed="$6"
  if [[ -d "${out}/successful_episode" ]]; then
    mv "${out}/successful_episode" "${out}/successful_episode.invalid_zero_tactile_single_body_mapper"
  fi
  export BIDEX_TACTILE_DIR="${out}"
  export BIDEX_TACTILE_STEPS="${budget}"
  printf '%s\trunning_tactile_repair\tmulti_body_object_mapper budget=%s seed=%s\n' \
    "${task}" "${budget}" "${seed}" | tee -a "${status}"
  if "${python_bin}" -m tactile_collection.rollout_tactile_rgb \
      --task "${task}" --algo ppo --model_dir "${checkpoint}" \
      --cfg_train "${train_cfg}" --num_envs 1 --headless --test \
      --seed "${seed}" --pipeline cpu > "${out}.tactile_repair.log" 2>&1; then
    if [[ -f "${out}/successful_episode/trajectory_env0.npz" ]]; then
      printf '%s\tcomplete_tactile_repair\t%s\n' "${task}" "${out}" | tee -a "${status}"
    else
      printf '%s\tzero_success_tactile_repair\tbudget=%s\n' "${task}" "${budget}" | tee -a "${status}"
    fi
  else
    printf '%s\tfailed_tactile_repair\t%s\n' "${task}" "${out}.tactile_repair.log" | tee -a "${status}"
  fi
}

run_task ShadowHandBottleCap \
  "${repo}/runs/task_pipeline/gpu3/shadow_hand_bottle_cap/evaluation/tactile_pa/paired_ego_success_v1" \
  "${repo}/runs/formal_2048/bottlecap_seed3/model_6500.pt" cfg/ppo/config.yaml 30000 1003
run_task ShadowHandScissors \
  "${repo}/runs/task_pipeline/gpu6/shadow_hand_scissors/evaluation/tactile_pa/paired_ego_success_v1" \
  "${repo}/runs/task_pipeline/gpu6/shadow_hand_scissors/full_2048_seed60/model_6500.pt" cfg/ppo/config.yaml 800 1060

printf 'tactile_repairs_done\t%s\n' "$(date -u +%FT%TZ)" | tee -a "${status}"
