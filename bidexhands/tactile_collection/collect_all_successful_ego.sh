#!/usr/bin/env bash
set -uo pipefail

repo=/lp-dev/qianqian/DexterousHands
python_bin=/lp-dev/qianqian/envs/rlgpu/bin/python
cd "${repo}/bidexhands"
export CUDA_VISIBLE_DEVICES=0
export PYTHON="${python_bin}"
export LD_LIBRARY_PATH=/lp-dev/qianqian/envs/rlgpu/lib
export PYTHONUNBUFFERED=1
export BIDEX_STOP_AFTER_SUCCESSES=1
export BIDEX_FRAME_STRIDE=2
export BIDEX_VIDEO_FPS=30
export BIDEX_VIDEO_WIDTH=640
export BIDEX_VIDEO_HEIGHT=480

# Only tasks with non-zero native deterministic success in the completed
# evaluation are included. Budgets are search caps, not training extensions;
# collection stops immediately after the first successful episode.
jobs=(
  'ShadowHandOver|gpu0/shadow_hand_over|runs/formal_2048/over_seed0/model_6500.pt|cfg/ppo/config.yaml|800|1000'
  'ShadowHandReOrientation|gpu2/shadow_hand_re_orientation|runs/formal_2048/reorient_seed2/model_6500.pt|cfg/ppo/re_orientation_config.yaml|12000|1002'
  'ShadowHandBottleCap|gpu3/shadow_hand_bottle_cap|runs/formal_2048/bottlecap_seed3/model_6500.pt|cfg/ppo/config.yaml|30000|1003'
  'ShadowHandScissors|gpu6/shadow_hand_scissors|runs/task_pipeline/gpu6/shadow_hand_scissors/full_2048_seed60/model_6500.pt|cfg/ppo/config.yaml|800|1060'
  'ShadowHandCatchUnderarm|gpu0/shadow_hand_catch_underarm|runs/task_pipeline/gpu0/shadow_hand_catch_underarm/full_2048_seed10/model_6500.pt|cfg/ppo/config.yaml|1200|1010'
  'ShadowHandCatchOver2Underarm|gpu0/shadow_hand_catch_over2_underarm|runs/task_pipeline/gpu0/shadow_hand_catch_over2_underarm/full_2048_seed11/model_6500.pt|cfg/ppo/config.yaml|1200|1011'
  'ShadowHandDoorCloseInward|gpu0/shadow_hand_door_close_inward|runs/task_pipeline/gpu0/shadow_hand_door_close_inward/full_2048_seed70/model_6500.pt|cfg/ppo/config.yaml|800|1070'
  'ShadowHandGraspAndPlace|gpu1/shadow_hand_grasp_and_place|runs/task_pipeline/gpu1/shadow_hand_grasp_and_place/full_2048_seed20/model_6500.pt|cfg/ppo/config.yaml|1200|1020'
  'ShadowHandBlockStack|gpu1/shadow_hand_block_stack|runs/task_pipeline/gpu1/shadow_hand_block_stack/full_2048_seed21/model_6500.pt|cfg/ppo/stack_block_config.yaml|1600|1021'
  'ShadowHandSwingCup|gpu1/shadow_hand_swing_cup|runs/task_pipeline/gpu1/shadow_hand_swing_cup/full_2048_seed71/model_4000.pt|cfg/ppo/config.yaml|4000|1071'
  'ShadowHandPen|gpu2/shadow_hand_pen|runs/task_pipeline/gpu2/shadow_hand_pen/full_2048_seed32/model_6500.pt|cfg/ppo/config.yaml|800|1032'
  'ShadowHandDoorOpenOutward|gpu3/shadow_hand_door_open_outward|runs/task_pipeline/gpu3/shadow_hand_door_open_outward/full_2048_seed41/model_6500.pt|cfg/ppo/config.yaml|800|1041'
  'ShadowHandCatchAbreast|gpu2/shadow_hand_catch_abreast|runs/task_pipeline/gpu2/shadow_hand_catch_abreast/full_2048_seed73/model_6500.pt|cfg/ppo/config.yaml|1600|1073'
)

status_root="${repo}/runs/task_pipeline/paired_ego_collection"
mkdir -p "${status_root}"
: > "${status_root}/status.tsv"

for spec in "${jobs[@]}"; do
  IFS='|' read -r task task_path checkpoint_rel train_cfg budget seed <<< "${spec}"
  checkpoint="${repo}/${checkpoint_rel}"
  out="${repo}/runs/task_pipeline/${task_path}/evaluation/tactile_pa/paired_ego_success_v1"
  log="${out}.log"
  if [[ ! -f "${checkpoint}" ]]; then
    printf '%s\tmissing_checkpoint\t%s\n' "${task}" "${checkpoint}" | tee -a "${status_root}/status.tsv"
    continue
  fi
  if [[ -f "${out}/successful_episode/trajectory_env0.npz" ]]; then
    printf '%s\talready_complete\t%s\n' "${task}" "${out}" | tee -a "${status_root}/status.tsv"
    continue
  fi
  mkdir -p "${out}"
  export BIDEX_TACTILE_STEPS="${budget}"
  export BIDEX_TACTILE_DIR="${out}"
  printf '%s\trunning\tbudget=%s seed=%s\n' "${task}" "${budget}" "${seed}" | tee -a "${status_root}/status.tsv"
  if "${python_bin}" -m tactile_collection.rollout_tactile_rgb \
      --task "${task}" --algo ppo --model_dir "${checkpoint}" \
      --cfg_train "${train_cfg}" --num_envs 1 --headless --test \
      --seed "${seed}" --pipeline cpu \
      > "${log}" 2>&1; then
    if [[ -f "${out}/successful_episode/trajectory_env0.npz" ]]; then
      printf '%s\tcomplete\t%s\n' "${task}" "${out}" | tee -a "${status_root}/status.tsv"
    else
      printf '%s\tzero_success\tbudget=%s\n' "${task}" "${budget}" | tee -a "${status_root}/status.tsv"
    fi
  else
    printf '%s\tfailed\t%s\n' "${task}" "${log}" | tee -a "${status_root}/status.tsv"
  fi
done

printf 'all_done\t%s\n' "$(date -u +%FT%TZ)" | tee -a "${status_root}/status.tsv"
