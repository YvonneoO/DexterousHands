#!/usr/bin/env bash
set -uo pipefail

wait_pid="${1:-44979}"
while [[ -r "/proc/${wait_pid}/stat" ]] && [[ "$(awk '{print $3}' "/proc/${wait_pid}/stat")" != "Z" ]]; do
  sleep 30
done

repo=/lp-dev/qianqian/DexterousHands
python_bin=/lp-dev/qianqian/envs/rlgpu/bin/python
out="${repo}/runs/task_pipeline/gpu3/shadow_hand_door_open_outward/evaluation/tactile_pa/paired_ego_success_v1"
status="${repo}/runs/task_pipeline/paired_ego_collection/status.tsv"
cd "${repo}/bidexhands"
export CUDA_VISIBLE_DEVICES=0 PYTHON="${python_bin}" LD_LIBRARY_PATH=/lp-dev/qianqian/envs/rlgpu/lib
export PYTHONUNBUFFERED=1 BIDEX_STOP_AFTER_SUCCESSES=1 BIDEX_SUCCESS_ONLY_BUFFER=1
export BIDEX_FRAME_STRIDE=2 BIDEX_VIDEO_FPS=30 BIDEX_VIDEO_WIDTH=640 BIDEX_VIDEO_HEIGHT=480
export BIDEX_TACTILE_STEPS=800 BIDEX_TACTILE_DIR="${out}"
if [[ -d "${out}/successful_episode" ]]; then
  mv "${out}/successful_episode" "${out}/successful_episode.valid_contact_root_pose_only"
fi
printf '%s\trunning_object_state_repair\tall_object_bodies_and_dofs seed=1041\n' ShadowHandDoorOpenOutward | tee -a "${status}"
if "${python_bin}" -m tactile_collection.rollout_tactile_rgb \
    --task ShadowHandDoorOpenOutward --algo ppo \
    --model_dir "${repo}/runs/task_pipeline/gpu3/shadow_hand_door_open_outward/full_2048_seed41/model_6500.pt" \
    --cfg_train cfg/ppo/config.yaml --num_envs 1 --headless --test --seed 1041 --pipeline cpu \
    > "${out}.object_state_repair.log" 2>&1; then
  printf '%s\tcomplete_object_state_repair\t%s\n' ShadowHandDoorOpenOutward "${out}" | tee -a "${status}"
else
  printf '%s\tfailed_object_state_repair\t%s\n' ShadowHandDoorOpenOutward "${out}.object_state_repair.log" | tee -a "${status}"
fi
