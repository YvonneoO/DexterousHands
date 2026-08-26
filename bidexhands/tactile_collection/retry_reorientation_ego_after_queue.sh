#!/usr/bin/env bash
set -uo pipefail

repo=/lp-dev/qianqian/DexterousHands
python_bin=/lp-dev/qianqian/envs/rlgpu/bin/python
queue_pid="${1:-31314}"
while [[ -r "/proc/${queue_pid}/stat" ]] && [[ "$(awk '{print $3}' "/proc/${queue_pid}/stat")" != "Z" ]]; do
  sleep 30
done

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
export BIDEX_TACTILE_STEPS=12000
export BIDEX_TACTILE_DIR="${repo}/runs/task_pipeline/gpu2/shadow_hand_re_orientation/evaluation/tactile_pa/paired_ego_success_v1"

printf '%s\trunning_retry\tcorrect_train_cfg seed=1002\n' ShadowHandReOrientation \
  | tee -a "${repo}/runs/task_pipeline/paired_ego_collection/status.tsv"
if "${python_bin}" -m tactile_collection.rollout_tactile_rgb \
    --task ShadowHandReOrientation --algo ppo \
    --model_dir "${repo}/runs/formal_2048/reorient_seed2/model_6500.pt" \
    --cfg_train cfg/ppo/re_orientation_config.yaml \
    --num_envs 1 --headless --test --seed 1002 --pipeline cpu \
    > "${BIDEX_TACTILE_DIR}.retry.log" 2>&1; then
  if [[ -f "${BIDEX_TACTILE_DIR}/successful_episode/trajectory_env0.npz" ]]; then
    printf '%s\tcomplete_retry\t%s\n' ShadowHandReOrientation "${BIDEX_TACTILE_DIR}" \
      | tee -a "${repo}/runs/task_pipeline/paired_ego_collection/status.tsv"
  else
    printf '%s\tzero_success_retry\tbudget=12000\n' ShadowHandReOrientation \
      | tee -a "${repo}/runs/task_pipeline/paired_ego_collection/status.tsv"
  fi
else
  printf '%s\tfailed_retry\t%s\n' ShadowHandReOrientation "${BIDEX_TACTILE_DIR}.retry.log" \
    | tee -a "${repo}/runs/task_pipeline/paired_ego_collection/status.tsv"
fi
