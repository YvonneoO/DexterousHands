#!/usr/bin/env bash
# One-off sanity check: does bidexhands_isaacgym_py38 still import isaacgym + gym
# cleanly? Written as a real script (not an inline sbatch --wrap string) because
# the LD_LIBRARY_PATH export this needs (matching scripts/collect_wilor_raw_rigid.sh's
# own working pattern -- conda activate alone does NOT set this reliably enough for
# isaacgym's precompiled .so bindings to find libpython3.8.so.1.0) kept getting
# mangled through nested tmux send-keys quoting layers.
set -euo pipefail

YQQ="/scratch/project/prj-02-phai-lab/yqq"
source "${YQQ}/env.sh"   # module load Miniforge3 CUDA/12.8.0 -- without this, numpy's C
                         # extension segfaults on import ("PyCapsule_Import could not
                         # import module 'datetime'", confirmed reproducing this session's
                         # earlier job-511093 failure when this step is skipped).

ENV_ROOT="${YQQ}/envs/bidexhands_isaacgym_py38"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# ninja (needed to JIT-compile isaacgym's gymtorch C++ extension, imported
# transitively by every bidexhands task file) lives at ${ENV_ROOT}/bin/ninja
# but is NOT put on PATH by anything else here -- a real `conda activate`
# would add this automatically; the manual export-only approach used
# elsewhere in this diagnostic session never did. Confirmed via
# `RuntimeError: Ninja is required to load C++ extensions` even after
# isaacgym/gym/torch had ALL already imported successfully.
export PATH="${ENV_ROOT}/bin:${PATH}"

cd "${YQQ}/DexterousHands/bidexhands"
BIDEX_RECORD_DIR="${YQQ}/tmp_smoke_rollout" BIDEX_TARGET_SUCCESSES=1 BIDEX_MAX_EPISODES=5 \
  "${ENV_ROOT}/bin/python" -m tactile_collection.ppo.rollout_success_videos \
  --task ShadowHandPen --algo ppo --cfg_env cfg/ShadowHandPenProprioOnly.yaml \
  --model_dir logs/ShadowHandPen/ppo/ppo_seed0/model_6500.pt \
  --num_envs 1 --headless --seed 1234
