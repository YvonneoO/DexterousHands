#!/usr/bin/env bash
# One-off sanity check: does bidexhands_isaacgym_py38 still import isaacgym + gym
# cleanly? Written as a real script (not an inline sbatch --wrap string) because
# the LD_LIBRARY_PATH export this needs (matching scripts/collect_wilor_raw_rigid.sh's
# own working pattern -- conda activate alone does NOT set this reliably enough for
# isaacgym's precompiled .so bindings to find libpython3.8.so.1.0) kept getting
# mangled through nested tmux send-keys quoting layers.
set -euo pipefail

ENV_ROOT="/scratch/project/prj-02-phai-lab/yqq/envs/bidexhands_isaacgym_py38"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${ENV_ROOT}/bin/python" -c "
import isaacgym
import gym
print('ISAACGYM_GYM_IMPORT_OK gym_version=' + gym.__version__)
"
