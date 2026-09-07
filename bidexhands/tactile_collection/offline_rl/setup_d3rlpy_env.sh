#!/bin/bash
# One-time setup for the `d3rlpy_offline` VISION conda env used by
# offline_rl_iql.sbatch / build_mdp_dataset.py / train_iql.py.
#
# NOT run automatically by anything in this pipeline -- run manually, once,
# from an interactive VISION session (env creation itself is light enough
# for the shared login node; just don't run actual training there).
#
# Matches this project's VISION conventions (see rules/vision-workflow.md):
# envs live under yqq/envs/, caches point into yqq/.cache/, never /home or /tmp.
set -euo pipefail

YQQ="/scratch/project/prj-02-phai-lab/yqq"
source "${YQQ}/env.sh"

module load Miniforge3
module load CUDA/12.8.0
module load cuDNN

conda create -y -p "${YQQ}/envs/d3rlpy_offline" python=3.11
# conda create only sets up the prefix; still need to activate it explicitly
# in this same shell before pip-installing into it.
source activate "${YQQ}/envs/d3rlpy_offline"

export PIP_CACHE_DIR="${YQQ}/.cache/pip"
export TORCH_HOME="${YQQ}/.cache/torch"

# torch pinned to a cu128 wheel to match the H200 nodes' driver (see
# ego2contact-vision-setup memory: this project's other VISION envs use
# torch+cu128); d3rlpy itself does not pin a torch version.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install d3rlpy
# optional -- only exercised if train_iql.py's --wandb_project is passed AND
# the installed d3rlpy has d3rlpy.logging.WanDBAdapterFactory (train_iql.py
# falls back to file/CSV-only logging if not); harmless to install regardless.
pip install wandb

python -c "import d3rlpy, torch; print('d3rlpy', d3rlpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
