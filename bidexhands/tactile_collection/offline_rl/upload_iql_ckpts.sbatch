#!/bin/bash
# Upload completed IQL checkpoints to HF then trim per-epoch snapshots off VISION
# once upload is confirmed (iql_final.d3/iql_final_policy.pt stay). See
# upload_iql_ckpts.py's own docstring for the full design.
#
# MUST be submitted with --dependency=afterany:<the 4 sweep-eval job ids> -- those
# jobs read model_<step>.d3 files mid-run; deleting checkpoints while a sweep is
# still in progress would break it. Example:
#   sbatch --dependency=afterany:513786:513787:513788:513789 upload_iql_ckpts.sbatch
#
#SBATCH --job-name=iql_ckpt_upload
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/iql_ckpt_upload_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/iql_ckpt_upload_%j.err

set -euo pipefail
YQQ=/scratch/project/prj-02-phai-lab/yqq
source "${YQQ}/env.sh"

PYTHON_BIN="${YQQ}/envs/d3rlpy_offline/bin/python"
if ! "${PYTHON_BIN}" -c "import huggingface_hub" 2>/dev/null; then
  echo "[upload] huggingface_hub not present in d3rlpy_offline env, installing (small, no known conflicts)"
  "${PYTHON_BIN}" -m pip install -q huggingface_hub
fi

echo "[upload] whoami check:"
"${PYTHON_BIN}" -c "from huggingface_hub import HfApi; import os; print(HfApi(token=os.environ['HF_TOKEN']).whoami()['name'])"

SCRIPT_DIR="${YQQ}/DexterousHands/bidexhands/tactile_collection/offline_rl"
cd "${YQQ}/DexterousHands/bidexhands"
"${PYTHON_BIN}" "${SCRIPT_DIR}/upload_iql_ckpts.py"

echo IQL_CKPT_UPLOAD_DONE
