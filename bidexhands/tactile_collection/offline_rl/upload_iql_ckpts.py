#!/usr/bin/env python3
"""Upload completed offline-RL IQL checkpoints (every per-epoch model_<step>.d3
snapshot + the final policy) to HF qqyang/hora-v4-shadow-tennis, under
tactile_sr_ablation/offline_rl_iql/<task>/<task>_<arm>/ -- mirrors this repo's
existing tactile_sr_ablation/vtdexmanip/checkpoints/ convention (see VTDexManip's
own upload_ckpt_and_rollout.py). Folder naming embeds task AND arm explicitly
(p_only / p_gt_tac / p_pred_tac) so runs never collide and a future p_pred_tac
arm needs no naming change -- RUNS below is just data.

Once (and only once) a run's checkpoint dir is confirmed uploaded, it's deleted
from VISION -- iql_final.d3/iql_final_policy.pt (train_iql.py's own explicit
final-policy save, NOT one of the epoch snapshots, lives one level up at run_dir)
are kept on VISION; that's the "last checkpoint" the epoch snapshots become
redundant with once backed up. Never trims on a failed upload (upload_folder
raises on failure, which aborts that run's function before the delete line runs).

Uses HfApi.upload_folder (ONE commit per folder), not upload_file per checkpoint --
an earlier version of this script called upload_file once per model_<step>.d3
(~100 checkpoints/run) and hit HF's 128-commits/hour rate limit partway through
the FIRST run, crashing with 429 Too Many Requests (job 513808, 2026-09-07).

Usage:
    python upload_iql_ckpts.py               # all runs in RUNS
    python upload_iql_ckpts.py --dry_run      # list what would upload/delete, do nothing
"""
import argparse
import os
import shutil

from huggingface_hub import HfApi

REPO_ID = "qqyang/hora-v4-shadow-tennis"
YQQ = "/scratch/project/prj-02-phai-lab/yqq"
RUNS_ROOT = f"{YQQ}/runs/offline_rl"

# (task, arm, ckpt_subdir) -- ckpt_subdir is the d3rlpy-timestamped dir under
# <RUNS_ROOT>/<task>/iql_<arm>/ holding the per-epoch model_<step>.d3 snapshots
# (confirmed via `ls` on VISION 2026-09-07). Extend this list as more runs finish
# (e.g. p_pred_tac once its Pred-Tac generation + IQL training completes).
RUNS = [
    ("pen", "p_only", "iql_p_only_20260907005003"),
    ("pen", "p_gt_tac", "iql_p_gt_tac_20260907005440"),
    ("scissors", "p_only", "iql_p_only_20260907002040"),
    ("scissors", "p_gt_tac", "iql_p_gt_tac_20260907002049"),
]


def upload_and_trim(api, task, arm, ckpt_subdir, dry_run):
    run_dir = os.path.join(RUNS_ROOT, task, f"iql_{arm}")
    step_dir = os.path.join(run_dir, ckpt_subdir)
    hf_prefix = f"tactile_sr_ablation/offline_rl_iql/{task}/{task}_{arm}"

    final_files = [f for f in ("iql_final.d3", "iql_final_policy.pt")
                   if os.path.isfile(os.path.join(run_dir, f))]
    has_step_dir = os.path.isdir(step_dir)

    if not final_files:
        print(f"[skip] {task}/{arm}: no iql_final.d3/iql_final_policy.pt under {run_dir}", flush=True)
        return
    if not has_step_dir:
        print(f"[warn] {task}/{arm}: no checkpoint subdir {step_dir} -- nothing to trim, "
              f"uploading final files only", flush=True)

    print(f"[{task}/{arm}] final_files={final_files} step_dir={'yes' if has_step_dir else 'no'} "
          f"-> {hf_prefix}/", flush=True)
    if dry_run:
        print(f"  DRY-RUN upload_folder({run_dir}, only final files) -> {hf_prefix}/", flush=True)
        if has_step_dir:
            print(f"  DRY-RUN upload_folder({step_dir}) -> {hf_prefix}/{ckpt_subdir}/", flush=True)
            print(f"  DRY-RUN would then rmtree {step_dir}", flush=True)
        return

    api.upload_folder(folder_path=run_dir, path_in_repo=hf_prefix, repo_id=REPO_ID,
                       repo_type="model", allow_patterns=["iql_final.d3", "iql_final_policy.pt"])
    print(f"  uploaded final files -> {hf_prefix}/", flush=True)

    if not has_step_dir:
        return

    api.upload_folder(folder_path=step_dir, path_in_repo=f"{hf_prefix}/{ckpt_subdir}",
                       repo_id=REPO_ID, repo_type="model")
    print(f"  uploaded {step_dir} -> {hf_prefix}/{ckpt_subdir}/", flush=True)

    # Both upload_folder calls above returned (didn't raise) -- only now is it
    # safe to trim. Removes the WHOLE step_dir (not just model_*.d3 -- config.json/
    # metrics csvs in there are already backed up too); iql_final.d3/
    # iql_final_policy.pt (the "last checkpoint") stay untouched at run_dir.
    shutil.rmtree(step_dir)
    print(f"[trim] {task}/{arm}: removed {step_dir} -- iql_final.d3/iql_final_policy.pt "
          f"kept on VISION", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    whoami = api.whoami()["name"]
    assert whoami == "qqyang", f"HF_TOKEN resolves to '{whoami}', not qqyang -- source yqq/env.sh first"

    for task, arm, ckpt_subdir in RUNS:
        upload_and_trim(api, task, arm, ckpt_subdir, args.dry_run)


if __name__ == "__main__":
    main()
