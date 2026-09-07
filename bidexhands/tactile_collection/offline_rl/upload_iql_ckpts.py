#!/usr/bin/env python3
"""Upload completed offline-RL IQL checkpoints (every per-epoch model_<step>.d3
snapshot + the final policy) to HF qqyang/hora-v4-shadow-tennis, under
tactile_sr_ablation/offline_rl_iql/<task>/<task>_<arm>/ -- mirrors this repo's
existing tactile_sr_ablation/vtdexmanip/checkpoints/ convention (see VTDexManip's
own upload_ckpt_and_rollout.py). Folder naming embeds task AND arm explicitly
(p_only / p_gt_tac / p_pred_tac) so runs never collide and a future p_pred_tac
arm needs no naming change -- RUNS below is just data.

Once (and only once) every file for a run is confirmed uploaded, its per-epoch
model_<step>.d3 snapshots are deleted from VISION -- iql_final.d3/iql_final_policy.pt
(train_iql.py's own explicit final-policy save, NOT one of the epoch snapshots)
are kept on VISION; that's the "last checkpoint" the epoch snapshots become
redundant with once backed up. Never trims on a partial/failed upload.

Usage:
    python upload_iql_ckpts.py               # all runs in RUNS
    python upload_iql_ckpts.py --dry_run      # list what would upload/delete, do nothing
"""
import argparse
import glob
import os

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
    step_files = sorted(glob.glob(os.path.join(step_dir, "model_*.d3")))

    if not final_files:
        print(f"[skip] {task}/{arm}: no iql_final.d3/iql_final_policy.pt under {run_dir}", flush=True)
        return
    if not step_files:
        print(f"[warn] {task}/{arm}: no model_*.d3 under {step_dir} -- nothing to trim, "
              f"uploading final files only", flush=True)

    to_upload = [(os.path.join(run_dir, f), f"{hf_prefix}/{f}") for f in final_files]
    to_upload += [(p, f"{hf_prefix}/{ckpt_subdir}/{os.path.basename(p)}") for p in step_files]

    print(f"[{task}/{arm}] {len(to_upload)} files -> {hf_prefix}/", flush=True)
    if dry_run:
        for local, remote in to_upload:
            print(f"  DRY-RUN upload: {local} -> {remote}", flush=True)
        for p in step_files:
            print(f"  DRY-RUN would delete after upload: {p}", flush=True)
        return

    uploaded = []
    for local, remote in to_upload:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                         repo_id=REPO_ID, repo_type="model")
        uploaded.append(remote)
        print(f"  uploaded {remote}", flush=True)

    # Only delete local epoch snapshots once EVERY one of them (and the final
    # files) is confirmed present in `uploaded` -- never trim on a partial batch.
    expected = {f"{hf_prefix}/{f}" for f in final_files}
    expected |= {f"{hf_prefix}/{ckpt_subdir}/{os.path.basename(p)}" for p in step_files}
    missing = expected - set(uploaded)
    if missing:
        print(f"[abort-trim] {task}/{arm}: {len(missing)} file(s) failed to upload, "
              f"NOT deleting anything on VISION: {missing}", flush=True)
        return

    for p in step_files:
        os.remove(p)
    print(f"[trim] {task}/{arm}: removed {len(step_files)} epoch snapshot(s) from "
          f"{step_dir} -- iql_final.d3/iql_final_policy.pt kept on VISION", flush=True)


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
