#!/usr/bin/env python3
"""Dump the exact set of episodes v2-dit's sim-finetune
(checkpoints/v2_wilor_feat_dit_sim_finetune_1k) was trained/val'd on for a given
bidexhands task, as a flat JSON list of [shard_dir, episode] pairs -- for use as
generate_episode_manifest.py's new --exclude argument, so downstream Pred-Tac /
offline-RL manifests never draw an episode the tactile predictor already saw
during its own fine-tuning (see check_pen_finetune_overlap.py, which found 191/
1000 episodes -- 19.1% -- of the FIRST pen_manifest_1000.json overlapping with
the finetune's own split before this exclusion existed).

Mirrors Ego2Contact's scripts/sim_data_collection/dexteroushands/plan_shards.py
TASKS roots + collect_episodes() EXACTLY (same fixed root order, same
sorted-listdir-within-root, same stop-at-n_total) -- re-derived from the live
filesystem rather than hardcoded, since no local task_split_1k.json exists to
read directly (VISION-only artifact, per the investigation that motivated this
script).

Usage:
    python dump_finetune_exposed_episodes.py --task pen \
        --out finetune_exposed_pen.json
    python dump_finetune_exposed_episodes.py --task scissors \
        --out finetune_exposed_scissors.json
"""
import argparse
import json
import os

BASE = "/scratch/project/prj-02-phai-lab/yqq/DexterousHands/runs/tactile_dataset"

TASK_ROOTS = {
    "scissors": [f"{BASE}/shadow_hand_scissors/seed_{s}_n250" for s in range(5100, 5112)],
    "pen": [
        f"{BASE}/shadow_hand_pen/{n}" for n in [
            "wilor_view_raw_rigid_batch_0000_0200", "wilor_view_raw_rigid_data_0200_1200",
            "wilor_view_raw_rigid_data_1200_2200", "wilor_view_raw_rigid_data_2200_3200",
            "wilor_view_raw_rigid_data_3200_3242_repair", "wilor_view_raw_rigid_data_3200_4200",
            "wilor_view_raw_rigid_data_4200_4600", "wilor_view_raw_rigid_data_4600_5000",
        ]
    ],
}


def collect_episodes(roots, n_total):
    """Exact mirror of plan_shards.py's collect_episodes()."""
    eps = []
    for root in roots:
        d = os.path.join(root, "successful_episodes")
        if not os.path.isdir(d):
            continue
        for ep in sorted(os.listdir(d)):
            ep_dir = os.path.join(d, ep)
            if os.path.isdir(os.path.join(ep_dir, "rgb_frames")):
                eps.append((root, ep))
        if len(eps) >= n_total:
            break
    return eps[:n_total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_ROOTS.keys()))
    ap.add_argument("--n_train", type=int, default=900)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eps = collect_episodes(TASK_ROOTS[args.task], args.n_train + args.n_val)
    if len(eps) < args.n_train + args.n_val:
        print(f"[warn] only found {len(eps)} episodes, expected "
              f"{args.n_train + args.n_val} -- exclusion list will be incomplete", flush=True)

    with open(args.out, "w") as f:
        json.dump([[shard_dir, ep] for shard_dir, ep in eps], f, indent=2)
    print(f"[dump] wrote {len(eps)} finetune-exposed episodes for task={args.task} "
          f"to {args.out}", flush=True)


if __name__ == "__main__":
    main()
