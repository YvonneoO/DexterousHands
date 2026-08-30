#!/usr/bin/env python3
"""Generate a fixed, deterministic episode manifest for one task's Stage-2 BC
data -- the single source of truth for which successful episodes go into the
train/val split, reused identically across all three arms (P-only, P+GT-tac,
and later P+Pred-tac) so an SR difference between arms is never confounded by
which episodes each one happened to see.

Each task's successful episodes are split across MULTIPLE shard directories
(one per collection seed, or ad-hoc named batches for Pen), NOT one flat
successful_episodes/ folder per task -- and episode ids restart at 0 within
each shard. So a sample is identified by (shard_dir, episode_dirname), not a
bare int, and every consumer downstream (train_bc_student.py, and later the
offline pred-tac labeling script) must key on that pair.

Usage:
    python generate_episode_manifest.py \
        --shard_glob "<DEXTEROUSHANDS_ROOT>/runs/tactile_dataset/shadow_hand_pen/*" \
        --task shadow_hand_pen --max_episodes 100 --val_frac 0.1 --seed 0 \
        --out shadow_hand_pen_manifest.json

--shard_glob may be repeated (e.g. Pen's several differently-named batches
all need their own glob, since they don't share one prefix). Every matched
directory containing a successful_episodes/ subfolder is treated as one
shard; episodes missing trajectory_env0.npz, pressure_grids.npz, or
rgb_frames/ are skipped rather than silently included half-complete.

Not yet run anywhere -- needs to run WHERE the data actually lives (VISION),
since this machine has no local copy of the collected episodes. The exact
--shard_glob patterns below are inferred from the collection launch scripts,
not verified against a real directory listing -- confirm they resolve to the
expected episode counts before trusting the output.
"""
import os
import glob
import json
import argparse

import numpy as np


def discover_episodes(shard_globs):
    """Returns list of {"shard_dir", "episode"} for every successful_episodes/
    episode_* found under any directory matching any of the given globs."""
    shard_dirs = []
    for pattern in shard_globs:
        shard_dirs.extend(sorted(glob.glob(pattern)))

    episodes = []
    seen_shards = set()
    for shard_dir in shard_dirs:
        if shard_dir in seen_shards:
            continue
        seen_shards.add(shard_dir)
        success_dir = os.path.join(shard_dir, "successful_episodes")
        if not os.path.isdir(success_dir):
            continue
        for ep_dirname in sorted(os.listdir(success_dir)):
            ep_path = os.path.join(success_dir, ep_dirname)
            if not ep_dirname.startswith("episode_") or not os.path.isdir(ep_path):
                continue
            has_traj = os.path.isfile(os.path.join(ep_path, "trajectory_env0.npz"))
            has_press = os.path.isfile(os.path.join(ep_path, "pressure_grids.npz"))
            has_rgb = os.path.isdir(os.path.join(ep_path, "rgb_frames"))
            if has_traj and has_press and has_rgb:
                episodes.append({"shard_dir": shard_dir, "episode": ep_dirname})
            else:
                print(f"[skip] incomplete episode: {ep_path} "
                      f"(traj={has_traj} press={has_press} rgb={has_rgb})", flush=True)
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_glob", action="append", required=True,
                     help="glob matching shard directories for this task; repeatable")
    ap.add_argument("--task", required=True)
    ap.add_argument("--max_episodes", type=int, default=100)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes = discover_episodes(args.shard_glob)
    n_shards = len(set(e["shard_dir"] for e in episodes))
    print(f"[discover] {len(episodes)} complete successful episodes found "
          f"across {n_shards} shard dirs", flush=True)
    if len(episodes) < args.max_episodes:
        print(f"[warn] only {len(episodes)} episodes available, "
              f"fewer than --max_episodes {args.max_episodes}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    idx = idx[:args.max_episodes]
    n_val = max(1, int(round(len(idx) * args.val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    manifest = {
        "task": args.task,
        "seed": args.seed,
        "max_episodes": args.max_episodes,
        "val_frac": args.val_frac,
        "shard_globs": args.shard_glob,
        "train": [episodes[i] for i in train_idx],
        "val": [episodes[i] for i in val_idx],
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] {len(manifest['train'])} train / {len(manifest['val'])} val "
          f"episodes written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
