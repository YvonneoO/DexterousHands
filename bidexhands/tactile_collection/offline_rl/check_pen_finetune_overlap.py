#!/usr/bin/env python3
"""One-off diagnostic: does an offline-RL/BC episode manifest (Pen or Scissors)
overlap with the episodes v2-dit's sim-finetune
(checkpoints/v2_wilor_feat_dit_sim_finetune_1k) was itself trained/val'd on?

The finetune's episode selection is Ego2Contact's plan_shards.py `collect_episodes()`
(scripts/sim_data_collection/dexteroushands/plan_shards.py): walks each task's shard
roots in a FIXED order, taking `sorted(os.listdir(successful_episodes))` within each
root, and stops once n_total episodes are collected. Per experiments/STATUS.md the
"1k" run used 900 train / 100 val PER TASK -- i.e. the first 900 episodes (in that
deterministic order) were the finetune's TRAIN set, and the next 100 were its VAL set.

This script re-derives that exact same first-1000 slice directly against the live
filesystem (no assumptions about per-shard episode counts, shared with
dump_finetune_exposed_episodes.py's TASK_ROOTS), then compares it against an existing
offline-RL/BC manifest JSON (the {"shard_dir":..., "episode":...} pair schema from
generate_episode_manifest.py) to report exact overlap counts.

Usage:
    python check_pen_finetune_overlap.py --task pen --manifest pen_manifest_1000.json
    python check_pen_finetune_overlap.py --task scissors --manifest scissors_manifest_1000.json
"""
import argparse
import json
import os

from dump_finetune_exposed_episodes import TASK_ROOTS


def collect_episodes(task, n_total):
    """Mirrors plan_shards.py's collect_episodes() exactly: fixed root order,
    sorted-listdir within each root, stop once n_total reached. Returns a list
    of (shard_dir, episode_name) pairs -- shard_dir is the ROOT itself (matching
    generate_episode_manifest.py's manifest schema, where shard_dir is the
    collection-run directory, one level above successful_episodes/)."""
    eps = []
    for root in TASK_ROOTS[task]:
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
    ap.add_argument("--manifest", required=True, help="an offline_rl/bc manifest JSON")
    ap.add_argument("--task", required=True, choices=list(TASK_ROOTS.keys()))
    ap.add_argument("--finetune_train", type=int, default=900)
    ap.add_argument("--finetune_val", type=int, default=100)
    args = ap.parse_args()

    n_total = args.finetune_train + args.finetune_val
    finetune_eps = collect_episodes(args.task, n_total)
    finetune_train_set = set(finetune_eps[:args.finetune_train])
    finetune_val_set = set(finetune_eps[args.finetune_train:n_total])
    print(f"[finetune] re-derived {len(finetune_eps)} episodes "
          f"({len(finetune_train_set)} train / {len(finetune_val_set)} val) "
          f"from the fixed plan_shards.py order")
    if finetune_eps:
        print(f"[finetune] first episode: {finetune_eps[0]}")
        print(f"[finetune] train/val boundary episode: {finetune_eps[args.finetune_train - 1]} "
              f"-> {finetune_eps[args.finetune_train]}")
        print(f"[finetune] last episode: {finetune_eps[-1]}")

    with open(args.manifest) as f:
        manifest = json.load(f)
    manifest_train = {(e["shard_dir"], e["episode"]) for e in manifest["train"]}
    manifest_val = {(e["shard_dir"], e["episode"]) for e in manifest.get("val", [])}
    print(f"[manifest] {args.manifest}: {len(manifest_train)} train / {len(manifest_val)} val episodes")

    overlap_train_train = manifest_train & finetune_train_set
    overlap_train_val = manifest_train & finetune_val_set
    overlap_val_train = manifest_val & finetune_train_set
    overlap_val_val = manifest_val & finetune_val_set
    total_overlap = len(overlap_train_train | overlap_train_val | overlap_val_train | overlap_val_val)

    print(f"[overlap] manifest-train ^ finetune-train: {len(overlap_train_train)}")
    print(f"[overlap] manifest-train ^ finetune-val:   {len(overlap_train_val)}")
    print(f"[overlap] manifest-val   ^ finetune-train: {len(overlap_val_train)}")
    print(f"[overlap] manifest-val   ^ finetune-val:   {len(overlap_val_val)}")
    print(f"[overlap] TOTAL overlapping episodes: {total_overlap} "
          f"({100.0 * total_overlap / max(1, len(manifest_train) + len(manifest_val)):.1f}% "
          f"of the manifest)")

    clean_train = manifest_train - finetune_train_set - finetune_val_set
    clean_val = manifest_val - finetune_train_set - finetune_val_set
    print(f"[clean] episodes in manifest with ZERO finetune exposure: "
          f"{len(clean_train)} train / {len(clean_val)} val "
          f"({len(clean_train) + len(clean_val)} total)")


if __name__ == "__main__":
    main()
