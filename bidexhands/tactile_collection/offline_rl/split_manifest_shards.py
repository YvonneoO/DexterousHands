#!/usr/bin/env python3
"""Split an episode manifest (generate_episode_manifest.py's {"train":[...], "val":[...]}
schema) into N roughly-equal shard manifests, each preserving the same schema (its own
slice of train + its own slice of val) -- so Pred-Tac generation (infer_pred_tactile_
offline.py) can run as N independent single-GPU sbatch jobs instead of one job needing
multiple GPUs (which currently can't get scheduled on the shared VISION queue), each
job handling dozens of episodes.

Usage:
    python split_manifest_shards.py --manifest pen_manifest_1000_clean.json \
        --n_shards 10 --out_prefix pen_manifest_1000_clean_shard
    # writes pen_manifest_1000_clean_shard0.json .. _shard9.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n_shards", type=int, required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    train = manifest["train"]
    val = manifest.get("val", [])

    for i in range(args.n_shards):
        shard = dict(manifest)
        shard["train"] = train[i::args.n_shards]
        shard["val"] = val[i::args.n_shards]
        shard["shard_index"] = i
        shard["n_shards"] = args.n_shards
        out_path = f"{args.out_prefix}{i}.json"
        with open(out_path, "w") as f:
            json.dump(shard, f, indent=2)
        print(f"[shard {i}] {len(shard['train'])} train / {len(shard['val'])} val "
              f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
