#!/usr/bin/env python3
"""Compute a task's GT tactile task_vmax (99th-percentile denormalization scalar),
for use as infer_pred_tactile_offline.py's required --task_vmax argument.

Same formula as train_bc_student.py's compute_tac_vmax() (percentile over
concatenated left+right pressure grids across a manifest's episodes, NaN treated
as 0) -- reimplemented standalone here rather than importing that module, since
train_bc_student.py pulls in torch/PIL/Dataset just to get one pure-numpy
function. Confirmed to reproduce the known Pen value (91130.06695312512, from
Ego2Contact's infer_pred_tactile_offline.py docstring) when run over the same
episode population -- this script exists mainly to get the SAME number for
tasks (Scissors, Over, Door) that don't have a recorded reference value.

Usage:
    python compute_task_vmax.py --manifest scissors_manifest_1000_clean.json \
        --percentile 99 --n_episodes 200
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--n_episodes", type=int, default=200,
                     help="sample this many train-split episodes (not all 900) -- "
                          "a percentile over a few hundred episodes' worth of "
                          "per-step grids is already a huge sample, no need to "
                          "load the full split")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    entries = manifest["train"][:args.n_episodes]

    values = []
    for e in entries:
        ep_dir = os.path.join(e["shard_dir"], "successful_episodes", e["episode"])
        with np.load(os.path.join(ep_dir, "pressure_grids.npz")) as press:
            left = np.nan_to_num(np.asarray(press["left_pressure_grid"], dtype=np.float32))
            right = np.nan_to_num(np.asarray(press["right_pressure_grid"], dtype=np.float32))
            values.append(left.reshape(-1))
            values.append(right.reshape(-1))
    all_values = np.concatenate(values)
    vmax = max(float(np.percentile(all_values, args.percentile)), 1e-6)
    print(f"[vmax] task={manifest.get('task')} n_episodes={len(entries)} "
          f"percentile={args.percentile} -> task_vmax={vmax}", flush=True)


if __name__ == "__main__":
    main()
