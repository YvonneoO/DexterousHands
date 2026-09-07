#!/usr/bin/env python3
"""Train IQL (Implicit Q-Learning) via d3rlpy's OWN implementation
(`d3rlpy.algos.IQLConfig`) on a dataset built by build_mdp_dataset.py.
Deliberately not a hand-rolled IQL -- the point of this pipeline is to reuse
a maintained offline-RL library's algorithm, not to re-derive it.

Usage:
    python train_iql.py --dataset shadow_hand_pen_p_gt_tac.h5 \
        --out runs/iql_pen_p_gt_tac --n_steps 100000

Not yet run anywhere -- no local d3rlpy install / cluster access from this
machine (see build_mdp_dataset.py's module docstring for the same caveat).
d3rlpy API surface used here (IQLConfig(...).create(device=...), .fit(...),
.save()/.save_policy(), d3rlpy.logging.*AdapterFactory) matches the
documented d3rlpy v2.x API as of this writing -- FLAG: if the pinned
version differs, `python -c "import d3rlpy; help(d3rlpy.algos.IQLConfig)"`
/ `help(d3rlpy.logging)` on the actual cluster env is the fastest way to
confirm before debugging a mysterious failure here.
"""
import os
import argparse

import numpy as np


def _load_dataset(path):
    """Load a dataset written by build_mdp_dataset.py's `dataset.dump(f)`.
    Confirmed against d3rlpy 2.8.1 on VISION (job 511910): MDPDataset.load()
    is NOT the right call -- MDPDataset(...) actually returns a ReplayBuffer
    instance, and reloading it needs `ReplayBuffer.load(f, buffer)` where
    `buffer` is a fresh InfiniteBuffer() for `load` to reconstruct episodes
    into (the bare MDPDataset.load(path)/```load(f)``` calls both raise
    "missing 1 required positional argument: 'buffer'")."""
    from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
    with open(path, "rb") as f:
        return ReplayBuffer.load(f, InfiniteBuffer())


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_logger_adapter(out_dir, wandb_project, wandb_name):
    """d3rlpy v2 ships a pluggable `d3rlpy.logging` adapter system (file/CSV,
    TensorBoard, and -- per a recent-enough d3rlpy release -- W&B). We're not
    fully certain the pinned cluster version has `WanDBAdapterFactory`
    (task explicitly said: don't guess at an API that might not exist), so
    this ALWAYS sets up the file/CSV logger (guaranteed to exist) and only
    layers wandb on top if the import succeeds -- printing a clear note and
    falling back to file-only logging otherwise, rather than crashing."""
    from d3rlpy.logging import FileAdapterFactory
    file_adapter = FileAdapterFactory(root_dir=out_dir)

    if wandb_project is None:
        return file_adapter

    try:
        from d3rlpy.logging import CombineAdapterFactory, WanDBAdapterFactory
    except ImportError:
        print("[wandb] d3rlpy.logging.WanDBAdapterFactory not available in this "
              "installed d3rlpy version -- continuing with file/CSV logging only "
              "(under --out/<experiment_name>/). Push those CSVs to wandb "
              "separately as a follow-up if needed, rather than guessing at an "
              "API this version doesn't have.", flush=True)
        return file_adapter

    wandb_kwargs = {"project": wandb_project}
    if wandb_name is not None:
        wandb_kwargs["experiment_name"] = wandb_name  # best-effort kwarg name; see note above
    try:
        wandb_adapter = WanDBAdapterFactory(**wandb_kwargs)
    except TypeError:
        wandb_adapter = WanDBAdapterFactory(project=wandb_project)
    return CombineAdapterFactory([file_adapter, wandb_adapter])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help=".h5 dataset from build_mdp_dataset.py")
    ap.add_argument("--out", required=True, help="output dir: policy checkpoint + logs")
    ap.add_argument("--n_steps", type=int, default=100_000)
    ap.add_argument("--n_steps_per_epoch", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                     help="e.g. 'cuda:0' or 'cpu:0'; default auto-detects CUDA")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_name", default=None)
    args = ap.parse_args()

    import d3rlpy
    from d3rlpy.algos import IQLConfig
    from d3rlpy.preprocessing import StandardObservationScaler

    d3rlpy.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    dataset = _load_dataset(args.dataset)
    print(f"[data] loaded {args.dataset}", flush=True)

    device = args.device or ("cuda:0" if _cuda_available() else "cpu:0")
    print(f"[train] device={device}", flush=True)

    # p_gt_tac/p_pred_tac observations concatenate proprioception (radians/meters,
    # O(0.1-3)) with RAW pressure_grids.npz values in Pa -- build_mdp_dataset.py's
    # flatten_tactile() applies no rescaling. Task contact-pressure vmax (99th
    # pctile) ranges from ~255 Pa (scissors) to ~19087 Pa (pen) -- 2 to 4 orders
    # of magnitude above proprio's own scale, with NO observation_scaler ever
    # configured here previously. Confirmed 2026-09-07: this correlates exactly
    # with per-task P+GT-tac IQL failure severity (pen, worst mismatch, never
    # learns at all; scissors, milder mismatch, learns then diverges/collapses
    # later in training) -- a classic unnormalized-input-scale optimization
    # failure, not evidence tactile itself is unhelpful. StandardObservationScaler
    # (mean=None,std=None) auto-fits from `dataset`'s own transition statistics
    # inside `.fit()` (via build_scalers_with_transition_picker, verified against
    # d3rlpy's real v2.8.1 source) -- no extra fit step needed here. Applying
    # this to EVERY arm (not just p_gt_tac) keeps p_only comparable/consistent,
    # and costs nothing for p_only's own naturally-small-scale proprio-only obs.
    iql = IQLConfig(batch_size=args.batch_size,
                     observation_scaler=StandardObservationScaler()).create(device=device)

    logger_adapter = _build_logger_adapter(args.out, args.wandb_project, args.wandb_name)
    experiment_name = os.path.basename(os.path.normpath(args.out))

    iql.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        experiment_name=experiment_name,
        logger_adapter=logger_adapter,
    )

    ckpt_path = os.path.join(args.out, "iql_final.d3")
    policy_path = os.path.join(args.out, "iql_final_policy.pt")
    iql.save(ckpt_path)          # full algorithm state (optimizer, both Q nets, V, actor)
    iql.save_policy(policy_path)  # inference-only exported policy
    print(f"[train] saved checkpoint -> {ckpt_path}", flush=True)
    print(f"[train] saved policy -> {policy_path}", flush=True)


if __name__ == "__main__":
    main()
