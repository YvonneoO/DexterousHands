#!/usr/bin/env python3
"""Sweep every numbered checkpoint from one train_bc_student.py run through a
quick success-rate eval, logging each point into the SAME wandb run training
already started -- produces a success-rate-vs-epoch curve next to the loss
curves. Mirrors the "quick biased val during training, full trustworthy eval
at the end" pattern already used for the human-data v2-dit training: each
checkpoint gets a small, fast (and therefore somewhat noisy) SR estimate for
the curve, and the last checkpoint additionally gets a much larger-N eval for
the trustworthy headline number.

Run as:
    cd <DexterousHands root>/bidexhands
    python -m tactile_collection.bc.sweep_eval_bc_student \
        --ckpt_dir <train_bc_student.py's --out> \
        --episodes_per_ckpt 20 --final_episodes 100 \
        -- --task ShadowHandPen --algo ppo

Reads <ckpt_dir>/wandb_run_id.txt (written by train_bc_student.py) to resume
logging into that exact run. If it's missing or wandb isn't installed, the
sweep still runs and prints results, just without logging.

Not yet run anywhere -- local draft only.
"""
import os
import re
import glob
import argparse

# rollout_eval_core pulls in isaacgym via bidexhands.utils.config; that must
# happen before `import torch` runs anywhere in the process, so it's imported
# first here even though torch is used below.
from tactile_collection.bc.rollout_eval_core import build_eval_context, run_rollout_eval

import torch

from tactile_collection.bc.train_bc_student import BCStudent

try:
    import wandb
except ImportError:
    wandb = None


def epoch_checkpoints(ckpt_dir):
    """Sorted list of (epoch, path) for every epoch_XXX.pt in ckpt_dir."""
    paths = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
    out = []
    for p in paths:
        m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                     help="train_bc_student.py's --out (must contain epoch_*.pt "
                          "checkpoints; wandb_run_id.txt is used if present)")
    ap.add_argument("--episodes_per_ckpt", type=int, default=20,
                     help="quick, biased SR estimate per checkpoint -- for the "
                          "curve, not the headline number")
    ap.add_argument("--final_episodes", type=int, default=100,
                     help="trustworthy final SR on the last checkpoint")
    ap.add_argument("--max_steps_per_episode", type=int, default=500)
    ap.add_argument("--frame_stride", type=int, default=2,
                     help="must match BIDEX_FRAME_STRIDE used for the training data")
    ap.add_argument("--camera_eye_offset", default="0.32,0.0,0.80")
    ap.add_argument("--camera_target_offset", default="0.0,0.0,0.08")
    ap.add_argument("--asset_dir", default="tactile_collection/assets")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--wandb_project", default="ego2contact-bc-distill")
    args, unknown = ap.parse_known_args()
    # parse_known_args leaves a literal "--" separator in `unknown` if the
    # caller passed one to mark where bidexhands' own args start; bidexhands'
    # parser doesn't handle a bare "--" and rejects everything after it.
    if unknown and unknown[0] == "--":
        unknown = unknown[1:]

    ckpts = epoch_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"no epoch_*.pt checkpoints found under {args.ckpt_dir}")
    print(f"[sweep] {len(ckpts)} checkpoints: epochs {[e for e, _ in ckpts]}", flush=True)

    first_ckpt = torch.load(ckpts[0][1], map_location="cpu")
    tac_mode = first_ckpt["tac_mode"]
    img_size = first_ckpt["args"]["img_size"]
    action_dim = first_ckpt["action_dim"]

    run_id_path = os.path.join(args.ckpt_dir, "wandb_run_id.txt")
    run = None
    if wandb is not None and os.path.isfile(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()
        run = wandb.init(project=args.wandb_project, id=run_id, resume="must")
        print(f"[wandb] resumed run {run_id}", flush=True)
    elif wandb is None:
        print("[wandb] wandb not installed -- sweep will run without logging", flush=True)
    else:
        print(f"[wandb] no run id found at {run_id_path} -- sweep will run without logging", flush=True)

    ctx = build_eval_context(
        unknown, args.seed, args.camera_eye_offset, args.camera_target_offset,
        tac_mode, args.asset_dir, args.frame_stride, args.max_steps_per_episode,
    )
    model = BCStudent(first_ckpt["prop_dim"], first_ckpt["tac_dim"], action_dim).to(ctx.device)

    last_epoch, last_ckpt_path = ckpts[-1]
    for epoch, path in ckpts:
        ckpt = torch.load(path, map_location=ctx.device)
        model.load_state_dict(ckpt["model"])
        sr, _ = run_rollout_eval(ctx, model, tac_mode, img_size, action_dim, args.episodes_per_ckpt,
                                  tac_vmax=ckpt.get("tac_vmax"))
        print(f"[sweep] epoch={epoch:4d}  quick_SR({args.episodes_per_ckpt} eps)={sr:.3f}", flush=True)
        if run is not None:
            wandb.log({"epoch": epoch, "success_rate": sr})

    ckpt = torch.load(last_ckpt_path, map_location=ctx.device)
    model.load_state_dict(ckpt["model"])
    final_sr, final_results = run_rollout_eval(
        ctx, model, tac_mode, img_size, action_dim, args.final_episodes,
        tac_vmax=ckpt.get("tac_vmax"),
    )
    print(f"[sweep] FINAL epoch={last_epoch}  SR({args.final_episodes} eps)={final_sr:.4f}", flush=True)
    if run is not None:
        wandb.summary["final_success_rate"] = final_sr
        wandb.summary["final_success_episodes"] = args.final_episodes
        wandb.finish()


if __name__ == "__main__":
    main()
