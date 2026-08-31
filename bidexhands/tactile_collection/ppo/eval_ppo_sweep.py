#!/usr/bin/env python3
"""Sweep every saved checkpoint from a PPO training run (model_<iter>.pt) through
a success-rate eval, logging a success-rate-vs-iteration curve to wandb -- mirrors
tactile_collection/bc/sweep_eval_bc_student.py's "quick biased eval per checkpoint,
trustworthy final eval on the last one" pattern, retargeted at bidexhands' own
built-in PPO.eval(num_episodes) (algorithms/rl/ppo/ppo.py:99-120, deterministic
act_inference, reads task.extras['consecutive_successes'], same statistic
compute_hand_reward already accumulates as "Mean episode consecutive_successes"
during training) instead of a custom rollout loop.

Also writes a JSON summary of the full curve next to the checkpoints, and
uploads the final checkpoint + that JSON to HF under
tactile_sr_ablation/bidexhands/<hf_name>/, matching the pen_ponly_seed0 convention
already established there.

Run as (bidexhands's own args after `--`, same split convention as
sweep_eval_bc_student.py):
    cd <DexterousHands root>/bidexhands
    python -m tactile_collection.ppo.eval_ppo_sweep \
        --ckpt_dir logs/ShadowHandPen/ppo/ppo_seed42 \
        --hf_name pen_gttac_seed42 \
        --wandb_run_name pen_gttac_seed42 --wandb_group ShadowHandPen \
        --episodes_per_ckpt 30 --final_episodes 100 \
        -- --task ShadowHandPen --algo ppo \
           --cfg_env cfg/ShadowHandPenProprioGTTac.yaml --num_envs 256 --headless --seed 1234
"""
import os
import re
import sys
import glob
import json
import argparse

# isaacgym must be imported before torch anywhere in the process -- these
# bidexhands imports pull it in, so `import torch` below must stay after them.
from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_sarl import process_sarl
from bidexhands.utils.process_marl import get_AgentIndex

import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def numbered_checkpoints(ckpt_dir):
    """Sorted list of (iteration, path) for every model_<iter>.pt in ckpt_dir."""
    paths = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    out = []
    for p in paths:
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                     help="training run's log_dir (must contain model_<iter>.pt checkpoints)")
    ap.add_argument("--hf_name", required=True,
                     help="destination subfolder under tactile_sr_ablation/bidexhands/ on HF")
    ap.add_argument("--hf_repo", default="qqyang/hora-v4-shadow-tennis")
    ap.add_argument("--episodes_per_ckpt", type=int, default=30,
                     help="quick, biased SR estimate per checkpoint -- for the curve, not the headline number")
    ap.add_argument("--final_episodes", type=int, default=100,
                     help="trustworthy final SR on the last checkpoint")
    ap.add_argument("--wandb_project", default="ego2contact-ppo-ablation")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    args, unknown = ap.parse_known_args()
    # parse_known_args leaves a literal "--" separator in `unknown` if the
    # caller passed one to mark where bidexhands' own args start; bidexhands'
    # own get_args() doesn't handle a bare "--" and would reject it.
    if unknown and unknown[0] == "--":
        unknown = unknown[1:]

    ckpts = numbered_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"no model_<iter>.pt checkpoints found under {args.ckpt_dir}")
    print(f"[sweep] {len(ckpts)} checkpoints: iterations {[it for it, _ in ckpts]}", flush=True)

    # bidexhands' get_args() parses from sys.argv directly (no explicit argv
    # param) -- same pattern used everywhere else in this repo (train.py,
    # verify_sensor_ordering.py), so hand it just the bidexhands-specific tail.
    sys.argv = [sys.argv[0]] + unknown
    set_np_formatting()
    bidex_args = get_args()
    cfg, cfg_train, logdir = load_cfg(bidex_args)
    sim_params = parse_sim_params(bidex_args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    task, env = parse_task(bidex_args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))
    # bidex_args.model_dir defaults to "" -- process_sarl builds a fresh PPO
    # without auto-loading anything; each checkpoint is loaded explicitly
    # below via model.test(path), one env/sim build reused across all of them.
    model = process_sarl(bidex_args, env, cfg_train, logdir)

    run = None
    if wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config={
                "task": bidex_args.task, "cfg_env": bidex_args.cfg_env,
                "ckpt_dir": args.ckpt_dir, "episodes_per_ckpt": args.episodes_per_ckpt,
                "final_episodes": args.final_episodes,
            },
        )
        print(f"[wandb] started run {run.id} ({run.url})", flush=True)
    else:
        print("[wandb] wandb not installed -- sweep will run without logging", flush=True)

    curve = []
    last_it, last_path = ckpts[-1]
    for it, path in ckpts:
        model.test(path)
        sr, _ = model.eval(num_episodes=args.episodes_per_ckpt)
        print(f"[sweep] iteration={it:5d}  quick_SR({args.episodes_per_ckpt} eps)={sr:.2f}%", flush=True)
        curve.append({"iteration": it, "success_rate_pct": sr, "num_episodes": args.episodes_per_ckpt})
        if run is not None:
            wandb.log({"iteration": it, "success_rate_pct": sr})

    model.test(last_path)
    final_sr, _final_successes = model.eval(num_episodes=args.final_episodes)
    print(f"[sweep] FINAL iteration={last_it}  SR({args.final_episodes} eps)={final_sr:.2f}%", flush=True)
    if run is not None:
        wandb.summary["final_success_rate_pct"] = final_sr
        wandb.summary["final_num_episodes"] = args.final_episodes
        wandb.finish()

    result = {
        "task": bidex_args.task,
        "cfg_env": bidex_args.cfg_env,
        "ckpt_dir": args.ckpt_dir,
        "final_checkpoint": os.path.basename(last_path),
        "final_iteration": last_it,
        "final_success_rate_pct": final_sr,
        "final_num_episodes": args.final_episodes,
        "curve": curve,
        "wandb_run_url": run.url if run is not None else None,
    }
    result_path = os.path.join(args.ckpt_dir, "eval_sweep_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sweep] wrote {result_path}", flush=True)

    if HfApi is not None and os.environ.get("HF_TOKEN"):
        api = HfApi(token=os.environ["HF_TOKEN"])
        dest_prefix = f"tactile_sr_ablation/bidexhands/{args.hf_name}"
        for local, name in [(last_path, os.path.basename(last_path)), (result_path, "eval_sweep_result.json")]:
            remote = f"{dest_prefix}/{name}"
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=args.hf_repo, repo_type="model")
            print(f"[hf] uploaded {remote}", flush=True)
    else:
        print("[hf] HfApi unavailable or HF_TOKEN unset -- skipping HF upload", flush=True)

    print("SWEEP_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
