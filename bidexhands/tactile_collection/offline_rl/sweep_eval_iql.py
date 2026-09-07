#!/usr/bin/env python3
"""Stage 2 of the two-stage IQL sweep-eval: sweep every checkpoint EXPORTED by
export_iql_policies.py (policy_step<N>.pt, TorchScript) from an IQL offline-RL
training run through a LIVE rollout success-rate eval, logging a
success-rate-vs-step curve to wandb. Runs ENTIRELY in `bidexhands_isaacgym_py38`
(python 3.8) -- NOT `d3rlpy_offline` (python 3.11) anymore.

Two-stage split (why this script no longer imports d3rlpy): IsaacGym's
precompiled .so bindings require the CPython 3.8 ABI, so `bidexhands` cannot be
installed into `d3rlpy_offline` (confirmed 2026-09-07: `ModuleNotFoundError: No
module named 'bidexhands'` -- not a missing-pip-install problem, the two worlds
genuinely cannot coexist in one process on this cluster). The reverse direction
was also already tried and rejected this session: installing d3rlpy directly
into `bidexhands_isaacgym_py38` silently upgraded gym 0.23.1->0.26.2, breaking
bidexhands' own pinned dependency. So the sweep now runs as two separate
processes/envs that never need both worlds at once:
  Stage 1 (export_iql_policies.py, `d3rlpy_offline`): loads each model_<step>.d3
    via `d3rlpy.load_learnable` and calls `.save_policy(...)` to write a portable
    `policy_step<N>.pt` under `<ckpt_dir>/exported_policies/` -- see that
    script's module docstring for the save_policy() API verification.
  Stage 2 (this script, `bidexhands_isaacgym_py38`): loads each `policy_step<N>.pt`
    via plain `torch.jit.load` (no d3rlpy import at all) and rolls it out live.
Chain them with `--dependency=afterok:<stage1_jobid>` at submit time (see
export_iql_policies.sbatch's header for the exact command).

Why the live-rollout eval exists at all: IQL trains entirely offline -- `.fit()`
(train_iql.py) never touches the IsaacGym env, so it never produces a
success-rate curve the way PPO's own on-policy training does. This script fills
that gap by actually rolling the saved policy out in the live sim, mirroring the
pattern already established for PPO (ppo/eval_ppo_sweep.py) and BC
(bc/sweep_eval_bc_student.py): a quick, biased per-checkpoint eval ->
wandb.log({"step": ..., "success_rate_pct": ...}). Like those two, `--episodes`
is deliberately modest (default 30) -- this is the curve, not a trustworthy
final number.

Env construction reuses bc/rollout_eval_core.py's build_eval_context (the
established task/env/camera/device setup pattern) for the ENV SETUP ONLY -- NOT
its run_rollout_eval, which is built around BC's `model(image, prop, tac)` call
signature and does live RGB camera capture every step, neither of which an IQL
policy needs (IQL trains on flat state only, no vision -- build_mdp_dataset.py
never touches rgb_frames/). A camera still gets created inside build_eval_context
(it always creates one), it's just never used here.

Observation construction reuses build_mdp_dataset.py's discover_prop_keys /
flatten_proprio / flatten_tactile DIRECTLY (imported, not reimplemented) -- any
hand-rolled reimplementation risks a key-order/concat-order mismatch that would
silently feed the trained policy a garbled observation. prop_keys itself is
discovered from ONE sample episode of the SAME manifest build_mdp_dataset.py used
to build the training set (not re-discovered independently against the live
task's attributes) -- this guarantees byte-identical key order to what the
checkpoint was actually trained on, per the task's own preferred approach.
Per-step proprio values are read live off `task` using build_mdp_dataset.py's own
CANDIDATE_PROP_KEYS attribute names (shadow_hand_dof_pos, object_pos, ... plus the
two rigid/dof-state extras) -- the exact same attributes
rollout_tactile_rgb_chest.py's state_names / trace population (lines ~632-636,
~755-769) reads to build the trajectory_env0.npz these prop_keys were originally
discovered from.

p_pred_tac caveat (read before using --arm p_pred_tac): there is no live
predicted-tactile rollout path anywhere in this codebase yet. That arm's tactile
input only exists as an OFFLINE-generated pred_pressure_grids.npz, written by
Ego2Contact's infer_pred_tactile_offline.py as a separate forward pass over
already-recorded frames (see rollout_eval_core.py's own tac_mode="pred" TODO,
which is equally unimplemented for the same reason -- no online
SAM3+WiLoR+DINO+v2-dit forward pass in any step loop). So for --arm p_pred_tac,
this script evaluates the p_pred_tac-TRAINED policy's live rollout using the LIVE
SIMULATOR'S ACTUAL GT TACTILE AS A STAND-IN INPUT (same code path as --arm
p_gt_tac, with a loud warning printed once at startup). This measures how well
that policy transfers to GT tactile at test time -- it is NOT identical to its
training distribution (which saw whatever infer_pred_tactile_offline.py's
predictions looked like) and the number should be reported/compared with that
caveat attached, never as a like-for-like p_gt_tac vs p_pred_tac comparison.

Success determination: task.extras['successes'] (a per-env sticky 0/1 "did this
episode ever hit the goal" flag, reset to 0 in each task's own reset_idx) -- NOT
task.extras['consecutive_successes'], which crashes on 3 of 4 bidexhands tasks due
to an inconsistent tensor size (see algorithms/rl/ppo/ppo.py's PPO.eval()
docstring for the full explanation of that bug and its fix). env.step()'s
returned `infos` IS `task.extras` (same dict object -- see
tasks/hand_base/vec_task.py lines 131/151: `return ..., self.task.extras`), so
reading task.extras directly after env.step() is equivalent to reading infos;
this script reads task.extras directly to match the fixed PPO pattern literally.

Run as (bidexhands's own args after `--`, same convention as the other sweep
scripts in this package; requires export_iql_policies.py to have already
written --ckpt_dir/exported_policies/policy_step<N>.pt -- Stage 1, run first
in the d3rlpy_offline env):
    cd <DexterousHands root>/bidexhands
    python -m tactile_collection.offline_rl.sweep_eval_iql \
        --ckpt_dir runs/offline_rl/pen_p_gt_tac/iql_p_gt_tac/iql_p_gt_tac_20260907002040 \
        --arm p_gt_tac --manifest shadow_hand_pen_manifest.json \
        --episodes 30 --wandb_project ego2contact-iql-ablation --wandb_name pen_p_gt_tac_sweep \
        -- --task ShadowHandPen --algo ppo --cfg_env cfg/ShadowHandPenProprioGTTac.yaml \
           --num_envs 1 --headless --seed 1234

Not yet run anywhere -- no cluster/IsaacGym access from this machine (same
caveat as build_mdp_dataset.py/train_iql.py's own module docstrings).

`model_<step>.d3` naming provenance (still relevant for matching a
`policy_step<N>.pt` back to its source checkpoint) -- VERIFIED against d3rlpy's
actual GitHub source at tag v2.8.1: `d3rlpy/logging/file_adapter.py`
`FileAdapter.save_model(epoch, algo)` does
`model_path = os.path.join(logdir, f"model_{epoch}.d3"); algo.save(model_path)`,
called with `epoch=total_step` by d3rlpy's own training loop -- hence "step" not
"epoch" in the filename. export_iql_policies.py reads that same step number back
out of each `model_<step>.d3` and writes `policy_step<step>.pt`, so the two
numberings line up by construction.

TorchScript policy-loading -- see export_iql_policies.py's module docstring for
the full save_policy() API verification (against d3rlpy's real GitHub source at
v2.8.1, same standard as everything else in this file). Summary of what matters
here: `torch.jit.load(path, map_location=device)` returns a module callable as
`action = policy(obs_tensor)` with `obs_tensor: (N, obs_dim) float32 ->
action: (N, action_dim) float32` -- this IS the deterministic/greedy action
(`predict_best_action`, the same call d3rlpy's own `.predict()` makes
internally). train_iql.py configures `observation_scaler=
StandardObservationScaler()` (added 2026-09-07, see its own comment for why) --
that z-score transform is baked directly into this traced module by
`save_policy()` itself, so this script still just calls `policy(obs_tensor)`
with the SAME raw (unnormalized) observation build_live_observation() already
constructs; no extra normalization step belongs here. No numpy round-trip is needed here (unlike
the old `algo.predict(numpy_array)` d3rlpy call this replaces) -- build a tensor
once and feed it straight to the traced module.
"""
import os
import re
import glob
import json
import argparse

import numpy as np

# isaacgym must be imported before torch anywhere in the process -- this pulls
# it in via bidexhands.utils.config (rollout_eval_core -> that chain), so
# `import torch` below must stay after this import, not before. (No d3rlpy
# import anywhere in this file anymore -- see module docstring's two-stage split.)
from tactile_collection.bc.rollout_eval_core import build_eval_context
from tactile_collection.rollout_tactile_rgb_chest import as_numpy, env0

import torch

from tactile_collection.offline_rl.build_mdp_dataset import (
    discover_prop_keys, flatten_proprio, flatten_tactile, episode_dir,
)

try:
    import wandb
except ImportError:
    wandb = None


def step_checkpoints(ckpt_dir, exported_subdir):
    """Sorted list of (step, path) for every policy_step<N>.pt under
    ckpt_dir/exported_subdir -- these are written by export_iql_policies.py
    (Stage 1), one per model_<step>.d3 it found. If this comes back empty,
    Stage 1 either hasn't run yet or hasn't finished -- run/wait on
    export_iql_policies.sbatch first (see this script's module docstring)."""
    paths = glob.glob(os.path.join(ckpt_dir, exported_subdir, "policy_step*.pt"))
    out = []
    for p in paths:
        m = re.search(r"policy_step(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def prop_keys_from_manifest(manifest_path):
    """Discover prop_keys from ONE sample episode of `manifest_path`'s 'train'
    list, using build_mdp_dataset.py's OWN discover_prop_keys against that
    episode's actual trajectory_env0.npz -- identical to what
    build_dataset() itself does, so this is guaranteed to match the exact key
    set/order the checkpoint being evaluated was trained on (as opposed to
    independently re-discovering against the live task and hoping the two
    agree)."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest["train"]
    if not entries:
        raise ValueError(f"manifest '{manifest_path}' has an empty 'train' list")
    sample_dir = episode_dir(entries[0])
    with np.load(os.path.join(sample_dir, "trajectory_env0.npz")) as sample_traj:
        prop_keys = discover_prop_keys(sample_traj.files)
    if not prop_keys:
        raise RuntimeError(
            f"none of the candidate proprio keys were found in "
            f"{sample_dir}/trajectory_env0.npz (has: {sample_traj.files}) -- "
            f"refusing to evaluate a policy against zero-dim observations"
        )
    print(f"[discover] prop_keys from manifest sample {sample_dir}: {prop_keys}", flush=True)
    return prop_keys


def object_actor_indices(task):
    """object_rigid_body_state/object_dof_state (two of build_mdp_dataset.py's
    CANDIDATE_PROP_KEYS) are per-BODY/per-DOF slices of the object actor's rigid
    body / dof state tensors, not a bare task attribute -- reconstruct the same
    index lists rollout_tactile_rgb_chest.py's main() computes once at startup
    (lines ~592-605) so a live lookup matches what got saved into
    trajectory_env0.npz exactly."""
    from isaacgym import gymapi
    object_actor = task.gym.find_actor_handle(task.envs[0], "object")
    if object_actor < 0:
        raise RuntimeError(
            "manifest prop_keys references object_rigid_body_state/object_dof_state "
            "but this task has no 'object' actor -- task/config mismatch between the "
            "manifest's source episodes and this rollout's --task/--cfg_env"
        )
    object_body_names = task.gym.get_actor_rigid_body_names(task.envs[0], object_actor)
    object_body_indices = [
        task.gym.get_actor_rigid_body_index(task.envs[0], object_actor, i, gymapi.DOMAIN_ENV)
        for i in range(len(object_body_names))
    ]
    object_dof_indices = [
        task.gym.get_actor_dof_index(task.envs[0], object_actor, i, gymapi.DOMAIN_ENV)
        for i in range(task.gym.get_actor_dof_count(task.envs[0], object_actor))
    ]
    return object_body_indices, object_dof_indices


def live_prop_row(task, key, num_envs, object_body_indices, object_dof_indices):
    """Fetch ONE prop_keys entry's current-step value from the live task,
    mirroring rollout_tactile_rgb_chest.py's per-step trace population (lines
    ~755-769): same attributes, same env0() reduction to a single-env row, same
    object-body/dof indexing for the two non-bare-attribute keys. Returns a flat
    per-env array (no time axis -- callers wrap it with [None] before handing it
    to flatten_proprio, which expects a (T, ...) leading axis).

    Fails loudly (never silently drops/pads a key) if the live task doesn't
    actually have what the manifest says it should -- a mismatch here means the
    manifest's source episodes were collected under a different --task/--cfg_env
    than this rollout is using, which is a bug to fix, not paper over.
    """
    if key == "object_rigid_body_state":
        rigid_states = env0(getattr(task, "rigid_body_states", None), num_envs)
        if rigid_states is None or not object_body_indices:
            raise RuntimeError(
                "prop_keys includes 'object_rigid_body_state' but the live task has no "
                "rigid_body_states tensor or no object body indices"
            )
        return np.asarray(rigid_states)[object_body_indices].astype(np.float32)
    if key == "object_dof_state":
        dof_states = env0(getattr(task, "dof_state", None), num_envs)
        if dof_states is None or not object_dof_indices:
            raise RuntimeError(
                "prop_keys includes 'object_dof_state' but the live task has no dof_state "
                "tensor or no object dof indices"
            )
        return np.asarray(dof_states)[object_dof_indices].astype(np.float32)
    if not hasattr(task, key):
        raise RuntimeError(
            f"manifest prop_keys includes '{key}' but the live task object has no such "
            f"attribute -- task/config mismatch between the manifest's source episodes "
            f"and this rollout's --task/--cfg_env; fix the mismatch, don't silently drop it"
        )
    value = env0(getattr(task, key), num_envs)
    if value is None:
        raise RuntimeError(
            f"task.{key} did not reduce to a per-env row of shape (num_envs, ...) -- "
            f"cannot build a matching observation"
        )
    return np.asarray(value, dtype=np.float32)


def build_live_observation(ctx, arm, prop_keys, object_body_indices, object_dof_indices):
    """One live-rollout step's observation vector, laid out IDENTICALLY to
    build_mdp_dataset.py's build_episode_arrays: flatten_proprio(prop_keys) first,
    then (for the two tactile arms) flatten_tactile concatenated after it -- same
    order, same functions, called on single-timestep arrays reshaped to carry a
    length-1 time axis so flatten_proprio/flatten_tactile's own (T, ...) handling
    applies unchanged (no hand-copied flatten/concat math here)."""
    task = ctx.task
    prop_step = {
        k: live_prop_row(task, k, ctx.env.num_envs, object_body_indices, object_dof_indices)[None]
        for k in prop_keys
    }
    prop = flatten_proprio(prop_step, prop_keys)  # (1, prop_dim)

    if arm not in ("p_gt_tac", "p_pred_tac"):
        return prop.astype(np.float32)

    # Both tactile arms read LIVE GT tactile here -- see module docstring's
    # p_pred_tac caveat for why p_pred_tac has no live predicted-tactile path yet.
    task.gym.refresh_net_contact_force_tensor(task.sim)
    env_forces = as_numpy(ctx.net_contact_forces[0])
    left_pa, _, _ = ctx.left_mapper.project_net_forces(env_forces)
    right_pa, _, _ = ctx.right_mapper.project_net_forces(env_forces)
    # NOTE: deliberately NOT dividing by tac_vmax here, unlike
    # rollout_eval_core.py's BC branch. flatten_tactile's contract is the RAW
    # pressure_grids.npz scale (only nan_to_num, no rescaling) -- that's what
    # build_mdp_dataset.py actually built the IQL training observations from,
    # and pressure_grids.npz's own "left_pressure_grid"/"right_pressure_grid"
    # are themselves just this same project_net_forces(...)[0] call's output
    # (see rollout_tactile_rgb_chest.py main(), lines ~675-676/683-684). Adding
    # BC's vmax-normalization here would silently retrain-vs-eval-mismatch the
    # tactile scale for this arm.
    press_step = {
        "left_pressure_grid": left_pa[None],
        "right_pressure_grid": right_pa[None],
    }
    tac = flatten_tactile(press_step)  # (1, 882)
    return np.concatenate([prop, tac], axis=1).astype(np.float32)


def run_iql_rollout(ctx, policy, arm, prop_keys, episodes, object_body_indices, object_dof_indices):
    """Roll the exported TorchScript `policy` out for `episodes` fresh episodes
    in ctx.env, querying and acting on the policy EVERY step (no
    frame_stride/zero-order-hold cadence -- unlike BC's
    rollout_eval_core.run_rollout_eval, IQL was trained on every per-step
    transition, not RGB-frame-subsampled ones, so eval-time cadence should match
    that: full density). Returns a list of bool successes, one per completed
    episode."""
    task, env, device = ctx.task, ctx.env, ctx.device
    action_dim = task.num_actions

    obs = env.reset()
    successes = []
    episode_steps = 0

    while len(successes) < episodes:
        obs_row = build_live_observation(ctx, arm, prop_keys, object_body_indices, object_dof_indices)
        obs_t = torch.as_tensor(obs_row, dtype=torch.float32, device=device)  # (1, obs_dim)

        with torch.no_grad():
            # policy() IS the greedy/deterministic action (predict_best_action) --
            # see export_iql_policies.py's module docstring for the save_policy()
            # verification this call signature is based on. No numpy round-trip
            # needed, unlike the old d3rlpy `algo.predict(numpy_array)` this replaces.
            action_t = policy(obs_t).reshape(1, action_dim)
            next_obs, rew, done, infos = env.step(action_t)
            obs.copy_(next_obs)

        episode_steps += 1
        done_np = as_numpy(done).astype(bool)
        timed_out = episode_steps >= ctx.max_steps_per_episode
        if bool(done_np[0]) or timed_out:
            if "successes" not in task.extras:
                raise RuntimeError(
                    "task.extras has no 'successes' key -- cannot determine episode "
                    "success (see module docstring for why NOT to fall back to "
                    "'consecutive_successes')"
                )
            success_val = as_numpy(task.extras["successes"])[0]
            successes.append(bool(success_val > 0))
            episode_steps = 0
            obs = env.reset()

    return successes[:episodes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                     help="an offline_rl_iql.sbatch training run's OUT_DIR/iql_<arm> dir -- "
                          "must contain <ckpt_dir>/<exported_subdir>/policy_step<N>.pt files "
                          "already written by export_iql_policies.py (Stage 1), NOT bare "
                          "model_<step>.d3 checkpoints (this script no longer reads those directly)")
    ap.add_argument("--exported_subdir", default="exported_policies",
                     help="subdir under --ckpt_dir holding policy_step<N>.pt -- must match "
                          "export_iql_policies.py's --out_subdir from Stage 1 (default matches)")
    ap.add_argument("--arm", choices=["p_only", "p_gt_tac", "p_pred_tac"], required=True)
    ap.add_argument("--manifest", required=True,
                     help="the SAME manifest build_mdp_dataset.py used to build this "
                          "checkpoint's training dataset -- prop_keys is discovered from it")
    ap.add_argument("--episodes", type=int, default=30,
                     help="quick, biased SR estimate per checkpoint -- for the curve, "
                          "not a trustworthy final number (mirrors eval_ppo_sweep.py's "
                          "episodes_per_ckpt / sweep_eval_bc_student.py's own tradeoff)")
    ap.add_argument("--max_steps_per_episode", type=int, default=500)
    ap.add_argument("--frame_stride", type=int, default=1,
                     help="unused for decision cadence (IQL acts every step, see "
                          "run_iql_rollout's docstring) -- only forwarded to "
                          "build_eval_context, which always creates an (unused here) camera")
    ap.add_argument("--camera_eye_offset", default="0.32,0.0,0.80")
    ap.add_argument("--camera_target_offset", default="0.0,0.0,0.08")
    ap.add_argument("--asset_dir", default="tactile_collection/assets")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--wandb_project", default="ego2contact-iql-ablation")
    ap.add_argument("--wandb_name", default=None)
    args, unknown = ap.parse_known_args()
    # parse_known_args leaves a literal "--" separator in `unknown` if the caller
    # passed one to mark where bidexhands' own args start; bidexhands' own
    # get_args() doesn't handle a bare "--" and would reject it.
    if unknown and unknown[0] == "--":
        unknown = unknown[1:]

    ckpts = step_checkpoints(args.ckpt_dir, args.exported_subdir)
    if not ckpts:
        raise RuntimeError(
            f"no policy_step<N>.pt files found under {args.ckpt_dir}/{args.exported_subdir} -- "
            f"run export_iql_policies.py (Stage 1, d3rlpy_offline env) on this ckpt_dir first"
        )
    print(f"[sweep] {len(ckpts)} checkpoints: steps {[s for s, _ in ckpts]}", flush=True)

    prop_keys = prop_keys_from_manifest(args.manifest)

    if args.arm == "p_pred_tac":
        print(
            "[WARNING] --arm p_pred_tac: evaluating a Pred-Tac-trained policy against "
            "LIVE GT tactile, not live predicted tactile, since no live pred-tac rollout "
            "path exists yet -- this measures how well the policy transfers to GT "
            "tactile at test time, NOT identical to its training distribution. See this "
            "script's module docstring.",
            flush=True,
        )

    # tac_mode="gt" builds the taxel mappers + contact-force tensor for BOTH
    # tactile arms (p_gt_tac and p_pred_tac-evaluated-against-GT, per the caveat
    # above); p_only needs neither, so pass anything else to skip that setup.
    tac_mode = "gt" if args.arm in ("p_gt_tac", "p_pred_tac") else "none"
    ctx = build_eval_context(
        unknown, args.seed, args.camera_eye_offset, args.camera_target_offset,
        tac_mode, args.asset_dir, args.frame_stride, args.max_steps_per_episode,
    )

    object_body_indices, object_dof_indices = [], []
    if "object_rigid_body_state" in prop_keys or "object_dof_state" in prop_keys:
        object_body_indices, object_dof_indices = object_actor_indices(ctx.task)

    run = None
    if wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "ckpt_dir": args.ckpt_dir, "arm": args.arm, "manifest": args.manifest,
                "episodes": args.episodes, "prop_keys": prop_keys,
            },
        )
        print(f"[wandb] started run {run.id} ({run.url})", flush=True)
    else:
        print("[wandb] wandb not installed -- sweep will run without logging", flush=True)

    curve = []
    for step, path in ckpts:
        # map_location=ctx.device: the trace is device-tagged at export time to
        # whatever --device export_iql_policies.py ran with (default cpu:0), which
        # will usually differ from this live rollout's cuda device -- see that
        # script's module docstring for why the export device doesn't need to match.
        policy = torch.jit.load(path, map_location=ctx.device)
        policy.eval()
        successes = run_iql_rollout(
            ctx, policy, args.arm, prop_keys, args.episodes, object_body_indices, object_dof_indices,
        )
        sr = 100.0 * sum(successes) / len(successes) if successes else float("nan")
        print(f"[sweep] step={step:7d}  quick_SR({args.episodes} eps)={sr:.2f}%", flush=True)
        curve.append({"step": step, "success_rate_pct": sr, "num_episodes": args.episodes})
        if run is not None:
            wandb.log({"step": step, "success_rate_pct": sr})

    if run is not None:
        last = curve[-1]
        wandb.summary["final_success_rate_pct"] = last["success_rate_pct"]
        wandb.summary["final_step"] = last["step"]
        wandb.finish()

    result_path = os.path.join(args.ckpt_dir, "eval_sweep_result.json")
    with open(result_path, "w") as f:
        json.dump({
            "ckpt_dir": args.ckpt_dir, "arm": args.arm, "manifest": args.manifest,
            "prop_keys": prop_keys, "episodes_per_ckpt": args.episodes, "curve": curve,
            "wandb_run_url": run.url if run is not None else None,
        }, f, indent=2)
    print(f"[sweep] wrote {result_path}", flush=True)
    print("SWEEP_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
