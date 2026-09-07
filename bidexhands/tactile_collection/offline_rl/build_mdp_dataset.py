#!/usr/bin/env python3
"""Build a d3rlpy MDPDataset (flat concatenated transitions) from an episode
manifest -- the offline-RL counterpart to train_bc_student.py's data loading,
but using EVERY per-step transition (not just RGB-frame steps: BC only
supervises at frames because it needs the rendered image; IQL needs full
transition density for its Q/V targets, so the RGB-frame subsampling in
train_bc_student.py's _episode_frame_rows is deliberately NOT reused here).

Three arms (--arm):
  p_only     : proprioception only                                     -> obs
  p_gt_tac   : proprioception + GT tactile (pressure_grids.npz)         -> obs
  p_pred_tac : proprioception + PREDICTED tactile (pred_pressure_grids.npz,
               written offline by Ego2Contact's infer_pred_tactile_offline.py)
               -> obs
(no vision -- offline RL here trains on state, not pixels, unlike
train_bc_student.py's BC arms which also read rgb_frames/.)

Proprio keys are NOT hardcoded: the exact set of per-step state arrays a
trajectory_env0.npz carries is task-dependent (see rollout_tactile_rgb_chest.py
lines ~632-636 and ~755-759, which only append a key if `hasattr(task, name)`
for that task). This script discovers the present subset from one sample
episode, prints exactly what it found, and then requires every other episode
in the manifest to carry that SAME key set -- silently falling back to a
narrower per-episode subset would make different episodes' observation
vectors mean different things at the same index, which is worse than failing
loudly.

Usage:
    python build_mdp_dataset.py --manifest shadow_hand_pen_manifest.json \
        --arm p_gt_tac --out shadow_hand_pen_p_gt_tac.h5

Not yet run anywhere -- no access to the actual cluster/episode data from
this machine. The pure-numpy flatten/concat helpers below are covered by
test_flatten.py (synthetic in-memory data, no d3rlpy import needed); the
d3rlpy-dependent save path (build_dataset/main) is unverified against a real
install -- see the MDPDataset construction/serialization note below.
"""
import os
import json
import argparse

import numpy as np


# Every per-step state array rollout_tactile_rgb_chest.py *might* attach to a
# trajectory_env0.npz, depending on which attributes the task object exposes
# (state_names in rollout_tactile_rgb_chest.py) plus the two rigid/dof-state
# extras it appends conditionally (object_rigid_body_state, object_dof_state).
# This is a CANDIDATE list only -- discover_prop_keys() intersects it against
# what a real file actually has; never assume all of these are present.
CANDIDATE_PROP_KEYS = [
    "shadow_hand_dof_pos", "shadow_hand_another_dof_pos", "dof_pos", "cur_targets",
    "object_pose", "object_pos", "object_rot", "goal_pose", "goal_pos", "goal_rot",
    "goal_states", "object_rigid_body_state", "object_dof_state",
]


def episode_dir(entry):
    return os.path.join(entry["shard_dir"], "successful_episodes", entry["episode"])


def discover_prop_keys(available_keys, candidate_keys=CANDIDATE_PROP_KEYS):
    """Returns the candidate keys present in `available_keys` (e.g. an npz's
    .files), in a fixed order (candidate-list order, not sorted-alphabetical)
    so the resulting observation layout is deterministic and stable across
    runs/episodes."""
    return [k for k in candidate_keys if k in available_keys]


def flatten_proprio(traj, prop_keys):
    """traj: dict-like (npz or plain dict) with each prop_keys[i] -> array of
    shape (T, ...). Returns (T, sum_i prod(shape_i)) float32."""
    t = None
    cols = []
    for k in prop_keys:
        arr = np.asarray(traj[k], dtype=np.float32)
        if t is None:
            t = arr.shape[0]
        elif arr.shape[0] != t:
            raise ValueError(f"prop key '{k}' has T={arr.shape[0]}, expected {t}")
        cols.append(arr.reshape(t, -1))
    if not cols:
        return np.zeros((t or 0, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1)


def flatten_tactile(press):
    """press: dict-like with left_pressure_grid/right_pressure_grid, each
    (T,21,21) -> concatenated+flattened (T, 882) float32. NaN (unmapped/
    non-sensor cells, distinct from a real zero reading -- see
    train_bc_student.py's compute_tac_vmax docstring) is zeroed, same as the
    BC pipeline's convention, so the two pipelines treat non-sensor cells
    identically."""
    left = np.nan_to_num(np.asarray(press["left_pressure_grid"], dtype=np.float32), nan=0.0)
    right = np.nan_to_num(np.asarray(press["right_pressure_grid"], dtype=np.float32), nan=0.0)
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"left/right pressure grids disagree on T: {left.shape[0]} vs {right.shape[0]}")
    t = left.shape[0]
    return np.concatenate([left.reshape(t, -1), right.reshape(t, -1)], axis=1)


def build_episode_arrays(traj, press, arm, prop_keys):
    """Returns (obs, actions, rewards, terminals) for one episode, using
    EVERY per-step transition (no RGB-frame subsampling).

    terminal = `done` as-is. This conflates true task-termination with
    timeout/max-step truncation (bidexhands's `done` doesn't distinguish
    the two) -- a known simplification, not fixed here; see module docstring
    of train_iql.py for how this could bootstrap incorrectly across a
    truncated (non-terminal) episode boundary if it mattered for our data,
    which it mostly does not since manifests only contain SUCCESSFUL
    episodes (done=True at genuine success in the overwhelming case).
    """
    assert arm in ("p_only", "p_gt_tac", "p_pred_tac")

    prop = flatten_proprio(traj, prop_keys)
    t = prop.shape[0]

    if arm in ("p_gt_tac", "p_pred_tac"):
        tac = flatten_tactile(press)
        if tac.shape[0] != t:
            raise ValueError(f"tactile T={tac.shape[0]} != proprio T={t}")
        obs = np.concatenate([prop, tac], axis=1)
    else:
        obs = prop

    actions = np.asarray(traj["actions"], dtype=np.float32)
    rewards = np.asarray(traj["reward"], dtype=np.float32).reshape(-1)
    terminals = np.asarray(traj["done"], dtype=np.bool_).reshape(-1)

    for name, arr in (("actions", actions), ("reward", rewards), ("done", terminals)):
        if arr.shape[0] != t:
            raise ValueError(f"'{name}' has T={arr.shape[0]}, expected {t} (from proprio)")

    return obs, actions, rewards, terminals


def concat_episode_transitions(episode_arrays):
    """episode_arrays: list of (obs, actions, rewards, terminals) tuples, one
    per episode, IN MANIFEST ORDER. Concatenates into flat arrays suitable
    for d3rlpy's flat-array MDPDataset constructor (episode boundaries are
    implied by `terminals`, not passed separately).

    Every stored episode is (by construction -- manifests only include
    entries under successful_episodes/) supposed to end in a genuine
    success/termination, so each episode's LAST row is forced to
    terminals=True before concatenation regardless of the raw `done` value
    there -- this is purely so the flat buffer's episode boundaries land in
    the right place (a false boundary would make d3rlpy bootstrap the value
    of one episode's last transition off the NEXT episode's unrelated first
    observation). This does not touch any INTERIOR row's done value, and
    does not otherwise resolve the true-termination-vs-timeout ambiguity
    noted in build_episode_arrays -- only guarantees a boundary exists where
    one structurally must.
    """
    obs_all, act_all, rew_all, term_all = [], [], [], []
    n_forced = 0
    for obs, actions, rewards, terminals in episode_arrays:
        terminals = terminals.copy()
        if not terminals[-1]:
            n_forced += 1
            terminals[-1] = True
        obs_all.append(obs)
        act_all.append(actions)
        rew_all.append(rewards)
        term_all.append(terminals)
    if n_forced:
        print(f"[warn] forced terminals=True on the last row of {n_forced}/{len(episode_arrays)} "
              f"episodes whose stored `done` was False at the final step (episode-boundary "
              f"correctness only -- see concat_episode_transitions docstring)", flush=True)
    return (
        np.concatenate(obs_all, axis=0),
        np.concatenate(act_all, axis=0),
        np.concatenate(rew_all, axis=0),
        np.concatenate(term_all, axis=0),
    )


# ---------------------------------------------------------------------------
# d3rlpy-dependent I/O -- kept out of the pure-numpy helpers above so
# test_flatten.py can unit-test those without requiring d3rlpy to be
# installed (it is not installed on this machine; this whole section is
# unverified against a real d3rlpy install, see the dump/load notes below).
# ---------------------------------------------------------------------------

def _dump_dataset(dataset, out_path):
    """d3rlpy v2's documented MDPDataset save convention is `dataset.dump(f)`
    with `f` an OPEN BINARY file handle (not a bare path string) -- e.g.
    `with open(path, "w+b") as f: dataset.dump(f)`. Some released versions
    instead accept a bare path directly. Try the file-handle form first (the
    documented v2 convention referenced in the task) and fall back to a bare
    path so this still works if the pinned version differs -- FLAG: confirm
    against `python -c "import d3rlpy; print(d3rlpy.__version__)"` +
    `help(d3rlpy.dataset.MDPDataset.dump)` on the actual cluster env before
    trusting either branch blindly."""
    try:
        with open(out_path, "w+b") as f:
            dataset.dump(f)
    except TypeError:
        dataset.dump(out_path)


def build_dataset(manifest_path, arm, out_path):
    import d3rlpy
    from d3rlpy.dataset import MDPDataset

    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest["train"]
    if not entries:
        raise ValueError(f"manifest '{manifest_path}' has an empty 'train' list")

    # --- discover proprio keys from ONE sample episode, then require every
    # other episode to match exactly (see module docstring). ---
    sample_dir = episode_dir(entries[0])
    with np.load(os.path.join(sample_dir, "trajectory_env0.npz")) as sample_traj:
        prop_keys = discover_prop_keys(sample_traj.files)
        print(f"[discover] proprio keys found in {sample_dir}:", flush=True)
        for k in prop_keys:
            shape = np.asarray(sample_traj[k]).shape
            print(f"    {k}: per-step shape {shape[1:]} (T={shape[0]})", flush=True)
        if not prop_keys:
            raise RuntimeError(
                f"none of the candidate proprio keys {CANDIDATE_PROP_KEYS} were found in "
                f"{sample_dir}/trajectory_env0.npz (has: {sample_traj.files}) -- refusing to "
                f"build a dataset with zero-dim observations"
            )

    episode_arrays = []
    for e in entries:
        ep_dir = episode_dir(e)
        traj_path = os.path.join(ep_dir, "trajectory_env0.npz")
        with np.load(traj_path) as traj:
            found = discover_prop_keys(traj.files)
            if found != prop_keys:
                raise RuntimeError(
                    f"inconsistent proprio key set: {sample_dir} has {prop_keys}, but "
                    f"{ep_dir} has {found} -- manifest '{manifest_path}' mixes episodes "
                    f"collected under different task configs; fix the manifest, don't "
                    f"silently intersect/pad around this"
                )
            tac_file = {"p_gt_tac": "pressure_grids.npz",
                        "p_pred_tac": "pred_pressure_grids.npz"}.get(arm)
            press_path = os.path.join(ep_dir, tac_file) if tac_file else None
            press = np.load(press_path) if press_path else None
            try:
                episode_arrays.append(build_episode_arrays(traj, press, arm, prop_keys))
            finally:
                if press is not None:
                    press.close()

    obs, actions, rewards, terminals = concat_episode_transitions(episode_arrays)

    dataset = MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals.astype(np.float32),
    )
    _dump_dataset(dataset, out_path)

    # --- summary stats ---
    episode_returns = [float(np.sum(r)) for (_, _, r, _) in episode_arrays]
    print(f"[summary] arm={arm}", flush=True)
    print(f"[summary] n_episodes={len(episode_arrays)}  n_transitions={obs.shape[0]}", flush=True)
    print(f"[summary] obs_dim={obs.shape[1]}  action_dim={actions.shape[1] if actions.ndim > 1 else 1}", flush=True)
    print(f"[summary] mean_episode_return={np.mean(episode_returns):.4f} "
          f"(min={np.min(episode_returns):.4f}, max={np.max(episode_returns):.4f})", flush=True)
    print(f"[summary] action_range=[{actions.min():.4f}, {actions.max():.4f}]", flush=True)
    print(f"[dataset] wrote {out_path}", flush=True)
    print(f"[dataset] d3rlpy version {d3rlpy.__version__}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                     help="episode manifest JSON from generate_episode_manifest.py "
                          "(uses manifest['train'] -- val is held out for eval, not "
                          "part of the offline-RL training buffer)")
    ap.add_argument("--arm", choices=["p_only", "p_gt_tac", "p_pred_tac"], required=True)
    ap.add_argument("--out", required=True, help="output .h5 dataset path")
    args = ap.parse_args()
    build_dataset(args.manifest, args.arm, args.out)


if __name__ == "__main__":
    main()
