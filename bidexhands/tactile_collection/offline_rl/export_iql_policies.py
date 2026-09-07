#!/usr/bin/env python3
"""Stage 1 of the two-stage IQL sweep-eval: export every model_<step>.d3
d3rlpy checkpoint under --ckpt_dir to a portable TorchScript .pt via
d3rlpy's own `.save_policy()`. Runs ENTIRELY in the `d3rlpy_offline` conda
env (python 3.11) -- no IsaacGym import anywhere, CPU-only, no sim.

Why this script exists: `d3rlpy_offline` (needed for `d3rlpy.load_learnable`)
and `bidexhands_isaacgym_py38` (needed for the live rollout -- IsaacGym's
precompiled .so requires the CPython 3.8 ABI) cannot coexist in one Python
process on this cluster. Confirmed 2026-09-07: `ModuleNotFoundError: No
module named 'bidexhands'` when trying to import bidexhands from
d3rlpy_offline's python -- not a missing-pip-install problem, IsaacGym's
C extensions simply cannot build/import under 3.11. Installing d3rlpy
directly into bidexhands_isaacgym_py38 was ALSO already tried this session
and is not an option either: it silently upgraded gym 0.23.1->0.26.2,
breaking bidexhands' own pinned dependency (had to
`pip uninstall d3rlpy gymnasium && pip install gym==0.23.1` to revert).

So the sweep splits into two processes that never need both worlds at
once: this script (Stage 1, d3rlpy_offline, exports policies) and
sweep_eval_iql.py (Stage 2, bidexhands_isaacgym_py38, rolls the exported
policies out live -- see that script's module docstring for the loader
side of this split).

save_policy() API -- VERIFIED against d3rlpy's actual GitHub source at tag
v2.8.1 (this project's pinned cluster version, same verification standard
already applied elsewhere in this package to `load_learnable`/
`FileAdapter.save_model`/`QLearningAlgoBase.predict` -- see
sweep_eval_iql.py's own module docstring):
  `d3rlpy/algos/qlearning/base.py` `QLearningAlgoBase.save_policy(fname)`:
  torch.jit.traces a closure `_func(*x) -> action` that calls
  `self._impl.predict_best_action(observation)` -- the SAME greedy/
  deterministic call `.predict()` makes internally -- and, ONLY if the
  algo's config carries an `observation_scaler`/`action_scaler`, wraps it
  with `transform(...)`/`reverse_transform(...)`. train_iql.py now DOES
  configure `observation_scaler=StandardObservationScaler()` (added
  2026-09-07 -- unnormalized raw-Pa tactile vs radian/meter-scale proprio
  was causing p_gt_tac's per-task-severity-correlated failure/collapse, see
  train_iql.py's own comment), so `save_policy`'s exported graph now
  automatically includes the input z-score transform baked in -- Stage 2
  never needs to know or replicate the scaler itself, it just calls the
  traced module and gets a correctly-normalized forward pass for free.
  `action_scaler` stays unset (action space is already the model's own
  bounded joint-target scale, no rescaling needed there).
  For a flat (non-tuple) observation shape -- true here, `build_mdp_dataset.py`
  writes one concatenated proprio(+tactile) vector per step, not a
  dict/tuple observation -- `save_policy` traces the closure against a
  SINGLE dummy tensor of shape (1, obs_dim), so the saved TorchScript
  module's forward signature is exactly:
      action = torch.jit.load(path)(obs_tensor)  # obs_tensor: (N, obs_dim) float32
                                                   # action:     (N, action_dim) float32
  The trace is device-tagged at export time (`dummy_x` is built with
  `device=self._device`); Stage 2 must `torch.jit.load(path,
  map_location=<rollout device>)` since the export device here (CPU, see
  --device below) will differ from the live rollout's cuda device.

Usage:
    python export_iql_policies.py \\
        --ckpt_dir runs/offline_rl/pen_p_gt_tac/iql_p_gt_tac/iql_p_gt_tac_20260907002040

Not yet run anywhere -- no cluster/d3rlpy access from this machine (same
caveat as every other script in this package's module docstrings). The
save_policy() source above was read from d3rlpy's real GitHub source at
v2.8.1, not guessed, but the resulting .pt has NOT been round-tripped
through a real d3rlpy-trained checkpoint here (no d3rlpy install on this
machine) -- only a throwaway hand-built torch.jit.trace/save/load was used
to confirm the TorchScript round-trip mechanics work at all (see this
repo's own commit for that check). Watch the FIRST real export on VISION
print a sane obs_dim/action_dim and a `policy(dummy_obs)` sanity shape
before trusting the whole sweep.
"""
import os
import re
import glob
import argparse


def step_checkpoints(ckpt_dir):
    """Sorted list of (step, path) for every model_<step>.d3 in ckpt_dir.
    Deliberately duplicated (not imported) from sweep_eval_iql.py's own
    identical helper -- that module now lives entirely in the isaacgym-only
    Stage 2 world (importing it here would drag in
    tactile_collection.bc.rollout_eval_core's isaacgym-before-torch import
    chain, which d3rlpy_offline's python 3.11 cannot satisfy)."""
    paths = glob.glob(os.path.join(ckpt_dir, "model_*.d3"))
    out = []
    for p in paths:
        m = re.search(r"model_(\d+)\.d3$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                     help="an offline_rl_iql.sbatch OUT_DIR/iql_<arm> dir (model_<step>.d3 files)")
    ap.add_argument("--out_subdir", default="exported_policies",
                     help="written under --ckpt_dir as policy_step<N>.pt; sweep_eval_iql.py's "
                          "Stage 2 reads from this same subdir name by default")
    ap.add_argument("--device", default="cpu:0",
                     help="d3rlpy device string for the export pass, e.g. 'cpu:0' or 'cuda:0' -- "
                          "CPU is plenty (this is a trace+save pass over an already-trained net, "
                          "not training itself)")
    args = ap.parse_args()

    import d3rlpy  # deferred: keep --help/argparse errors usable even if the env is broken

    ckpts = step_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"no model_<step>.d3 checkpoints found under {args.ckpt_dir}")
    print(f"[export] {len(ckpts)} checkpoints: steps {[s for s, _ in ckpts]}", flush=True)

    out_dir = os.path.join(args.ckpt_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    exported, skipped = 0, 0
    for step, path in ckpts:
        policy_path = os.path.join(out_dir, f"policy_step{step}.pt")
        if os.path.exists(policy_path):
            # Mirrors offline_rl_iql.sbatch's own dataset-skip idiom -- safe to
            # resubmit this stage after a partial run without redoing finished work.
            print(f"[export] step={step:7d} already exported -> {policy_path}, skipping", flush=True)
            skipped += 1
            continue
        algo = d3rlpy.load_learnable(path, device=args.device)
        algo.save_policy(policy_path)
        print(f"[export] step={step:7d} exported -> {policy_path}", flush=True)
        exported += 1

    print(f"[export] done: {exported} exported, {skipped} already present, "
          f"{len(ckpts)} total in {out_dir}", flush=True)
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
