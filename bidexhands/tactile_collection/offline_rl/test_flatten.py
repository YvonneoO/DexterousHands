#!/usr/bin/env python3
"""Cheap, no-cluster-needed sanity check for build_mdp_dataset.py's pure-numpy
flatten/concat helpers, using synthetic in-memory dicts standing in for a
real trajectory_env0.npz / pressure_grids.npz. Deliberately has NO d3rlpy
dependency (that half of build_mdp_dataset.py -- build_dataset()/_dump_dataset()
-- can't be exercised without a real cluster/data + a d3rlpy install, so it's
left unverified; see build_mdp_dataset.py's module docstring).

Run directly: python test_flatten.py  (asserts; prints PASS on success)
"""
import numpy as np

from build_mdp_dataset import (
    discover_prop_keys,
    flatten_proprio,
    flatten_tactile,
    build_episode_arrays,
    concat_episode_transitions,
    CANDIDATE_PROP_KEYS,
)


def make_fake_episode(t, action_dim=20, prop_spec=None, final_done=True, seed=0):
    """prop_spec: dict of {key: per-step-shape-tuple}, e.g. {"object_pose": (7,)}."""
    rng = np.random.default_rng(seed)
    prop_spec = prop_spec or {}
    traj = {}
    for k, shape in prop_spec.items():
        traj[k] = rng.standard_normal((t,) + shape).astype(np.float32)
    traj["actions"] = rng.standard_normal((t, action_dim)).astype(np.float32)
    traj["reward"] = rng.standard_normal(t).astype(np.float32)
    done = np.zeros(t, dtype=bool)
    done[-1] = final_done
    traj["done"] = done

    press = {
        "left_pressure_grid": rng.random((t, 21, 21)).astype(np.float32),
        "right_pressure_grid": rng.random((t, 21, 21)).astype(np.float32),
    }
    # sprinkle some NaN to exercise the sanitize path (unmapped/non-sensor cells)
    press["left_pressure_grid"][0, 0, 0] = np.nan
    return traj, press


def test_discover_prop_keys_order_and_intersection():
    available = {"object_pose", "cur_targets", "goal_states", "something_unrelated"}
    found = discover_prop_keys(available, CANDIDATE_PROP_KEYS)
    # must preserve CANDIDATE_PROP_KEYS order, not input-set order, and drop unknowns
    expected = [k for k in CANDIDATE_PROP_KEYS if k in available]
    assert found == expected, f"{found} != {expected}"
    assert "something_unrelated" not in found


def test_flatten_proprio_shape_and_concat_order():
    t = 5
    traj = {
        "object_pose": np.arange(t * 7).reshape(t, 7).astype(np.float32),
        "cur_targets": np.arange(t * 20).reshape(t, 20).astype(np.float32),
    }
    keys = ["object_pose", "cur_targets"]  # deliberately not alphabetical
    prop = flatten_proprio(traj, keys)
    assert prop.shape == (t, 27), prop.shape
    # first 7 cols must be object_pose, next 20 must be cur_targets (order preserved)
    np.testing.assert_array_equal(prop[:, :7], traj["object_pose"])
    np.testing.assert_array_equal(prop[:, 7:], traj["cur_targets"])


def test_flatten_proprio_mismatched_T_raises():
    traj = {
        "object_pose": np.zeros((5, 7), dtype=np.float32),
        "cur_targets": np.zeros((4, 20), dtype=np.float32),  # wrong T
    }
    try:
        flatten_proprio(traj, ["object_pose", "cur_targets"])
        raise AssertionError("expected ValueError on mismatched T")
    except ValueError:
        pass


def test_flatten_tactile_shape_and_nan_zeroed():
    t = 3
    press = {
        "left_pressure_grid": np.full((t, 21, 21), np.nan, dtype=np.float32),
        "right_pressure_grid": np.zeros((t, 21, 21), dtype=np.float32),
    }
    tac = flatten_tactile(press)
    assert tac.shape == (t, 21 * 21 * 2)
    assert np.all(tac == 0.0), "NaN should be zeroed, not propagated"


def test_build_episode_arrays_p_only_vs_p_gt_tac_dims():
    t = 10
    prop_spec = {"object_pose": (7,), "cur_targets": (20,)}
    traj, press = make_fake_episode(t, action_dim=20, prop_spec=prop_spec)
    keys = ["object_pose", "cur_targets"]

    obs_p, actions, rewards, terminals = build_episode_arrays(traj, None, "p_only", keys)
    assert obs_p.shape == (t, 27)
    assert actions.shape == (t, 20)
    assert rewards.shape == (t,)
    assert terminals.shape == (t,)
    assert terminals[-1] and not terminals[:-1].any()

    obs_tac, *_ = build_episode_arrays(traj, press, "p_gt_tac", keys)
    assert obs_tac.shape == (t, 27 + 882), obs_tac.shape
    # p_only prefix of p_gt_tac's obs must be identical to the p_only obs
    np.testing.assert_array_equal(obs_tac[:, :27], obs_p)


def test_concat_forces_terminal_on_truncated_episode():
    t1, t2 = 4, 6
    prop_spec = {"object_pose": (7,)}
    keys = ["object_pose"]

    traj1, press1 = make_fake_episode(t1, prop_spec=prop_spec, final_done=True, seed=1)
    traj2, press2 = make_fake_episode(t2, prop_spec=prop_spec, final_done=False, seed=2)  # truncated!

    ep1 = build_episode_arrays(traj1, press1, "p_gt_tac", keys)
    ep2 = build_episode_arrays(traj2, press2, "p_gt_tac", keys)
    assert not ep2[3][-1], "test setup: episode 2 should start with done=False at last row"

    obs, actions, rewards, terminals = concat_episode_transitions([ep1, ep2])
    assert obs.shape[0] == t1 + t2
    # episode 1's own last-row terminal (already True) unaffected; episode 2's forced to True
    assert terminals[t1 - 1] == True
    assert terminals[t1 + t2 - 1] == True, "must be forced True despite raw done=False"
    # interior rows untouched (still False)
    assert not terminals[:t1 - 1].any()
    assert not terminals[t1:t1 + t2 - 1].any()


def main():
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} synthetic checks passed.")


if __name__ == "__main__":
    main()
