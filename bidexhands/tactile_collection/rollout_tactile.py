#!/usr/bin/env python3
"""Roll out a trained Bi-DexHands policy and collect EgoTouch-layout pressure in Pa."""

import json
import os
import time

import numpy as np

# Isaac Gym must be imported before torch; the Bi-DexHands config import does that.
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.utils.process_sarl import process_sarl
import torch

from tactile_collection.egotouch_taxels import EgoTouchTaxelMapper


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def env0(value, num_envs):
    arr = as_numpy(value)
    if arr.ndim and arr.shape[0] == num_envs:
        return arr[0].copy()
    return None


def native_success(task, infos):
    if isinstance(infos, dict) and "successes" in infos:
        return as_numpy(infos["successes"])
    if hasattr(task, "successes"):
        return as_numpy(task.successes)
    return None


def scalar_text(value):
    return np.asarray(value, dtype=np.str_)


def main():
    set_np_formatting()
    args = get_args()
    if not args.model_dir:
        raise ValueError("--model_dir must point to a PPO checkpoint")

    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    task, env = parse_task(args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))
    runner = process_sarl(args, env, cfg_train, logdir)
    policy = runner.actor_critic
    policy.eval()

    if env.num_envs != 1:
        raise ValueError("Tactile collector currently requires --num_envs=1; got {}".format(env.num_envs))

    steps = int(os.environ.get("BIDEX_TACTILE_STEPS", "1200"))
    result_dir = os.path.abspath(os.environ.get("BIDEX_TACTILE_DIR", "tactile_rollout"))
    asset_dir = os.path.abspath(os.environ.get("BIDEX_TACTILE_ASSET_DIR", "tactile_collection/assets"))
    os.makedirs(result_dir, exist_ok=True)

    right_mapper = EgoTouchTaxelMapper(
        task.gym, task.envs[0], "hand", "right",
        os.path.join(asset_dir, "pressure_position_mapping_right.json"),
    )
    left_mapper = EgoTouchTaxelMapper(
        task.gym, task.envs[0], "another_hand", "left",
        os.path.join(asset_dir, "pressure_position_mapping_left.json"),
    )

    obs = env.reset()
    tactile = {
        "left_pressure_grid": [], "right_pressure_grid": [],
        "left_force_grid_n": [], "right_force_grid_n": [],
        "left_source_force_n": [], "right_source_force_n": [],
        "left_reconstructed_force_n": [], "right_reconstructed_force_n": [],
        "left_contact_count": [], "right_contact_count": [],
    }
    trace = {
        "frame_index": [], "episode_id": [], "actions": [], "reward": [], "done": [],
        "native_success": [],
    }
    state_names = [
        "shadow_hand_dof_pos", "shadow_hand_another_dof_pos", "dof_pos", "cur_targets",
        "object_pose", "object_pos", "object_rot", "goal_pose", "goal_pos", "goal_rot",
        "goal_states",
    ]
    episode_count = 0
    successful_episodes = 0
    episode_id = 0
    body_force_totals = {"left": {}, "right": {}}
    start = time.time()

    for _step in range(steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)
            next_obs, rew, done, infos = env.step(actions)
            obs.copy_(next_obs)

        contacts = task.gym.get_env_rigid_contacts(task.envs[0])
        if _step == 0:
            print("CONTACT_DTYPE {}".format(getattr(getattr(contacts, "dtype", None), "names", None)), flush=True)
        left_pa, left_force, left_diag = left_mapper.project(contacts)
        right_pa, right_force, right_diag = right_mapper.project(contacts)
        tactile["left_pressure_grid"].append(left_pa)
        tactile["right_pressure_grid"].append(right_pa)
        tactile["left_force_grid_n"].append(left_force)
        tactile["right_force_grid_n"].append(right_force)
        for side, diag in (("left", left_diag), ("right", right_diag)):
            tactile[side + "_source_force_n"].append(diag["source_force_n"])
            tactile[side + "_reconstructed_force_n"].append(diag["reconstructed_force_n"])
            tactile[side + "_contact_count"].append(diag["contact_count"])
            for name, force in diag["per_body_force_n"].items():
                body_force_totals[side][name] = body_force_totals[side].get(name, 0.0) + force

        rew_np = as_numpy(rew)
        done_np = as_numpy(done).astype(bool)
        success_np = native_success(task, infos)
        trace["frame_index"].append(_step)
        trace["episode_id"].append(episode_id)
        trace["actions"].append(as_numpy(actions)[0].copy())
        trace["reward"].append(float(rew_np[0]))
        trace["done"].append(bool(done_np[0]))
        trace["native_success"].append(float(success_np[0]) if success_np is not None else np.nan)
        if bool(done_np[0]):
            episode_count += 1
            if success_np is not None and success_np[0] > 0:
                successful_episodes += 1
            episode_id += 1

        for name in state_names:
            if hasattr(task, name):
                value = env0(getattr(task, name), env.num_envs)
                if value is not None:
                    trace.setdefault(name, []).append(value)

    elapsed = time.time() - start
    arrays = {key: np.asarray(value) for key, value in tactile.items()}
    arrays.update({
        "left_valid_mask": left_mapper.valid_mask,
        "right_valid_mask": right_mapper.valid_mask,
        "left_taxel_area_m2": left_mapper.taxel_area_m2.astype(np.float32),
        "right_taxel_area_m2": right_mapper.taxel_area_m2.astype(np.float32),
        "pressure_unit": scalar_text("Pa"),
        "force_unit": scalar_text("N"),
        "area_unit": scalar_text("m^2"),
        "normalization": scalar_text("none"),
        "layout": scalar_text("EgoTouch-21x21-217-taxels-per-hand"),
        "grid_size": np.asarray(21, dtype=np.int32),
        "num_frames": np.asarray(steps, dtype=np.int32),
        "frame_index": np.asarray(trace["frame_index"], dtype=np.int32),
        "episode_id": np.asarray(trace["episode_id"], dtype=np.int32),
        "reward": np.asarray(trace["reward"], dtype=np.float32),
        "done": np.asarray(trace["done"], dtype=bool),
        "native_success": np.asarray(trace["native_success"], dtype=np.float32),
        "control_dt_s": np.asarray(
            float(sim_params.dt) * int(cfg.get("env", {}).get("controlFrequencyInv", 1)),
            dtype=np.float32,
        ),
        "pressure_definition": scalar_text(
            "allocated Isaac RigidContact.lambda normal force [N] / represented taxel area [m^2]"
        ),
    })
    pressure_path = os.path.join(result_dir, "pressure_grids.npz")
    trajectory_path = os.path.join(result_dir, "trajectory_env0.npz")
    np.savez_compressed(pressure_path, **arrays)
    np.savez_compressed(trajectory_path, **{key: np.asarray(value) for key, value in trace.items()})

    left_error = arrays["left_reconstructed_force_n"] - arrays["left_source_force_n"]
    right_error = arrays["right_reconstructed_force_n"] - arrays["right_source_force_n"]
    valid_pressures = np.concatenate([
        arrays["left_pressure_grid"][:, left_mapper.valid_mask].ravel(),
        arrays["right_pressure_grid"][:, right_mapper.valid_mask].ravel(),
    ])
    total_contacts = int(arrays["left_contact_count"].sum() + arrays["right_contact_count"].sum())
    summary = {
        "task": args.task,
        "checkpoint": os.path.abspath(args.model_dir),
        "steps": steps,
        "elapsed_seconds": elapsed,
        "completed_episodes": episode_count,
        "successful_episodes": successful_episodes,
        "episode_success_rate": (float(successful_episodes) / episode_count) if episode_count else None,
        "mean_reward_per_step": float(np.mean(trace["reward"])),
        "pressure_unit": "Pa",
        "normalization": "none",
        "layout": "EgoTouch 21x21, 217 valid taxels per hand",
        "max_pressure_pa": float(np.max(valid_pressures)),
        "mean_pressure_pa_all_valid_taxels": float(np.mean(valid_pressures)),
        "nonzero_pressure_fraction": float(np.mean(valid_pressures > 0)),
        "total_object_hand_contacts": total_contacts,
        "left_peak_total_normal_force_n": float(np.max(arrays["left_source_force_n"])),
        "right_peak_total_normal_force_n": float(np.max(arrays["right_source_force_n"])),
        "max_abs_force_conservation_error_n": float(max(np.max(np.abs(left_error)), np.max(np.abs(right_error)))),
        "force_by_body_sum_over_frames_n": body_force_totals,
        "pressure_grids": pressure_path,
        "trajectory": trajectory_path,
        "left_mapper": left_mapper.metadata(),
        "right_mapper": right_mapper.metadata(),
        "warning": None if total_contacts else "No object-hand contacts were observed; tactile output is all zero.",
    }
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("TACTILE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
