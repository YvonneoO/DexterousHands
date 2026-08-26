#!/usr/bin/env python3
"""Collect EgoTouch-layout pressure and RGB frames from the same rollout."""

import json
import os
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

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


def create_camera(task, width, height):
    from isaacgym import gymapi

    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    props.enable_tensors = False
    handle = task.gym.create_camera_sensor(task.envs[0], props)
    if handle < 0:
        raise RuntimeError("Isaac Gym failed to create the paired rollout camera")

    actor = task.gym.find_actor_handle(task.envs[0], "hand")
    if actor < 0:
        raise RuntimeError("Cannot create ego camera: actor 'hand' was not found")
    names = task.gym.get_actor_rigid_body_names(task.envs[0], actor)
    palm_index = next((i for i, name in enumerate(names) if "palm" in name.lower()), 0)
    palm_handle = task.gym.get_actor_rigid_body_handle(task.envs[0], actor, palm_index)
    return handle, palm_handle, names[palm_index]


def position_ego_camera(task, camera, palm_handle):
    """Place the camera at the right palm and aim it at the task object.

    Unlike the old fixed world camera, the optical center now follows the
    robot palm.  A small backoff keeps the camera out of the hand/object mesh;
    the exact eye and target are logged with every RGB frame.
    """
    from isaacgym import gymapi

    palm = task.gym.get_rigid_transform(task.envs[0], palm_handle).p
    palm_xyz = np.asarray([palm.x, palm.y, palm.z], dtype=np.float32)
    obj = env0(getattr(task, "object_pos"), task.num_envs)
    if obj is None:
        obj = np.asarray(
            [float(x) for x in os.environ.get("BIDEX_CAMERA_TARGET", "0,0,0.55").split(",")],
            dtype=np.float32,
        )
    else:
        obj = np.asarray(obj[:3], dtype=np.float32)
    away = palm_xyz - obj
    norm = float(np.linalg.norm(away))
    if norm < 1.0e-6:
        away = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        away /= norm
    # Lift above the palm so the palm mesh itself does not occlude the view.
    # The previous 2.5 cm lift left the optical center inside/behind the hand.
    backoff = float(os.environ.get("BIDEX_EGO_BACKOFF_M", "0.05"))
    up = float(os.environ.get("BIDEX_EGO_UP_M", "0.14"))
    eye = palm_xyz + backoff * away + np.asarray([0.0, 0.0, up], dtype=np.float32)
    task.gym.set_camera_location(
        camera, task.envs[0], gymapi.Vec3(*eye.tolist()), gymapi.Vec3(*obj.tolist())
    )
    return eye, obj


def capture_frame(task, camera, width, height, path):
    from isaacgym import gymapi

    task.gym.fetch_results(task.sim, True)
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    rgba = np.asarray(
        task.gym.get_camera_image(task.sim, task.envs[0], camera, gymapi.IMAGE_COLOR),
        dtype=np.uint8,
    )
    rgba = rgba.reshape(height, width, 4)
    Image.fromarray(rgba[:, :, :3], mode="RGB").save(path)


def encode_rgb(frames_dir, output_mp4, fps):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output_mp4,
    ])


def render_tactile(script_dir, pressure_path, output_mp4, fps, stride):
    subprocess.check_call([
        os.environ.get("PYTHON", "python"), os.path.join(script_dir, "render_tactile.py"),
        pressure_path, output_mp4, "--fps", str(fps), "--stride", str(stride),
    ])


def compose_side_by_side(rgb_mp4, tactile_mp4, output_mp4):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", rgb_mp4, "-i", tactile_mp4,
        "-filter_complex",
        "[0:v]scale=-2:600,pad=800:600:(ow-iw)/2:(oh-ih)/2,setsar=1[rgb];"
        "[1:v]scale=1200:600,setsar=1[tactile];"
        "[rgb][tactile]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        output_mp4,
    ])


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
        raise ValueError("Paired collector currently requires --num_envs=1; got {}".format(env.num_envs))

    steps = int(os.environ.get("BIDEX_TACTILE_STEPS", "1200"))
    result_dir = os.path.abspath(os.environ.get("BIDEX_TACTILE_DIR", "tactile_rgb_rollout"))
    asset_dir = os.path.abspath(os.environ.get("BIDEX_TACTILE_ASSET_DIR", "tactile_collection/assets"))
    frame_stride = max(1, int(os.environ.get("BIDEX_FRAME_STRIDE", "2")))
    fps = int(os.environ.get("BIDEX_VIDEO_FPS", "30"))
    width = int(os.environ.get("BIDEX_VIDEO_WIDTH", "640"))
    height = int(os.environ.get("BIDEX_VIDEO_HEIGHT", "480"))
    frames_dir = os.path.join(result_dir, "rgb_frames")
    os.makedirs(frames_dir, exist_ok=True)

    right_mapper = EgoTouchTaxelMapper(
        task.gym, task.envs[0], "hand", "right",
        os.path.join(asset_dir, "pressure_position_mapping_right.json"),
    )
    left_mapper = EgoTouchTaxelMapper(
        task.gym, task.envs[0], "another_hand", "left",
        os.path.join(asset_dir, "pressure_position_mapping_left.json"),
    )
    camera, camera_palm_handle, camera_palm_name = create_camera(task, width, height)
    from isaacgym import gymapi
    object_actor = task.gym.find_actor_handle(task.envs[0], "object")
    object_body_names = task.gym.get_actor_rigid_body_names(task.envs[0], object_actor)
    object_body_indices = [
        task.gym.get_actor_rigid_body_index(
            task.envs[0], object_actor, i, gymapi.DOMAIN_ENV
        )
        for i in range(len(object_body_names))
    ]
    object_dof_indices = [
        task.gym.get_actor_dof_index(task.envs[0], object_actor, i, gymapi.DOMAIN_ENV)
        for i in range(task.gym.get_actor_dof_count(task.envs[0], object_actor))
    ]

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
        "native_success": [], "rgb_frame_step": [], "rgb_camera_eye": [],
        "rgb_camera_target": [],
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
    captured_frames = 0
    stop_after_successes = max(0, int(os.environ.get("BIDEX_STOP_AFTER_SUCCESSES", "0")))
    success_only_buffer = bool(int(os.environ.get(
        "BIDEX_SUCCESS_ONLY_BUFFER", "1" if stop_after_successes else "0"
    )))
    start = time.time()

    for step in range(steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)
            next_obs, rew, done, infos = env.step(actions)
            obs.copy_(next_obs)

        contacts = task.gym.get_env_rigid_contacts(task.envs[0])
        if step == 0:
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
        trace["frame_index"].append(step)
        trace["episode_id"].append(episode_id)
        trace["actions"].append(as_numpy(actions)[0].copy())
        trace["reward"].append(float(rew_np[0]))
        trace["done"].append(bool(done_np[0]))
        trace["native_success"].append(float(success_np[0]) if success_np is not None else np.nan)
        stop_now = False
        failed_episode_ended = False
        if bool(done_np[0]):
            episode_count += 1
            if success_np is not None and success_np[0] > 0:
                successful_episodes += 1
                stop_now = bool(stop_after_successes and successful_episodes >= stop_after_successes)
            else:
                failed_episode_ended = True
            episode_id += 1

        for name in state_names:
            if hasattr(task, name):
                value = env0(getattr(task, name), env.num_envs)
                if value is not None:
                    trace.setdefault(name, []).append(value)
        rigid_states = env0(getattr(task, "rigid_body_states", []), env.num_envs)
        if rigid_states is not None:
            trace.setdefault("object_rigid_body_state", []).append(
                np.asarray(rigid_states)[object_body_indices].copy()
            )
        dof_states = env0(getattr(task, "dof_state", []), env.num_envs)
        if dof_states is not None and object_dof_indices:
            trace.setdefault("object_dof_state", []).append(
                np.asarray(dof_states)[object_dof_indices].copy()
            )

        if step % frame_stride == 0:
            eye, target = position_ego_camera(task, camera, camera_palm_handle)
            capture_frame(
                task, camera, width, height,
                os.path.join(frames_dir, "frame_{:06d}.png".format(captured_frames)),
            )
            trace["rgb_frame_step"].append(step)
            trace["rgb_camera_eye"].append(eye)
            trace["rgb_camera_target"].append(target)
            captured_frames += 1
        if stop_now:
            print("SUCCESS_TARGET_REACHED step={} episode_id={}".format(step, episode_id - 1), flush=True)
            break
        if success_only_buffer and failed_episode_ended:
            # Rare-success tasks may require thousands of rollout steps.  Keep
            # only the current episode so RGB, tactile, and state memory/disk
            # usage is bounded while the deterministic success search runs.
            for values in tactile.values():
                values[:] = []
            for values in trace.values():
                values[:] = []
            for frame_name in os.listdir(frames_dir):
                if frame_name.endswith(".png"):
                    os.unlink(os.path.join(frames_dir, frame_name))
            captured_frames = 0
            body_force_totals = {"left": {}, "right": {}}

    elapsed = time.time() - start
    actual_steps = len(trace["frame_index"])
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
        "num_frames": np.asarray(actual_steps, dtype=np.int32),
        "frame_index": np.asarray(trace["frame_index"], dtype=np.int32),
        "episode_id": np.asarray(trace["episode_id"], dtype=np.int32),
        "reward": np.asarray(trace["reward"], dtype=np.float32),
        "done": np.asarray(trace["done"], dtype=bool),
        "native_success": np.asarray(trace["native_success"], dtype=np.float32),
        "control_dt_s": np.asarray(
            float(sim_params.dt) * int(cfg.get("env", {}).get("controlFrequencyInv", 1)),
            dtype=np.float32,
        ),
        "video_frame_stride": np.asarray(frame_stride, dtype=np.int32),
        "video_fps": np.asarray(fps, dtype=np.int32),
        "pressure_definition": scalar_text(
            "allocated Isaac RigidContact.lambda normal force [N] / represented taxel area [m^2]"
        ),
    })
    pressure_path = os.path.join(result_dir, "pressure_grids.npz")
    trajectory_path = os.path.join(result_dir, "trajectory_env0.npz")
    np.savez_compressed(pressure_path, **arrays)
    np.savez_compressed(trajectory_path, **{key: np.asarray(value) for key, value in trace.items()})

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rgb_path = os.path.join(result_dir, "rgb.mp4")
    tactile_path = os.path.join(result_dir, "tactile.mp4")
    paired_path = os.path.join(result_dir, "rgb_tactile_side_by_side.mp4")
    encode_rgb(frames_dir, rgb_path, fps)
    render_tactile(script_dir, pressure_path, tactile_path, fps, frame_stride)
    compose_side_by_side(rgb_path, tactile_path, paired_path)

    successful_episode_ids = []
    for i, is_done in enumerate(trace["done"]):
        if is_done and np.isfinite(trace["native_success"][i]) and trace["native_success"][i] > 0:
            successful_episode_ids.append(int(trace["episode_id"][i]))

    successful_artifact = None
    if successful_episode_ids:
        selected_episode = successful_episode_ids[0]
        step_mask = np.asarray(trace["episode_id"]) == selected_episode
        success_dir = os.path.join(result_dir, "successful_episode")
        success_frames = os.path.join(success_dir, "rgb_frames")
        os.makedirs(success_frames, exist_ok=True)
        success_pressure = {}
        for key, value in arrays.items():
            arr = np.asarray(value)
            success_pressure[key] = arr[step_mask] if arr.ndim and arr.shape[0] == actual_steps else arr
        success_pressure["num_frames"] = np.asarray(int(step_mask.sum()), dtype=np.int32)
        success_pressure_path = os.path.join(success_dir, "pressure_grids.npz")
        np.savez_compressed(success_pressure_path, **success_pressure)
        success_trace = {}
        for key, value in trace.items():
            arr = np.asarray(value)
            if arr.ndim and arr.shape[0] == actual_steps:
                success_trace[key] = arr[step_mask]
            elif key.startswith("rgb_"):
                # RGB metadata is indexed by captured frames, not simulation steps.
                frame_steps = np.asarray(trace["rgb_frame_step"], dtype=np.int32)
                episode_steps = set(np.asarray(trace["frame_index"])[step_mask].tolist())
                frame_mask = np.asarray([int(s) in episode_steps for s in frame_steps])
                success_trace[key] = arr[frame_mask]
        success_trajectory_path = os.path.join(success_dir, "trajectory_env0.npz")
        np.savez_compressed(success_trajectory_path, **success_trace)
        selected_steps = set(np.asarray(trace["frame_index"])[step_mask].tolist())
        success_frame_count = 0
        for frame_number, sim_step in enumerate(trace["rgb_frame_step"]):
            if int(sim_step) in selected_steps:
                shutil.copy2(
                    os.path.join(frames_dir, "frame_{:06d}.png".format(frame_number)),
                    os.path.join(success_frames, "frame_{:06d}.png".format(success_frame_count)),
                )
                success_frame_count += 1
        success_rgb = os.path.join(success_dir, "rgb.mp4")
        success_tactile = os.path.join(success_dir, "tactile.mp4")
        success_paired = os.path.join(success_dir, "rgb_tactile_side_by_side.mp4")
        encode_rgb(success_frames, success_rgb, fps)
        render_tactile(script_dir, success_pressure_path, success_tactile, fps, frame_stride)
        compose_side_by_side(success_rgb, success_tactile, success_paired)
        successful_artifact = {
            "episode_id": selected_episode,
            "steps": int(step_mask.sum()),
            "rgb_frames": success_frame_count,
            "pressure_grids": success_pressure_path,
            "trajectory": success_trajectory_path,
            "rgb_video": success_rgb,
            "tactile_video": success_tactile,
            "side_by_side_video": success_paired,
        }

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
        "steps": actual_steps,
        "elapsed_seconds": elapsed,
        "completed_episodes": episode_count,
        "successful_episodes": successful_episodes,
        "successful_episode_ids": successful_episode_ids,
        "successful_episode_artifact": successful_artifact,
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
        "rgb_video": rgb_path,
        "tactile_video": tactile_path,
        "side_by_side_video": paired_path,
        "video_frame_stride": frame_stride,
        "video_fps": fps,
        "video_frames": captured_frames,
        "camera_mode": "right-palm optical center, dynamically aimed at object",
        "camera_palm_body": camera_palm_name,
        "camera_extrinsics": "rgb_camera_eye and rgb_camera_target in trajectory_env0.npz",
        "object_rigid_body_names": list(object_body_names),
        "object_rigid_body_state": "trajectory_env0.npz, [position xyz, quaternion xyzw, linear velocity xyz, angular velocity xyz] per object body",
        "object_dof_state": "trajectory_env0.npz, [position, velocity] per articulated object DOF when present",
        "sync": "RGB and tactile videos are rendered from identical simulation steps in this paired rollout.",
        "left_mapper": left_mapper.metadata(),
        "right_mapper": right_mapper.metadata(),
        "warning": None if total_contacts else "No object-hand contacts were observed; tactile output is all zero.",
    }
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("PAIRED_TACTILE_RGB_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
