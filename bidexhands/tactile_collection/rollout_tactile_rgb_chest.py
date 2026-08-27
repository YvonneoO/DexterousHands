#!/usr/bin/env python3
"""Collect EgoTouch-layout pressure and RGB frames from the same rollout."""

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# Isaac Gym must be imported before torch; the Bi-DexHands config import does that.
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.utils.process_sarl import process_sarl
import torch
from isaacgym import gymtorch

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


def parse_vec3_env(name, default):
    text = os.environ.get(name, default)
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise ValueError("{} must contain exactly 3 comma-separated floats".format(name))
    return np.asarray(values, dtype=np.float32)


def parse_crop_box_env(name, default):
    text = os.environ.get(name, default)
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 4:
        raise ValueError("{} must contain exactly 4 comma-separated floats: left,top,right,bottom".format(name))
    left, top, right, bottom = values
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError("{} must be normalized crop coords within [0,1] with left<right and top<bottom".format(name))
    return values


def parse_bool_env(name, default=False):
    text = os.environ.get(name)
    if text is None:
        return bool(default)
    return text.strip().lower() in ("1", "true", "yes", "y", "on")


def apply_visual_style(task):
    """Optional render-only styling for camera QA.

    This changes Isaac Gym visual materials only; it does not touch physics,
    observations, rewards, actions, or saved robot/object states.
    """
    if not parse_bool_env("BIDEX_HAND_COLOR_SAME", False):
        return
    from isaacgym import gymapi

    hand_color = parse_vec3_env("BIDEX_HAND_COLOR_RGB", "0.42,0.52,0.56")
    object_color = parse_vec3_env("BIDEX_OBJECT_COLOR_RGB", "0.54,0.48,0.38")
    for actor_name, color in (
        ("hand", hand_color),
        ("another_hand", hand_color),
        ("object", object_color),
    ):
        actor = task.gym.find_actor_handle(task.envs[0], actor_name)
        if actor < 0:
            continue
        body_count = task.gym.get_actor_rigid_body_count(task.envs[0], actor)
        for body_id in range(body_count):
            task.gym.set_rigid_body_color(
                task.envs[0], actor, body_id, gymapi.MESH_VISUAL, gymapi.Vec3(*color.tolist())
            )


def _goal_object_indices_for_render(task):
    indices = getattr(task, "goal_object_indices", None)
    root_states = getattr(task, "root_state_tensor", None)
    if indices is None or root_states is None:
        return None, None
    if not torch.is_tensor(indices):
        indices = torch.as_tensor(indices, device=task.device)
    indices_long = indices.to(device=root_states.device, dtype=torch.long).flatten()
    if indices_long.numel() == 0:
        return None, None
    valid = indices_long[(indices_long >= 0) & (indices_long < root_states.shape[0])]
    if valid.numel() == 0:
        return None, None
    return valid, valid.to(dtype=torch.int32)


def hide_goal_object_for_render(task):
    """Temporarily hide Handover-style target/goal visual from RGB capture.

    Some Bi-DexHands tasks create a static `goal_object` actor as a visual
    target.  For RGB->tactile learning this ghost object is a label leak /
    distractor, but we still want the policy, reward, observations, and
    physics step to stay identical.  During camera rendering only, move the
    goal actor far away, render the image, then restore its root state before
    the next environment step.
    """
    if not parse_bool_env("BIDEX_HIDE_GOAL_OBJECT_VISUAL", False):
        return None
    indices_long, indices_int = _goal_object_indices_for_render(task)
    if indices_long is None:
        return None
    original = task.root_state_tensor[indices_long].clone()
    far = torch.as_tensor(
        parse_vec3_env("BIDEX_HIDE_GOAL_OBJECT_POS", "1000,1000,-1000"),
        device=task.root_state_tensor.device,
        dtype=task.root_state_tensor.dtype,
    )
    task.root_state_tensor[indices_long, 0:3] = far
    task.root_state_tensor[indices_long, 7:13] = 0
    task.gym.set_actor_root_state_tensor_indexed(
        task.sim,
        gymtorch.unwrap_tensor(task.root_state_tensor),
        gymtorch.unwrap_tensor(indices_int),
        int(indices_int.numel()),
    )
    return indices_long, indices_int, original


def restore_goal_object_after_render(task, hidden_state):
    if hidden_state is None:
        return
    indices_long, indices_int, original = hidden_state
    task.root_state_tensor[indices_long] = original
    task.gym.set_actor_root_state_tensor_indexed(
        task.sim,
        gymtorch.unwrap_tensor(task.root_state_tensor),
        gymtorch.unwrap_tensor(indices_int),
        int(indices_int.numel()),
    )


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


def task_object_center(task):
    obj = env0(getattr(task, "object_pos"), task.num_envs)
    if obj is None:
        return parse_vec3_env("BIDEX_CAMERA_TARGET", "0,0,0.55")
    return np.asarray(obj[:3], dtype=np.float32)


def actor_body_env_indices(task, actor_name):
    from isaacgym import gymapi

    actor = task.gym.find_actor_handle(task.envs[0], actor_name)
    if actor < 0:
        return []
    count = task.gym.get_actor_rigid_body_count(task.envs[0], actor)
    return [
        task.gym.get_actor_rigid_body_index(task.envs[0], actor, i, gymapi.DOMAIN_ENV)
        for i in range(count)
    ]


def actor_named_body_env_indices(task, actor_name, name_patterns):
    from isaacgym import gymapi

    actor = task.gym.find_actor_handle(task.envs[0], actor_name)
    if actor < 0:
        return []
    names = task.gym.get_actor_rigid_body_names(task.envs[0], actor)
    patterns = tuple(pattern.lower() for pattern in name_patterns)
    indices = []
    for i, name in enumerate(names):
        lower = name.lower()
        if any(pattern in lower for pattern in patterns):
            indices.append(task.gym.get_actor_rigid_body_index(task.envs[0], actor, i, gymapi.DOMAIN_ENV))
    return indices


def rigid_body_positions(task, body_indices):
    if not body_indices:
        return None
    states = env0(getattr(task, "rigid_body_states", []), task.num_envs)
    if states is None:
        return None
    states = np.asarray(states)
    valid = [idx for idx in body_indices if 0 <= int(idx) < states.shape[0]]
    if not valid:
        return None
    return states[np.asarray(valid, dtype=np.int64), :3].astype(np.float32)


def workspace_center_from_tensors(task, target_mode):
    """GPU-pipeline-safe dynamic target center for chest-view collection."""
    points = []
    if target_mode in ("hands", "palms"):
        for actor_name in ("hand", "another_hand"):
            pts = rigid_body_positions(task, actor_named_body_env_indices(task, actor_name, ("palm",)))
            if pts is not None:
                points.append(pts)
    elif target_mode in ("workspace", "hands_object", "motion"):
        # Distal/palm/object points follow the manipulation workspace without
        # letting large forearm meshes dominate the camera center.
        for actor_name in ("hand", "another_hand"):
            pts = rigid_body_positions(
                task,
                actor_named_body_env_indices(task, actor_name, ("palm", "distal", "middle", "proximal")),
            )
            if pts is not None:
                points.append(pts)
        pts = rigid_body_positions(task, actor_body_env_indices(task, "object"))
        if pts is not None:
            points.append(pts)
    if not points:
        return None
    pts = np.concatenate(points, axis=0)
    center_mode = os.environ.get("BIDEX_CHEST_TARGET_CENTER", "bbox").strip().lower()
    if center_mode == "mean":
        return pts.mean(axis=0).astype(np.float32)
    return ((pts.min(axis=0) + pts.max(axis=0)) * 0.5).astype(np.float32)


def position_camera(task, camera, palm_handle):
    mode = os.environ.get("BIDEX_CAMERA_MODE", "palm").strip().lower()
    if mode == "chest":
        return position_chest_camera(task, camera)
    return position_ego_camera(task, camera, palm_handle)


def position_chest_camera(task, camera):
    """Place a chest-style camera looking forward at both hands.

    `BIDEX_CHEST_TARGET_MODE=workspace` follows the current hand/object
    workspace center using rigid-body tensors, which is safe under Isaac Gym's
    GPU pipeline and keeps later rollout frames from drifting out of view.
    """
    from isaacgym import gymapi

    target_mode = os.environ.get("BIDEX_CHEST_TARGET_MODE", "object").strip().lower()
    obj = task_object_center(task)
    dynamic_center = workspace_center_from_tensors(task, target_mode)
    if dynamic_center is not None:
        obj = dynamic_center
    smoothing = float(os.environ.get("BIDEX_CHEST_TARGET_SMOOTHING", "0.0"))
    smoothing = min(max(smoothing, 0.0), 0.99)
    previous = getattr(position_chest_camera, "_smoothed_center", None)
    if previous is not None and smoothing > 0.0:
        obj = (smoothing * previous + (1.0 - smoothing) * obj).astype(np.float32)
    position_chest_camera._smoothed_center = obj.copy()
    eye_offset = parse_vec3_env("BIDEX_CHEST_EYE_OFFSET", "0.0,-0.72,0.24")
    target_offset = parse_vec3_env("BIDEX_CHEST_TARGET_OFFSET", "0.0,0.0,0.02")
    eye = obj + eye_offset
    target = obj + target_offset
    task.gym.set_camera_location(
        camera, task.envs[0], gymapi.Vec3(*eye.tolist()), gymapi.Vec3(*target.tolist())
    )
    return eye, target


def position_ego_camera(task, camera, palm_handle):
    """Place the camera at the right palm and aim it at the task object.

    Unlike the old fixed world camera, the optical center now follows the
    robot palm.  A small backoff keeps the camera out of the hand/object mesh;
    the exact eye and target are logged with every RGB frame.
    """
    from isaacgym import gymapi

    palm = task.gym.get_rigid_transform(task.envs[0], palm_handle).p
    palm_xyz = np.asarray([palm.x, palm.y, palm.z], dtype=np.float32)
    obj = task_object_center(task)
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

    hidden_goal_state = hide_goal_object_for_render(task)
    task.gym.fetch_results(task.sim, True)
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    rgba = np.asarray(
        task.gym.get_camera_image(task.sim, task.envs[0], camera, gymapi.IMAGE_COLOR),
        dtype=np.uint8,
    )
    restore_goal_object_after_render(task, hidden_goal_state)
    rgba = rgba.reshape(height, width, 4)
    image = Image.fromarray(rgba[:, :, :3], mode="RGB")
    if parse_bool_env("BIDEX_CROP_RGB", False):
        left, top, right, bottom = parse_crop_box_env("BIDEX_RGB_CROP_BOX", "0.18,0.10,0.82,0.72")
        crop = (
            int(round(left * width)),
            int(round(top * height)),
            int(round(right * width)),
            int(round(bottom * height)),
        )
        image = image.crop(crop).resize((width, height), Image.Resampling.LANCZOS)
    image.save(path)


def encode_rgb(frames_dir, output_mp4, fps):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output_mp4,
    ])


def render_tactile(script_dir, pressure_path, output_mp4, fps, stride):
    subprocess.check_call([
        os.environ.get("PYTHON", sys.executable), os.path.join(script_dir, "render_tactile.py"),
        pressure_path, output_mp4, "--fps", str(fps), "--stride", str(stride),
    ])


def compose_side_by_side(rgb_mp4, tactile_mp4, output_mp4):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", rgb_mp4, "-i", tactile_mp4,
        "-filter_complex",
        "[0:v]scale=800:-2,pad=800:600:(ow-iw)/2:(oh-ih)/2,setsar=1[rgb];"
        "[1:v]scale=1200:600,setsar=1[tactile];"
        "[rgb][tactile]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        output_mp4,
    ])


def contact_coverage_report(arrays, step_mask, unmapped_body_totals, min_mapped_force_fraction):
    """Summarize how much physical hand-object contact is represented by the taxel chart."""
    report = {
        "schema_version": 1,
        "definition": (
            "Positive Isaac RigidContact.lambda contacts between any rigid body of each hand actor "
            "and any rigid body of the object actor. Mapped contacts land on the 217-taxel chart; "
            "unmapped contacts occur on hand rigid bodies without taxels."
        ),
        "minimum_mapped_force_fraction": float(min_mapped_force_fraction),
        "sides": {},
    }
    combined = {
        "total_contact_count": 0,
        "mapped_contact_count": 0,
        "unmapped_contact_count": 0,
        "total_normal_force_n": 0.0,
        "mapped_normal_force_n": 0.0,
        "unmapped_normal_force_n": 0.0,
    }
    available = True
    for side in ("left", "right"):
        prefix = side + "_"
        availability = np.asarray(arrays[prefix + "coverage_available"], dtype=bool)[step_mask]
        side_available = bool(availability.size and availability.all())
        available = available and side_available
        values = {
            "total_contact_count": int(np.asarray(
                arrays[prefix + "total_hand_object_contact_count"]
            )[step_mask].sum()),
            "mapped_contact_count": int(np.asarray(
                arrays[prefix + "mapped_hand_object_contact_count"]
            )[step_mask].sum()),
            "unmapped_contact_count": int(np.asarray(
                arrays[prefix + "unmapped_hand_object_contact_count"]
            )[step_mask].sum()),
            "total_normal_force_n": float(np.asarray(
                arrays[prefix + "total_hand_object_normal_force_n"]
            )[step_mask].sum()),
            "mapped_normal_force_n": float(np.asarray(
                arrays[prefix + "mapped_hand_object_normal_force_n"]
            )[step_mask].sum()),
            "unmapped_normal_force_n": float(np.asarray(
                arrays[prefix + "unmapped_hand_object_normal_force_n"]
            )[step_mask].sum()),
        }
        total_force = values["total_normal_force_n"]
        total_count = values["total_contact_count"]
        values["mapped_force_fraction"] = (
            values["mapped_normal_force_n"] / total_force if total_force > 0.0 else None
        )
        values["mapped_contact_fraction"] = (
            float(values["mapped_contact_count"]) / float(total_count) if total_count > 0 else None
        )
        values["coverage_available"] = side_available
        values["unmapped_body_force_n"] = dict(
            sorted(unmapped_body_totals.get(side, {}).get("force_n", {}).items())
        )
        values["unmapped_body_contact_count"] = dict(
            sorted(unmapped_body_totals.get(side, {}).get("contact_count", {}).items())
        )
        report["sides"][side] = values
        for key in combined:
            combined[key] += values[key]

    total_force = combined["total_normal_force_n"]
    total_count = combined["total_contact_count"]
    combined["mapped_force_fraction"] = (
        combined["mapped_normal_force_n"] / total_force if total_force > 0.0 else None
    )
    combined["mapped_contact_fraction"] = (
        float(combined["mapped_contact_count"]) / float(total_count) if total_count > 0 else None
    )
    combined["coverage_available"] = available
    report["combined"] = combined
    report["pass"] = bool(
        available
        and total_force > 0.0
        and combined["mapped_force_fraction"] >= float(min_mapped_force_fraction)
    )
    if not available:
        report["failure_reason"] = "Per-contact coverage requires BIDEX_CONTACT_PROJECTION=rigid_contacts."
    elif total_force <= 0.0:
        report["failure_reason"] = "No positive hand-object normal force was observed."
    elif not report["pass"]:
        report["failure_reason"] = "Mapped normal-force fraction is below the configured threshold."
    else:
        report["failure_reason"] = None
    return report


def write_success_episode_artifact(
    result_dir, frames_dir, script_dir, arrays, trace, episode_id, actual_steps, fps, frame_stride,
    coverage_unmapped_by_episode, min_mapped_force_fraction
):
    frame_indices = np.asarray(trace["frame_index"], dtype=np.int32)
    episode_ids = np.asarray(trace["episode_id"], dtype=np.int32)
    step_mask = episode_ids == int(episode_id)
    success_dir = os.path.join(result_dir, "successful_episodes", "episode_{:06d}".format(int(episode_id)))
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
    frame_steps = np.asarray(trace["rgb_frame_step"], dtype=np.int32)
    selected_steps = set(frame_indices[step_mask].tolist())
    frame_mask = np.asarray([int(s) in selected_steps for s in frame_steps])
    for key, value in trace.items():
        arr = np.asarray(value)
        if arr.ndim and arr.shape[0] == actual_steps:
            success_trace[key] = arr[step_mask]
        elif key.startswith("rgb_"):
            success_trace[key] = arr[frame_mask]
    success_trajectory_path = os.path.join(success_dir, "trajectory_env0.npz")
    np.savez_compressed(success_trajectory_path, **success_trace)

    coverage_report = contact_coverage_report(
        arrays,
        step_mask,
        coverage_unmapped_by_episode.get(int(episode_id), {}),
        min_mapped_force_fraction,
    )
    coverage_path = os.path.join(success_dir, "contact_coverage.json")
    with open(coverage_path, "w") as handle:
        json.dump(coverage_report, handle, indent=2, sort_keys=True)

    success_frame_count = 0
    for frame_number, sim_step in enumerate(frame_steps):
        if int(sim_step) in selected_steps:
            shutil.copy2(
                os.path.join(frames_dir, "frame_{:06d}.png".format(frame_number)),
                os.path.join(success_frames, "frame_{:06d}.png".format(success_frame_count)),
            )
            success_frame_count += 1

    success_rgb = os.path.join(success_dir, "rgb.mp4")
    success_tactile = os.path.join(success_dir, "tactile.mp4")
    success_paired = os.path.join(success_dir, "rgb_tactile_side_by_side.mp4")
    keep_component_videos = parse_bool_env("BIDEX_KEEP_COMPONENT_VIDEOS", False)
    write_success_side_by_side = parse_bool_env("BIDEX_WRITE_SUCCESS_SIDE_BY_SIDE", True)
    if write_success_side_by_side or keep_component_videos:
        encode_rgb(success_frames, success_rgb, fps)
        render_tactile(script_dir, success_pressure_path, success_tactile, fps, frame_stride)
        if write_success_side_by_side:
            compose_side_by_side(success_rgb, success_tactile, success_paired)
        if not keep_component_videos:
            for component_video in (success_rgb, success_tactile):
                if os.path.exists(component_video):
                    os.unlink(component_video)
    return {
        "episode_id": int(episode_id),
        "steps": int(step_mask.sum()),
        "rgb_frames": int(success_frame_count),
        "pressure_grids": success_pressure_path,
        "trajectory": success_trajectory_path,
        "contact_coverage": coverage_path,
        "contact_coverage_pass": coverage_report["pass"],
        "mapped_force_fraction": coverage_report["combined"]["mapped_force_fraction"],
        "rgb_video": success_rgb if keep_component_videos else None,
        "tactile_video": success_tactile if keep_component_videos else None,
        "side_by_side_video": success_paired if write_success_side_by_side else None,
        "keep_component_videos": keep_component_videos,
        "write_side_by_side_video": write_success_side_by_side,
    }


def main():
    set_np_formatting()
    args = get_args()
    if not args.model_dir:
        raise ValueError("--model_dir must point to a PPO checkpoint")

    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    task, env = parse_task(args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))
    apply_visual_style(task)
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
    net_contact_force_tensor = task.gym.acquire_net_contact_force_tensor(task.sim)
    net_contact_forces = gymtorch.wrap_tensor(net_contact_force_tensor).view(env.num_envs, -1, 3)
    contact_projection = os.environ.get("BIDEX_CONTACT_PROJECTION", "net_force_tensor").strip().lower()
    min_mapped_force_fraction = float(os.environ.get("BIDEX_MIN_MAPPED_FORCE_FRACTION", "0.95"))

    obs = env.reset()
    tactile = {
        "left_pressure_grid": [], "right_pressure_grid": [],
        "left_force_grid_n": [], "right_force_grid_n": [],
        "left_source_force_n": [], "right_source_force_n": [],
        "left_reconstructed_force_n": [], "right_reconstructed_force_n": [],
        "left_contact_count": [], "right_contact_count": [],
        "left_coverage_available": [], "right_coverage_available": [],
        "left_total_hand_object_contact_count": [], "right_total_hand_object_contact_count": [],
        "left_mapped_hand_object_contact_count": [], "right_mapped_hand_object_contact_count": [],
        "left_unmapped_hand_object_contact_count": [], "right_unmapped_hand_object_contact_count": [],
        "left_total_hand_object_normal_force_n": [], "right_total_hand_object_normal_force_n": [],
        "left_mapped_hand_object_normal_force_n": [], "right_mapped_hand_object_normal_force_n": [],
        "left_unmapped_hand_object_normal_force_n": [], "right_unmapped_hand_object_normal_force_n": [],
        "left_mapped_force_fraction": [], "right_mapped_force_fraction": [],
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
    coverage_unmapped_body_totals = {
        "left": {"force_n": {}, "contact_count": {}},
        "right": {"force_n": {}, "contact_count": {}},
    }
    coverage_unmapped_by_episode = {}
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

        if contact_projection == "net_force_tensor":
            task.gym.refresh_net_contact_force_tensor(task.sim)
            env_forces = as_numpy(net_contact_forces[0])
            if step == 0:
                print("CONTACT_PROJECTION net_force_tensor shape={}".format(env_forces.shape), flush=True)
            left_pa, left_force, left_diag = left_mapper.project_net_forces(env_forces)
            right_pa, right_force, right_diag = right_mapper.project_net_forces(env_forces)
        else:
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
            for key in (
                "coverage_available",
                "total_hand_object_contact_count",
                "mapped_hand_object_contact_count",
                "unmapped_hand_object_contact_count",
                "total_hand_object_normal_force_n",
                "mapped_hand_object_normal_force_n",
                "unmapped_hand_object_normal_force_n",
                "mapped_force_fraction",
            ):
                tactile[side + "_" + key].append(diag[key])
            for name, force in diag["per_body_force_n"].items():
                body_force_totals[side][name] = body_force_totals[side].get(name, 0.0) + force
            episode_coverage = coverage_unmapped_by_episode.setdefault(
                int(episode_id),
                {
                    "left": {"force_n": {}, "contact_count": {}},
                    "right": {"force_n": {}, "contact_count": {}},
                },
            )
            for name, force in diag["unmapped_body_force_n"].items():
                coverage_unmapped_body_totals[side]["force_n"][name] = (
                    coverage_unmapped_body_totals[side]["force_n"].get(name, 0.0) + force
                )
                episode_coverage[side]["force_n"][name] = (
                    episode_coverage[side]["force_n"].get(name, 0.0) + force
                )
            for name, count in diag["unmapped_body_contact_count"].items():
                coverage_unmapped_body_totals[side]["contact_count"][name] = (
                    coverage_unmapped_body_totals[side]["contact_count"].get(name, 0) + count
                )
                episode_coverage[side]["contact_count"][name] = (
                    episode_coverage[side]["contact_count"].get(name, 0) + count
                )

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
            eye, target = position_camera(task, camera, camera_palm_handle)
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
            coverage_unmapped_body_totals = {
                "left": {"force_n": {}, "contact_count": {}},
                "right": {"force_n": {}, "contact_count": {}},
            }
            coverage_unmapped_by_episode = {}

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
            "net contact force tensor magnitude [N] distributed over each tactile body region / represented taxel area [m^2]"
            if contact_projection == "net_force_tensor"
            else "allocated Isaac RigidContact.lambda normal force [N] / represented taxel area [m^2]"
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
    write_combined_video = parse_bool_env("BIDEX_WRITE_COMBINED_VIDEO", True)
    if write_combined_video:
        encode_rgb(frames_dir, rgb_path, fps)
        render_tactile(script_dir, pressure_path, tactile_path, fps, frame_stride)
        compose_side_by_side(rgb_path, tactile_path, paired_path)
    else:
        rgb_path = None
        tactile_path = None
        paired_path = None

    successful_episode_ids = []
    for i, is_done in enumerate(trace["done"]):
        if is_done and np.isfinite(trace["native_success"][i]) and trace["native_success"][i] > 0:
            successful_episode_ids.append(int(trace["episode_id"][i]))

    successful_episode_artifacts = [
        write_success_episode_artifact(
            result_dir, frames_dir, script_dir, arrays, trace, episode, actual_steps, fps, frame_stride,
            coverage_unmapped_by_episode, min_mapped_force_fraction
        )
        for episode in successful_episode_ids
    ]
    successful_artifact = successful_episode_artifacts[0] if successful_episode_artifacts else None

    left_error = arrays["left_reconstructed_force_n"] - arrays["left_source_force_n"]
    right_error = arrays["right_reconstructed_force_n"] - arrays["right_source_force_n"]
    valid_pressures = np.concatenate([
        arrays["left_pressure_grid"][:, left_mapper.valid_mask].ravel(),
        arrays["right_pressure_grid"][:, right_mapper.valid_mask].ravel(),
    ])
    total_contacts = int(arrays["left_contact_count"].sum() + arrays["right_contact_count"].sum())
    all_steps_mask = np.ones(actual_steps, dtype=bool)
    coverage_report = contact_coverage_report(
        arrays, all_steps_mask, coverage_unmapped_body_totals, min_mapped_force_fraction
    )
    coverage_path = os.path.join(result_dir, "contact_coverage.json")
    with open(coverage_path, "w") as handle:
        json.dump(coverage_report, handle, indent=2, sort_keys=True)
    summary = {
        "task": args.task,
        "checkpoint": os.path.abspath(args.model_dir),
        "steps": actual_steps,
        "elapsed_seconds": elapsed,
        "completed_episodes": episode_count,
        "successful_episodes": successful_episodes,
        "successful_episode_ids": successful_episode_ids,
        "successful_episode_artifact": successful_artifact,
        "successful_episode_artifacts": successful_episode_artifacts,
        "episode_success_rate": (float(successful_episodes) / episode_count) if episode_count else None,
        "mean_reward_per_step": float(np.mean(trace["reward"])),
        "pressure_unit": "Pa",
        "normalization": "none",
        "layout": "EgoTouch 21x21, 217 valid taxels per hand",
        "max_pressure_pa": float(np.max(valid_pressures)),
        "mean_pressure_pa_all_valid_taxels": float(np.mean(valid_pressures)),
        "nonzero_pressure_fraction": float(np.mean(valid_pressures > 0)),
        "total_object_hand_contacts": total_contacts,
        "contact_coverage": coverage_path,
        "contact_coverage_pass": coverage_report["pass"],
        "mapped_force_fraction": coverage_report["combined"]["mapped_force_fraction"],
        "minimum_mapped_force_fraction": min_mapped_force_fraction,
        "left_peak_total_normal_force_n": float(np.max(arrays["left_source_force_n"])),
        "right_peak_total_normal_force_n": float(np.max(arrays["right_source_force_n"])),
        "max_abs_force_conservation_error_n": float(max(np.max(np.abs(left_error)), np.max(np.abs(right_error)))),
        "force_by_body_sum_over_frames_n": body_force_totals,
        "pressure_grids": pressure_path,
        "trajectory": trajectory_path,
        "rgb_video": rgb_path,
        "tactile_video": tactile_path,
        "side_by_side_video": paired_path,
        "write_combined_video": write_combined_video,
        "contact_projection": contact_projection,
        "video_frame_stride": frame_stride,
        "video_fps": fps,
        "video_frames": captured_frames,
        "camera_mode": os.environ.get("BIDEX_CAMERA_MODE", "palm"),
        "rgb_crop_enabled": parse_bool_env("BIDEX_CROP_RGB", False),
        "rgb_crop_box": os.environ.get("BIDEX_RGB_CROP_BOX", None),
        "hand_color_same": parse_bool_env("BIDEX_HAND_COLOR_SAME", False),
        "hand_color_rgb": os.environ.get("BIDEX_HAND_COLOR_RGB", None),
        "camera_palm_body": camera_palm_name,
        "camera_extrinsics": "rgb_camera_eye and rgb_camera_target in trajectory_env0.npz",
        "object_rigid_body_names": list(object_body_names),
        "object_rigid_body_state": "trajectory_env0.npz, [position xyz, quaternion xyzw, linear velocity xyz, angular velocity xyz] per object body",
        "object_dof_state": "trajectory_env0.npz, [position, velocity] per articulated object DOF when present",
        "sync": "RGB and tactile videos are rendered from identical simulation steps in this paired rollout.",
        "left_mapper": left_mapper.metadata(),
        "right_mapper": right_mapper.metadata(),
        "warning": None if total_contacts else "No hand rigid-body net contact forces were observed; tactile output is all zero.",
    }
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("PAIRED_TACTILE_RGB_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
