#!/usr/bin/env python3
"""Per-env generalization of rollout_tactile_rgb_chest.py's chest-camera code
(same file, same directory), for online use with num_envs > 1 (PPO training),
where that module's functions are hardcoded to task.envs[0] / rigid-body-state
index 0 (correct for the num_envs=1 data-collection convention every existing
rollout script uses, wrong for a live multi-env rollout).

Added here as NEW functions rather than editing rollout_tactile_rgb_chest.py
in place, so the existing, already-validated single-env data-collection
pipeline is not touched.

Key simplification that makes this tractable: Isaac Gym's DOMAIN_ENV rigid-body
indices (used throughout rollout_tactile_rgb_chest.py's
actor_named_body_env_indices/actor_body_env_indices) are PER-ENV-LOCAL and
identical across envs (every env replicates the same actor/body layout) -- so
those two functions need NO change at all. Only three things are genuinely
per-env: (1) which row of a (num_envs, ...) state tensor to read, (2) which
`task.envs[i]` a camera gets attached to / positioned in, (3) the smoothing
state in rollout_tactile_rgb_chest.position_chest_camera -- inert here because
the real data-collection recipe already runs with
BIDEX_CHEST_TARGET_SMOOTHING=0.0 (see collect_wilor_raw_rigid.sh), so the
smoothed-center blend is always skipped and the shared function-attribute
being overwritten across envs each tick is harmless.

Also provides an in-memory frame capture (no disk I/O) -- rollout_tactile_rgb_chest
.capture_frame always PIL-saves to a path, which is fine for a num_envs=1 video
dump but far too slow to call once per env per PPO tick.
"""
import os

import numpy as np

from tactile_collection.rollout_tactile_rgb_chest import (  # noqa: E402
    actor_named_body_env_indices,
    actor_body_env_indices,
    as_numpy,
    parse_vec3_env,
)

HAND_ACTORS = ("hand", "another_hand")


def env_origin(task, env_idx):
    """Isaac Gym's per-env local origin, as a (3,) numpy array. set_camera_
    location's eye/target are in EACH ENV'S OWN LOCAL FRAME, not the shared
    world frame rigid_body_states/object_pos are read in (confirmed against
    bidexhands/tasks/shadow_hand_point_cloud.py's own multi-env camera setup,
    which calls set_camera_location with the SAME literal Vec3 for every
    env -- that only makes sense if the call is env-local). Subtract this
    from any world-frame point before passing it to set_camera_location OR
    before projecting it with that camera's view/proj matrices (both operate
    in the camera's own env-local frame). Caught 2026-09-08: the single-env
    code (rollout_tactile_rgb_chest.py, gt_pose_crop.py's non-"_env"
    functions) never needed this because env 0's origin is (0,0,0) in this
    codebase's env layout -- world-frame and env-local coincide there by
    coincidence, which is why only env 0 "worked" before this fix."""
    origin = task.gym.get_env_origin(task.envs[env_idx])
    return np.asarray([origin.x, origin.y, origin.z], dtype=np.float32)


def env_row(value, num_envs, env_idx):
    arr = as_numpy(value)
    if arr.ndim and arr.shape[0] == num_envs:
        return arr[env_idx].copy()
    return None


def task_object_center_env(task, env_idx):
    obj = env_row(getattr(task, "object_pos"), task.num_envs, env_idx)
    if obj is None:
        return parse_vec3_env("BIDEX_CAMERA_TARGET", "0,0,0.55")
    return np.asarray(obj[:3], dtype=np.float32)


def rigid_body_positions_env(task, body_indices, env_idx):
    if not body_indices:
        return None
    states = env_row(getattr(task, "rigid_body_states", []), task.num_envs, env_idx)
    if states is None:
        return None
    states = np.asarray(states)
    valid = [idx for idx in body_indices if 0 <= int(idx) < states.shape[0]]
    if not valid:
        return None
    return states[np.asarray(valid, dtype=np.int64), :3].astype(np.float32)


def workspace_center_from_tensors_env(task, target_mode, env_idx):
    points = []
    if target_mode in ("hands", "palms"):
        for actor_name in HAND_ACTORS:
            pts = rigid_body_positions_env(
                task, actor_named_body_env_indices(task, actor_name, ("palm",)), env_idx)
            if pts is not None:
                points.append(pts)
    elif target_mode in ("workspace", "hands_object", "motion"):
        for actor_name in HAND_ACTORS:
            pts = rigid_body_positions_env(
                task,
                actor_named_body_env_indices(task, actor_name, ("palm", "distal", "middle", "proximal")),
                env_idx,
            )
            if pts is not None:
                points.append(pts)
        pts = rigid_body_positions_env(task, actor_body_env_indices(task, "object"), env_idx)
        if pts is not None:
            points.append(pts)
    if not points:
        return None
    pts = np.concatenate(points, axis=0)
    center_mode = os.environ.get("BIDEX_CHEST_TARGET_CENTER", "bbox").strip().lower()
    if center_mode == "mean":
        return pts.mean(axis=0).astype(np.float32)
    return ((pts.min(axis=0) + pts.max(axis=0)) * 0.5).astype(np.float32)


def position_chest_camera_env(task, camera, env_idx):
    """Per-env chest-camera placement -- same formula as
    rollout_tactile_rgb_chest.position_chest_camera, generalized off env0.
    Smoothing is intentionally NOT carried per-env (see module docstring):
    the real recipe runs BIDEX_CHEST_TARGET_SMOOTHING=0.0, which already
    disables the blend, so this is equivalent for that recipe."""
    from isaacgym import gymapi

    target_mode = os.environ.get("BIDEX_CHEST_TARGET_MODE", "object").strip().lower()
    obj = task_object_center_env(task, env_idx)
    dynamic_center = workspace_center_from_tensors_env(task, target_mode, env_idx)
    if dynamic_center is not None:
        obj = dynamic_center
    eye_offset = parse_vec3_env("BIDEX_CHEST_EYE_OFFSET", "0.0,-0.72,0.24")
    target_offset = parse_vec3_env("BIDEX_CHEST_TARGET_OFFSET", "0.0,0.0,0.02")
    eye = obj + eye_offset
    target = obj + target_offset
    origin = env_origin(task, env_idx)
    eye_local = eye - origin
    target_local = target - origin
    if os.environ.get("GT_POSE_CROP_DEBUG") == "1":
        print(f"[multi_env_camera debug] env={env_idx} obj(world)={obj.tolist()} "
              f"origin={origin.tolist()} eye_local={eye_local.tolist()} target_local={target_local.tolist()}",
              flush=True)
    task.gym.set_camera_location(
        camera, task.envs[env_idx],
        gymapi.Vec3(*eye_local.tolist()), gymapi.Vec3(*target_local.tolist())
    )
    return eye, target  # world-frame, for logging/consistency with the single-env functions


def position_ego_camera_env(task, camera, palm_handle, env_idx):
    from isaacgym import gymapi

    palm = task.gym.get_rigid_transform(task.envs[env_idx], palm_handle).p
    palm_xyz = np.asarray([palm.x, palm.y, palm.z], dtype=np.float32)
    obj = task_object_center_env(task, env_idx)
    away = palm_xyz - obj
    norm = float(np.linalg.norm(away))
    if norm < 1.0e-6:
        away = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        away /= norm
    backoff = float(os.environ.get("BIDEX_EGO_BACKOFF_M", "0.05"))
    up = float(os.environ.get("BIDEX_EGO_UP_M", "0.14"))
    eye = palm_xyz + backoff * away + np.asarray([0.0, 0.0, up], dtype=np.float32)
    origin = env_origin(task, env_idx)
    task.gym.set_camera_location(
        camera, task.envs[env_idx],
        gymapi.Vec3(*(eye - origin).tolist()), gymapi.Vec3(*(obj - origin).tolist())
    )
    return eye, obj  # world-frame, for logging/consistency with the single-env functions


def position_camera_env(task, camera, palm_handle, env_idx):
    mode = os.environ.get("BIDEX_CAMERA_MODE", "palm").strip().lower()
    if mode == "chest":
        return position_chest_camera_env(task, camera, env_idx)
    return position_ego_camera_env(task, camera, palm_handle, env_idx)


def create_cameras(task, width, height):
    """One camera sensor per env (Isaac Gym has no notion of a camera shared
    across envs). Returns (cameras, palm_handles, palm_name) -- cameras[i] /
    palm_handles[i] belong to task.envs[i]; palm_name is the same rigid-body
    name in every env (actor layouts are identical across envs)."""
    from isaacgym import gymapi

    cameras = []
    palm_handles = []
    palm_name = None
    for i in range(task.num_envs):
        props = gymapi.CameraProperties()
        props.width = width
        props.height = height
        props.enable_tensors = False
        handle = task.gym.create_camera_sensor(task.envs[i], props)
        if handle < 0:
            raise RuntimeError(f"Isaac Gym failed to create the camera sensor for env {i}")
        actor = task.gym.find_actor_handle(task.envs[i], "hand")
        if actor < 0:
            raise RuntimeError(f"Cannot create camera for env {i}: actor 'hand' was not found")
        names = task.gym.get_actor_rigid_body_names(task.envs[i], actor)
        idx = next((j for j, name in enumerate(names) if "palm" in name.lower()), 0)
        palm_handles.append(task.gym.get_actor_rigid_body_handle(task.envs[i], actor, idx))
        if palm_name is None:
            palm_name = names[idx]
        cameras.append(handle)
    return cameras, palm_handles, palm_name


def render_all_and_capture(task, cameras, width, height):
    """ONE step_graphics + render_all_camera_sensors call covers every env's
    camera at once (Isaac Gym renders the whole sim per call, not per camera)
    -- then one get_camera_image() per env to read that env's own buffer back.
    Returns a (num_envs, height, width, 3) uint8 RGB array, in memory only
    (no disk I/O -- this runs once per PPO tick, unlike capture_frame)."""
    from isaacgym import gymapi

    task.gym.fetch_results(task.sim, True)
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)

    frames = np.empty((task.num_envs, height, width, 3), dtype=np.uint8)
    for i, camera in enumerate(cameras):
        rgba = np.asarray(
            task.gym.get_camera_image(task.sim, task.envs[i], camera, gymapi.IMAGE_COLOR),
            dtype=np.uint8,
        ).reshape(height, width, 4)
        frames[i] = rgba[:, :, :3]
    return frames
