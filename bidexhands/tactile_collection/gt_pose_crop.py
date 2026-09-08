#!/usr/bin/env python3
"""GT-pose hand-crop boxes for online tactile prediction -- replaces SAM3
detection (sim_cond_cache.sam_instances + assign_bimanual, in the Ego2Contact
repo) with a direct projection of known GT hand-link 3D positions through the
sim camera's own view/projection matrices. SAM3 alone is ~79% of the current
per-frame online cost (see Ego2Contact's benchmark_online_tactile_pred.py);
this needs no detection model at all, since in sim both the hand geometry and
the camera are ground truth.

Camera placement is NOT reimplemented here -- it is imported unchanged from
this package's own tactile_collection.rollout_tactile_rgb_chest (create_camera,
position_camera, task_object_center, actor_named_body_env_indices,
rigid_body_positions, workspace_center_from_tensors), the exact module that
produced the RGB frames the tactile-prediction model was trained on. Calling
that SAME code online guarantees the camera pose distribution matches
training, rather than approximating it.

Projection math: IsaacGym's own point-cloud deprojection in
bidexhands/tasks/shadow_hand_point_cloud.py::depth_image_to_point_cloud_GPU
goes pixel+depth -> world via
    X = -(u-centerU)/width * Z * fu      fu = 2/proj[0,0]
    Y =  (v-centerV)/height * Z * fv     fv = 2/proj[1,1]
    world = [X,Y,Z,1] @ view_matrix_inv
This module needs the reverse direction (world -> pixel), obtained by
algebraically inverting those exact equations rather than deriving fresh
projection math, so it stays consistent with the one IsaacGym camera
convention already validated elsewhere in this codebase:
    cam = [X,Y,Z,1] = world_homogeneous @ view_matrix
    u = centerU - X * width  / (Z * fu)
    v = centerV + Y * height / (Z * fv)
Z < 0 for points in front of the camera (matches point_cloud.py's
`valid = Z > -depth_bar` with depth_bar > 0).

Not yet validated against a real render -- that is exactly what the VISION
smoke test (gt_pose_crop_smoke.py, same directory) is for. Run this file
directly for a numpy-only self-test of the projection algebra (no Isaac Gym
required -- safe to run locally, e.g. for a quick regression check).
"""
import os

import numpy as np

# Reused unchanged from the real data-collection camera code -- do not
# reimplement, see module docstring. Top-level `tactile_collection` import
# (not `bidexhands.tactile_collection`) matches how rollout_tactile_rgb_chest.py
# itself imports its sibling egotouch_taxels -- both resolve because
# collect_wilor_raw_rigid.sh puts bidex_root (this package's parent-of-parent)
# on PYTHONPATH alongside dex_root.
from tactile_collection.rollout_tactile_rgb_chest import (  # noqa: E402
    actor_named_body_env_indices,
    create_camera,
    position_camera,
    rigid_body_positions,
)

HAND_LINK_PATTERNS = ("palm", "distal", "middle", "proximal")
HAND_ACTORS = ("hand", "another_hand")
MIN_PAD_PX = 12.0
PAD_FRAC = 0.15


def project_world_to_pixel(points_world, view_matrix, proj_matrix, width, height):
    """(N,3) world points -> (u, v, in_front), each shape (N,). Pure numpy,
    no Isaac Gym dependency -- see module docstring for the derivation."""
    points_world = np.asarray(points_world, dtype=np.float64)
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1), dtype=np.float64)], axis=1)
    cam = homo @ np.asarray(view_matrix, dtype=np.float64)
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]

    fu = 2.0 / float(proj_matrix[0, 0])
    fv = 2.0 / float(proj_matrix[1, 1])
    center_u = width / 2.0
    center_v = height / 2.0

    in_front = z < 0.0
    safe_z = np.where(in_front, z, -1.0)  # avoid div-by-zero for the discarded points
    u = center_u - x * width / (safe_z * fu)
    v = center_v + y * height / (safe_z * fv)
    return u, v, in_front


def _hand_link_points(task, actor_name):
    idx = actor_named_body_env_indices(task, actor_name, HAND_LINK_PATTERNS)
    return rigid_body_positions(task, idx)


def gt_hand_boxes(task, view_matrix, proj_matrix, width, height,
                   pad_frac=PAD_FRAC, min_pad_px=MIN_PAD_PX):
    """Returns the same shape sim_cond_cache.assign_bimanual() does (in the
    Ego2Contact repo):
    {"left": {"box": [x1,y1,x2,y2]}, "right": {...}} (only sides whose hand
    has >=1 in-frustum link point), so this is a drop-in replacement for
    `assign_bimanual(sam_instances(proc, sam, bgr))` in the online pipeline.
    """
    candidates = []
    for actor_name in HAND_ACTORS:
        pts = _hand_link_points(task, actor_name)
        if pts is None or len(pts) == 0:
            continue
        u, v, in_front = project_world_to_pixel(pts, view_matrix, proj_matrix, width, height)
        if not np.any(in_front):
            continue
        u_k, v_k = u[in_front], v[in_front]
        x1, x2 = float(u_k.min()), float(u_k.max())
        y1, y2 = float(v_k.min()), float(v_k.max())
        pad_x = max((x2 - x1) * pad_frac, min_pad_px)
        pad_y = max((y2 - y1) * pad_frac, min_pad_px)
        x1 = float(np.clip(x1 - pad_x, 0.0, width))
        x2 = float(np.clip(x2 + pad_x, 0.0, width))
        y1 = float(np.clip(y1 - pad_y, 0.0, height))
        y2 = float(np.clip(y2 + pad_y, 0.0, height))
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        candidates.append({
            "actor": actor_name,
            "box": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2.0,
        })

    if not candidates:
        return {}
    candidates.sort(key=lambda d: d["cx"])
    sides = {}
    if len(candidates) == 1:
        sides["left"] = candidates[0]
    else:
        sides["left"] = candidates[0]
        sides["right"] = candidates[-1]
    return sides


def build_bimanual_boxes_from_task(task, camera, palm_handle, width, height):
    """Full drop-in for the online pipeline's SAM3 step: position the SAME
    dynamic chest camera used at data-collection time, then project GT hand
    geometry instead of running SAM3 detection on the rendered frame.
    Single-env only (task.envs[0]) -- see gt_hand_boxes_env / build_bimanual_
    boxes_all_envs below for the num_envs>1 (online PPO) version."""
    eye, target = position_camera(task, camera, palm_handle)
    view_matrix = task.gym.get_camera_view_matrix(task.sim, task.envs[0], camera)
    proj_matrix = task.gym.get_camera_proj_matrix(task.sim, task.envs[0], camera)
    sides = gt_hand_boxes(task, view_matrix, proj_matrix, width, height)
    return sides, eye, target


def _hand_link_points_env(task, actor_name, env_idx):
    """rigid_body_states (and hence these points) are already reported in
    each env's own local frame in this codebase -- confirmed empirically
    2026-09-08 (multi-env smoke test debug output: env1's workspace center
    read the same ballpark as env0's, not offset by its own grid spacing).
    No origin conversion needed -- see multi_env_camera.env_origin's
    docstring for the dead-end version of this file that subtracted one."""
    from tactile_collection.multi_env_camera import rigid_body_positions_env
    idx = actor_named_body_env_indices(task, actor_name, HAND_LINK_PATTERNS)
    return rigid_body_positions_env(task, idx, env_idx)


def gt_hand_boxes_env(task, view_matrix, proj_matrix, width, height, env_idx,
                       pad_frac=PAD_FRAC, min_pad_px=MIN_PAD_PX):
    """Same as gt_hand_boxes, but reads env_idx's own rigid-body state row
    instead of always env 0 -- for online PPO with num_envs > 1."""
    debug = os.environ.get("GT_POSE_CROP_DEBUG") == "1"
    candidates = []
    for actor_name in HAND_ACTORS:
        pts = _hand_link_points_env(task, actor_name, env_idx)
        if debug:
            from tactile_collection.multi_env_camera import env_origin
            print(f"[gt_pose_crop debug] env={env_idx} actor={actor_name} "
                  f"origin={env_origin(task, env_idx).tolist()} "
                  f"pts={None if pts is None else pts.tolist()}", flush=True)
        if pts is None or len(pts) == 0:
            continue
        u, v, in_front = project_world_to_pixel(pts, view_matrix, proj_matrix, width, height)
        if debug:
            print(f"[gt_pose_crop debug] env={env_idx} actor={actor_name} "
                  f"u={u.tolist()} v={v.tolist()} in_front={in_front.tolist()}", flush=True)
        if not np.any(in_front):
            continue
        u_k, v_k = u[in_front], v[in_front]
        x1, x2 = float(u_k.min()), float(u_k.max())
        y1, y2 = float(v_k.min()), float(v_k.max())
        pad_x = max((x2 - x1) * pad_frac, min_pad_px)
        pad_y = max((y2 - y1) * pad_frac, min_pad_px)
        x1 = float(np.clip(x1 - pad_x, 0.0, width))
        x2 = float(np.clip(x2 + pad_x, 0.0, width))
        y1 = float(np.clip(y1 - pad_y, 0.0, height))
        y2 = float(np.clip(y2 + pad_y, 0.0, height))
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        candidates.append({"actor": actor_name, "box": [x1, y1, x2, y2], "cx": (x1 + x2) / 2.0})

    if not candidates:
        return {}
    candidates.sort(key=lambda d: d["cx"])
    sides = {"left": candidates[0]}
    if len(candidates) > 1:
        sides["right"] = candidates[-1]
    return sides


def build_bimanual_boxes_all_envs(task, cameras, palm_handles, width, height):
    """Multi-env drop-in for the online PPO rollout loop: positions each
    env's own dynamic chest camera (multi_env_camera.position_camera_env)
    and projects that env's own GT hand geometry -- one call per PPO tick
    covers all num_envs environments. Returns a list of `sides` dicts
    (possibly {} for an env with no hand in frame) and parallel eye/target
    lists, same order as `cameras`/`palm_handles`/task.envs."""
    from tactile_collection.multi_env_camera import position_camera_env

    all_sides, eyes, targets = [], [], []
    for i in range(task.num_envs):
        eye, target = position_camera_env(task, cameras[i], palm_handles[i], i)
        view_matrix = task.gym.get_camera_view_matrix(task.sim, task.envs[i], cameras[i])
        proj_matrix = task.gym.get_camera_proj_matrix(task.sim, task.envs[i], cameras[i])
        sides = gt_hand_boxes_env(task, view_matrix, proj_matrix, width, height, i)
        all_sides.append(sides)
        eyes.append(eye)
        targets.append(target)
    return all_sides, eyes, targets


def _reference_deproject_point(u, v, z, view_matrix_inv, proj_matrix, width, height):
    """Exact port of bidexhands/tasks/shadow_hand_point_cloud.py::
    depth_image_to_point_cloud_GPU for a single point (pixel+depth -> world).
    Used only by _self_test() to round-trip-verify project_world_to_pixel is
    its true algebraic inverse -- see module docstring."""
    fu = 2.0 / float(proj_matrix[0, 0])
    fv = 2.0 / float(proj_matrix[1, 1])
    center_u, center_v = width / 2.0, height / 2.0
    x = -(u - center_u) / width * z * fu
    y = (v - center_v) / height * z * fv
    world = np.array([x, y, z, 1.0]) @ view_matrix_inv
    return world[:3]


def _self_test():
    """Numpy-only sanity check of the projection algebra -- no Isaac Gym.
    Builds a camera roughly matching the real chest-camera offset (see
    module docstring / collect_wilor_raw_rigid.sh), then:
      1. round-trips world -> pixel -> world via the codebase's own
         deprojection formula and checks exact recovery (this is the real
         correctness proof: it fails immediately if the convention this
         file assumes ever diverges from shadow_hand_point_cloud.py's), and
      2. checks the aim-target point lands at the image center and that a
         shifted point moves off-center in the expected direction."""
    width, height = 960, 720
    hfov = np.deg2rad(90.0)  # IsaacGym CameraProperties default horizontal_fov
    aspect = width / height
    proj = np.eye(4)
    proj[0, 0] = 1.0 / np.tan(hfov / 2.0)
    proj[1, 1] = aspect / np.tan(hfov / 2.0)

    eye = np.array([0.32, -0.9, 0.9])       # ~ workspace_center + BIDEX_CHEST_EYE_OFFSET
    target = np.array([0.0, 0.0, 0.08])     # ~ workspace_center + BIDEX_CHEST_TARGET_OFFSET
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # Row-vector world->camera convention consistent with this module's
    # projection code: cam = [x,y,z,1] @ view_matrix.
    rot = np.stack([right, up, -fwd], axis=1)
    view = np.eye(4)
    view[:3, :3] = rot
    view[3, :3] = -eye @ rot
    view_inv = np.linalg.inv(view)

    rng = np.random.default_rng(0)
    world_pts = target[None, :] + rng.normal(scale=0.05, size=(20, 3))
    u, v, in_front = project_world_to_pixel(world_pts, view, proj, width, height)
    assert np.all(in_front), "all sample points should be in front of the camera"

    max_err = 0.0
    for i in range(len(world_pts)):
        homo = np.concatenate([world_pts[i], [1.0]])
        z = float((homo @ view)[2])
        recovered = _reference_deproject_point(u[i], v[i], z, view_inv, proj, width, height)
        max_err = max(max_err, float(np.linalg.norm(recovered - world_pts[i])))
    assert max_err < 1e-8, f"world->pixel->world round trip should be numerically exact, got err={max_err:.3e}"
    print(f"[self_test] round-trip max error over {len(world_pts)} points: {max_err:.3e}")

    center_pt = target[None, :]
    u0, v0, front0 = project_world_to_pixel(center_pt, view, proj, width, height)
    assert front0[0]
    assert abs(u0[0] - width / 2.0) < 1e-6, f"aim target should project to image center u, got {u0[0]}"
    assert abs(v0[0] - height / 2.0) < 1e-6, f"aim target should project to image center v, got {v0[0]}"
    print(f"[self_test] aim target -> u={u0[0]:.2f} v={v0[0]:.2f} (expect {width/2:.2f} {height/2:.2f})")

    shifted = center_pt + np.array([[0.1, 0.0, 0.0]])
    u1, v1, front1 = project_world_to_pixel(shifted, view, proj, width, height)
    assert front1[0]
    assert abs(u1[0] - width / 2.0) > 1.0, "shifted point should move off the horizontal center"
    print(f"[self_test] +X-shifted -> u={u1[0]:.2f} v={v1[0]:.2f}")
    print("[self_test] OK")


if __name__ == "__main__":
    _self_test()
