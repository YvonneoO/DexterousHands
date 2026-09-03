#!/usr/bin/env python3
"""Shared live-rollout logic for Stage 2 student evaluation, used by both
eval_bc_student.py (one checkpoint, N episodes) and sweep_eval_bc_student.py
(many checkpoints from one training run, fewer episodes each, for a wandb
success-rate-vs-epoch curve).

Kept in one place so both scripts run the EXACT same rollout loop -- camera
positioning, tactile projection, decision cadence -- rather than risk two
copies drifting apart. Camera positioning, GT-tactile projection, and the
goal-hiding render helpers are imported from rollout_tactile_rgb_chest.py
rather than reimplemented, so eval-time inputs match training-time inputs
exactly (same camera formula, same taxel projection).
"""
import os
import sys

import numpy as np
from PIL import Image

# Isaac Gym must be imported before torch anywhere in the process (it raises
# ImportError otherwise) -- these bidexhands imports pull in isaacgym, so
# `import torch` below MUST stay after them, not just after this comment.
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex

from tactile_collection.egotouch_taxels import EgoTouchTaxelMapper
from tactile_collection.rollout_tactile_rgb_chest import (
    as_numpy, env0, native_success, create_camera, position_camera,
    hide_goal_object_for_render, restore_goal_object_after_render,
)

import torch
from tactile_collection.bc.train_bc_student import PROP_KEYS


def capture_frame_array(task, camera, width, height):
    """Same render path as rollout_tactile_rgb_chest.capture_frame, but
    returns a PIL Image instead of writing to disk -- avoids per-step disk
    I/O during rollout."""
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
    return Image.fromarray(rgba[:, :, :3], mode="RGB")


def image_to_tensor(pil_img, img_size, device):
    img = pil_img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def prop_from_task(task, num_envs, device):
    vals = []
    for key in PROP_KEYS:
        v = env0(getattr(task, key), num_envs)
        if v is None:
            raise RuntimeError(f"expected task.{key} to be a per-env tensor of shape "
                                f"(num_envs, ...); check PROP_KEYS matches this task")
        vals.append(np.asarray(v, dtype=np.float32).reshape(-1))
    prop = np.concatenate(vals)
    return torch.from_numpy(prop).unsqueeze(0).to(device)


class EvalContext:
    """Everything expensive to set up once: the IsaacGym task/env, camera,
    and (for tac_mode=gt) the taxel mappers + contact-force tensor handle."""

    def __init__(self, task, env, device, camera, camera_palm_handle,
                 left_mapper, right_mapper, net_contact_forces,
                 frame_stride, max_steps_per_episode):
        self.task = task
        self.env = env
        self.device = device
        self.camera = camera
        self.camera_palm_handle = camera_palm_handle
        self.left_mapper = left_mapper
        self.right_mapper = right_mapper
        self.net_contact_forces = net_contact_forces
        self.frame_stride = frame_stride
        self.max_steps_per_episode = max_steps_per_episode


def build_eval_context(bidex_argv, seed, camera_eye_offset, camera_target_offset,
                        tac_mode, asset_dir, frame_stride, max_steps_per_episode):
    """bidex_argv: list of args for bidexhands' own parser (e.g.
    ["--task", "ShadowHandPen", "--algo", "ppo"]), NOT including this
    process's own sys.argv[0]. Call ONCE per process -- IsaacGym env creation
    is expensive; both eval_bc_student.py and sweep_eval_bc_student.py build
    the context once and reuse it across every checkpoint they evaluate."""
    sys.argv = [sys.argv[0]] + list(bidex_argv)

    os.environ["BIDEX_CAMERA_MODE"] = "chest"
    os.environ["BIDEX_CHEST_TARGET_MODE"] = "workspace"
    os.environ["BIDEX_CHEST_TARGET_CENTER"] = "bbox"
    os.environ["BIDEX_CHEST_TARGET_SMOOTHING"] = "0.0"
    os.environ["BIDEX_CHEST_EYE_OFFSET"] = camera_eye_offset
    os.environ["BIDEX_CHEST_TARGET_OFFSET"] = camera_target_offset

    set_np_formatting()
    bidex_args = get_args()
    cfg, cfg_train, logdir = load_cfg(bidex_args)
    sim_params = parse_sim_params(bidex_args, cfg, cfg_train)
    set_seed(seed, cfg_train.get("torch_deterministic", False))
    task, env = parse_task(bidex_args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))

    if env.num_envs != 1:
        raise ValueError(f"Student eval currently requires --num_envs=1; got {env.num_envs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera, camera_palm_handle, _ = create_camera(task, 640, 480)

    left_mapper = right_mapper = net_contact_forces = None
    if tac_mode == "gt":
        from isaacgym import gymtorch
        left_mapper = EgoTouchTaxelMapper(
            task.gym, task.envs[0], "another_hand", "left",
            os.path.join(asset_dir, "pressure_position_mapping_left.json"),
        )
        right_mapper = EgoTouchTaxelMapper(
            task.gym, task.envs[0], "hand", "right",
            os.path.join(asset_dir, "pressure_position_mapping_right.json"),
        )
        net_contact_force_tensor = task.gym.acquire_net_contact_force_tensor(task.sim)
        net_contact_forces = gymtorch.wrap_tensor(net_contact_force_tensor).view(env.num_envs, -1, 3)

    return EvalContext(task, env, device, camera, camera_palm_handle,
                        left_mapper, right_mapper, net_contact_forces,
                        frame_stride, max_steps_per_episode)


def run_rollout_eval(ctx, model, tac_mode, img_size, action_dim, episodes, tac_vmax=None):
    """Roll out `model` for `episodes` fresh episodes in ctx.env. The student
    is queried and its action refreshed only every ctx.frame_stride steps,
    held (zero-order) in between -- the only cadence it's ever been trained
    at. Returns (success_rate, per_episode_results)."""
    task, env, device = ctx.task, ctx.env, ctx.device
    model.eval()

    obs = env.reset()
    zero_action = torch.zeros(1, action_dim, device=device)
    last_action = zero_action.clone()

    results = []
    episode_steps = 0
    successes = 0
    completed = 0

    while completed < episodes:
        if episode_steps % ctx.frame_stride == 0:
            position_camera(task, ctx.camera, ctx.camera_palm_handle)
            frame = capture_frame_array(task, ctx.camera, 640, 480)
            img_t = image_to_tensor(frame, img_size, device)
            prop_t = prop_from_task(task, env.num_envs, device)

            if tac_mode == "gt":
                task.gym.refresh_net_contact_force_tensor(task.sim)
                env_forces = as_numpy(ctx.net_contact_forces[0])
                left_pa, _, _ = ctx.left_mapper.project_net_forces(env_forces)
                right_pa, _, _ = ctx.right_mapper.project_net_forces(env_forces)
                # NaN marks unmapped/non-sensor cells (distinct from a real
                # zero-contact reading) -- sanitize before it hits the network,
                # matching train_bc_student.py's BCEpisodeDataset.
                left_pa = np.nan_to_num(left_pa.astype(np.float32), nan=0.0)
                right_pa = np.nan_to_num(right_pa.astype(np.float32), nan=0.0)
                tac = np.concatenate([left_pa.reshape(-1), right_pa.reshape(-1)])
                # raw pressure reaches into the tens/hundreds of thousands --
                # scale against the SAME per-task vmax training used (from the
                # checkpoint), matching train_bc_student.py's BCEpisodeDataset.
                assert tac_vmax is not None, "tac_vmax is required when tac_mode='gt'"
                tac = np.clip(tac / tac_vmax, 0.0, 1.0).astype(np.float32)
                tac_t = torch.from_numpy(tac).unsqueeze(0).to(device)
            else:
                # TODO(pred-tac live rollout): tac_mode="pred" is NOT supported here yet.
                # train_bc_student.py's tac_mode="pred" arm only trains offline against a
                # pre-generated pred_pressure_grids.npz (Ego2Contact's
                # infer_pred_tactile_offline.py, a v2-dit sim-finetuned forward pass over the
                # episode's saved rgb_frames). A live closed-loop rollout would need that same
                # SAM3+WiLoR+DINO+v2-dit forward pass run online, per step, inside this loop --
                # a separate, harder problem (see the project's earlier online-inference-latency
                # findings) and explicitly out of scope for now. Falls through to the "none"
                # (zero-tactile) branch below if ever called with tac_mode="pred".
                tac_t = torch.zeros(1, 0, device=device)

            with torch.no_grad():
                last_action = model(img_t, prop_t, tac_t)

        with torch.no_grad():
            next_obs, rew, done, infos = env.step(last_action)
            obs.copy_(next_obs)

        episode_steps += 1
        done_np = as_numpy(done).astype(bool)
        success_np = native_success(task, infos)

        if bool(done_np[0]) or episode_steps >= ctx.max_steps_per_episode:
            completed += 1
            success = bool(success_np is not None and success_np[0] > 0)
            successes += int(success)
            results.append({"episode": completed, "steps": episode_steps, "success": success})
            episode_steps = 0
            last_action = zero_action.clone()
            obs = env.reset()

    return successes / max(completed, 1), results
