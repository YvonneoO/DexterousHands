#!/usr/bin/env python3
"""Finite deterministic PPO rollout with task metrics, state traces, and RGB video."""

import json
import math
import os
import subprocess
import time

import numpy as np
from PIL import Image

# Isaac Gym must be imported before torch; config imports gymapi first.
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.utils.process_sarl import process_sarl
import torch


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def env0(value, num_envs):
    arr = as_numpy(value)
    if arr.ndim and arr.shape[0] == num_envs:
        return arr[0].copy()
    return None


def quaternion_distance(q1, q2):
    q1 = q1 / torch.clamp(torch.linalg.norm(q1, dim=-1, keepdim=True), min=1e-8)
    q2 = q2 / torch.clamp(torch.linalg.norm(q2, dim=-1, keepdim=True), min=1e-8)
    dots = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dots)


def create_camera(task, width, height):
    from isaacgym import gymapi

    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    props.enable_tensors = False
    handle = task.gym.create_camera_sensor(task.envs[0], props)
    if handle < 0:
        raise RuntimeError("Isaac Gym failed to create the rollout camera")

    position = [float(x) for x in os.environ.get("BIDEX_CAMERA_POS", "1.35,-1.35,1.15").split(",")]
    target = [float(x) for x in os.environ.get("BIDEX_CAMERA_TARGET", "-0.25,-0.25,0.55").split(",")]
    task.gym.set_camera_location(
        handle,
        task.envs[0],
        gymapi.Vec3(*position),
        gymapi.Vec3(*target),
    )
    return handle


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

    steps = int(os.environ.get("BIDEX_EVAL_STEPS", "1200"))
    record_dir = os.environ.get("BIDEX_RECORD_DIR", "")
    frame_stride = max(1, int(os.environ.get("BIDEX_FRAME_STRIDE", "2")))
    width = int(os.environ.get("BIDEX_VIDEO_WIDTH", "640"))
    height = int(os.environ.get("BIDEX_VIDEO_HEIGHT", "480"))
    fps = int(os.environ.get("BIDEX_VIDEO_FPS", "30"))
    camera = None
    frames_dir = ""
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        frames_dir = os.path.join(record_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        camera = create_camera(task, width, height)

    obs = env.reset()
    num_envs = env.num_envs
    episode_count = 0
    successful_episodes = 0
    native_success_samples = []
    rewards = []
    position_errors = []
    rotation_errors = []
    trace = {"actions": [], "reward": [], "done": []}
    state_names = [
        "shadow_hand_dof_pos",
        "shadow_hand_another_dof_pos",
        "dof_pos",
        "cur_targets",
        "object_pose",
        "object_pos",
        "object_rot",
        "goal_pose",
        "goal_pos",
        "goal_rot",
        "goal_states",
    ]
    start = time.time()

    for step in range(steps):
        with torch.no_grad():
            actions = policy.act_inference(obs)
            next_obs, rew, done, infos = env.step(actions)
            obs.copy_(next_obs)

        rew_np = as_numpy(rew)
        done_np = as_numpy(done).astype(bool)
        rewards.extend(rew_np.tolist())
        trace["actions"].append(as_numpy(actions)[0].copy())
        trace["reward"].append(float(rew_np[0]))
        trace["done"].append(bool(done_np[0]))

        native = None
        if isinstance(infos, dict) and "successes" in infos:
            native = as_numpy(infos["successes"])
        elif hasattr(task, "successes"):
            native = as_numpy(task.successes)
        if native is not None:
            native_success_samples.append(float(np.mean(native > 0)))
            if np.any(done_np):
                episode_count += int(np.sum(done_np))
                successful_episodes += int(np.sum(native[done_np] > 0))
        elif np.any(done_np):
            episode_count += int(np.sum(done_np))

        if hasattr(task, "object_pos") and hasattr(task, "goal_pos"):
            pos_error = torch.linalg.norm(task.object_pos - task.goal_pos, dim=-1)
            position_errors.extend(as_numpy(pos_error).tolist())
        if hasattr(task, "object_rot") and hasattr(task, "goal_rot"):
            rot_error = quaternion_distance(task.object_rot, task.goal_rot)
            rotation_errors.extend(as_numpy(rot_error).tolist())

        for name in state_names:
            if hasattr(task, name):
                value = env0(getattr(task, name), num_envs)
                if value is not None:
                    trace.setdefault(name, []).append(value)

        if camera is not None and step % frame_stride == 0:
            capture_frame(
                task,
                camera,
                width,
                height,
                os.path.join(frames_dir, "frame_{:06d}.png".format(step // frame_stride)),
            )

    elapsed = time.time() - start
    result_dir = record_dir or os.environ.get("BIDEX_RESULT_DIR", os.getcwd())
    os.makedirs(result_dir, exist_ok=True)
    trace_path = os.path.join(result_dir, "trajectory_env0.npz")
    np.savez_compressed(trace_path, **{key: np.asarray(value) for key, value in trace.items()})

    summary = {
        "task": args.task,
        "checkpoint": os.path.abspath(args.model_dir),
        "num_envs": num_envs,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "mean_reward_per_step": float(np.mean(rewards)) if rewards else None,
        "completed_episodes": episode_count,
        "successful_episodes": successful_episodes,
        "episode_success_rate": (float(successful_episodes) / episode_count) if episode_count else None,
        "mean_native_success_fraction": float(np.mean(native_success_samples)) if native_success_samples else None,
        "position_error_mean_m": float(np.mean(position_errors)) if position_errors else None,
        "position_error_p10_m": float(np.percentile(position_errors, 10)) if position_errors else None,
        "rotation_error_mean_rad": float(np.mean(rotation_errors)) if rotation_errors else None,
        "rotation_error_p10_rad": float(np.percentile(rotation_errors, 10)) if rotation_errors else None,
        "trajectory": trace_path,
    }

    if camera is not None:
        video_path = os.path.join(record_dir, "rollout.mp4")
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video_path,
        ])
        summary["video"] = video_path

    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("ROLLOUT_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
