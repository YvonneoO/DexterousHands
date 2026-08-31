#!/usr/bin/env python3
"""Roll out a trained PPO checkpoint (num_envs=1) until `target_successes`
complete, SUCCESSFUL episodes have been recorded as separate mp4 clips, or a
`max_episodes` safety cap is hit.

Unlike bidexhands/rollout_validate.py's single fixed-length continuous
recording (which mixes successful and failed episodes into one video with no
way to tell them apart), this buffers frames per-episode and only encodes +
keeps the ones where task.extras['successes'][0] > 0 at the episode boundary
(env auto-resets in-place on `done`, same continuous-stream convention
rollout_validate.py already relies on -- no manual env.reset() needed between
episodes). Arms with a near-0% eval success rate correctly end up with fewer
than `target_successes` clips (or zero) instead of hanging forever, bounded
by `max_episodes`.

Run as:
    cd <DexterousHands root>/bidexhands
    BIDEX_RECORD_DIR=<out_dir> BIDEX_TARGET_SUCCESSES=2 BIDEX_MAX_EPISODES=25 \
    python -m tactile_collection.ppo.rollout_success_videos \
        --task ShadowHandPen --algo ppo --cfg_env cfg/ShadowHandPenProprioGTTac.yaml \
        --model_dir logs/ShadowHandPen/ppo/ppo_seed42/model_6500.pt \
        --num_envs 1 --headless --seed 1234
"""
import json
import os
import subprocess

import numpy as np
from PIL import Image

# Isaac Gym must be imported before torch; config imports gymapi first.
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.utils.process_sarl import process_sarl

import torch
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # bare "ffmpeg" isn't on $PATH on VISION; use the imageio-bundled binary


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


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
    task.gym.set_camera_location(handle, task.envs[0], gymapi.Vec3(*position), gymapi.Vec3(*target))
    return handle


def capture_frame(task, camera, width, height):
    from isaacgym import gymapi

    task.gym.fetch_results(task.sim, True)
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    rgba = np.asarray(
        task.gym.get_camera_image(task.sim, task.envs[0], camera, gymapi.IMAGE_COLOR), dtype=np.uint8
    )
    return rgba.reshape(height, width, 4)[:, :, :3].copy()


def encode_video(frames, path, fps):
    tmp_dir = path + "_frames"
    os.makedirs(tmp_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame, mode="RGB").save(os.path.join(tmp_dir, f"frame_{i:05d}.png"))
    subprocess.check_call([
        FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path,
    ])
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)


def main():
    set_np_formatting()
    args = get_args()
    if not args.model_dir:
        raise ValueError("--model_dir must point to a PPO checkpoint")
    if args.num_envs != 1:
        raise ValueError(f"--num_envs must be 1 for per-episode success tracking (got {args.num_envs})")

    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    task, env = parse_task(args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))
    runner = process_sarl(args, env, cfg_train, logdir)
    policy = runner.actor_critic
    policy.eval()

    out_dir = os.environ["BIDEX_RECORD_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    target_successes = int(os.environ.get("BIDEX_TARGET_SUCCESSES", "2"))
    max_episodes = int(os.environ.get("BIDEX_MAX_EPISODES", "25"))
    max_episode_steps = int(os.environ.get("BIDEX_MAX_EPISODE_STEPS", "600"))  # safety net, not the real reset signal
    frame_stride = max(1, int(os.environ.get("BIDEX_FRAME_STRIDE", "2")))
    width = int(os.environ.get("BIDEX_VIDEO_WIDTH", "640"))
    height = int(os.environ.get("BIDEX_VIDEO_HEIGHT", "480"))
    fps = int(os.environ.get("BIDEX_VIDEO_FPS", "30"))

    camera = create_camera(task, width, height)
    obs = env.reset()

    successes_found = 0
    episodes_tried = 0
    frame_buf = []
    step_in_ep = 0
    saved_paths = []

    with torch.no_grad():
        while successes_found < target_successes and episodes_tried < max_episodes:
            actions = policy.act_inference(obs)
            next_obs, rew, done, infos = env.step(actions)
            obs.copy_(next_obs)
            step_in_ep += 1

            if step_in_ep % frame_stride == 0:
                frame_buf.append(capture_frame(task, camera, width, height))

            done_np = as_numpy(done).astype(bool)
            episode_over = bool(done_np[0]) or step_in_ep >= max_episode_steps
            if episode_over:
                success = False
                if isinstance(infos, dict) and "successes" in infos:
                    success = bool(as_numpy(infos["successes"])[0] > 0)
                elif hasattr(task, "successes"):
                    success = bool(as_numpy(task.successes)[0] > 0)
                episodes_tried += 1
                if success and frame_buf:
                    successes_found += 1
                    path = os.path.join(out_dir, f"success_ep{successes_found}.mp4")
                    encode_video(frame_buf, path, fps)
                    saved_paths.append(path)
                    print(f"[rollout] episode {episodes_tried}: SUCCESS -> {path} ({len(frame_buf)} frames)", flush=True)
                else:
                    print(f"[rollout] episode {episodes_tried}: fail (success={success}, frames={len(frame_buf)})", flush=True)
                frame_buf = []
                step_in_ep = 0

    summary = {
        "task": args.task,
        "cfg_env": args.cfg_env,
        "checkpoint": os.path.abspath(args.model_dir),
        "target_successes": target_successes,
        "successes_found": successes_found,
        "episodes_tried": episodes_tried,
        "videos": saved_paths,
    }
    with open(os.path.join(out_dir, "rollout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("ROLLOUT_SUCCESS_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
