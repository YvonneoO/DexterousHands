#!/usr/bin/env python3
"""Extract and render the first native-success episode from a paired rollout."""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np


def run(command):
    subprocess.check_call(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    trajectory_path = os.path.join(root, "trajectory_env0.npz")
    pressure_path = os.path.join(root, "pressure_grids.npz")
    trace = np.load(trajectory_path)
    pressure = np.load(pressure_path)
    steps = len(trace["done"])
    done_success = np.flatnonzero((trace["done"] > 0) & (trace["native_success"] > 0))
    if not len(done_success):
        raise RuntimeError("No native-success episode exists in {}".format(trajectory_path))
    episode = int(trace["episode_id"][done_success[0]])
    step_mask = np.asarray(trace["episode_id"] == episode)
    selected_steps = set(np.asarray(trace["frame_index"])[step_mask].tolist())

    success_dir = os.path.join(root, "successful_episode")
    frames_out = os.path.join(success_dir, "rgb_frames")
    os.makedirs(frames_out, exist_ok=True)

    success_pressure = {}
    for key in pressure.files:
        value = pressure[key]
        success_pressure[key] = value[step_mask] if value.ndim and value.shape[0] == steps else value
    success_pressure["num_frames"] = np.asarray(int(step_mask.sum()), dtype=np.int32)
    success_pressure_path = os.path.join(success_dir, "pressure_grids.npz")
    np.savez_compressed(success_pressure_path, **success_pressure)

    frame_steps = np.asarray(trace["rgb_frame_step"], dtype=np.int32)
    frame_mask = np.asarray([int(step) in selected_steps for step in frame_steps])
    success_trace = {}
    for key in trace.files:
        value = trace[key]
        if value.ndim and value.shape[0] == steps:
            success_trace[key] = value[step_mask]
        elif key.startswith("rgb_") and value.shape[0] == len(frame_steps):
            success_trace[key] = value[frame_mask]
    success_trajectory_path = os.path.join(success_dir, "trajectory_env0.npz")
    np.savez_compressed(success_trajectory_path, **success_trace)

    copied = 0
    for frame_number, sim_step in enumerate(frame_steps):
        if int(sim_step) in selected_steps:
            shutil.copy2(
                os.path.join(root, "rgb_frames", "frame_{:06d}.png".format(frame_number)),
                os.path.join(frames_out, "frame_{:06d}.png".format(copied)),
            )
            copied += 1

    rgb = os.path.join(success_dir, "rgb.mp4")
    tactile = os.path.join(success_dir, "tactile.mp4")
    paired = os.path.join(success_dir, "rgb_tactile_side_by_side.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", os.path.join(frames_out, "frame_%06d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", rgb])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run([os.environ.get("PYTHON", "python"), os.path.join(script_dir, "render_tactile.py"),
         success_pressure_path, tactile, "--fps", str(args.fps), "--stride", "2"])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", rgb, "-i", tactile,
         "-filter_complex",
         "[0:v]scale=-2:600,pad=800:600:(ow-iw)/2:(oh-ih)/2,setsar=1[rgb];"
         "[1:v]scale=1200:600,setsar=1[tactile];[rgb][tactile]hstack=inputs=2[v]",
         "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", paired])

    left = success_pressure.get("left_pressure_grid")
    right = success_pressure.get("right_pressure_grid")
    summary = {
        "successful_episode_artifact": {
            "episode_id": episode,
            "steps": int(step_mask.sum()),
            "rgb_frames": copied,
            "pressure_grids": success_pressure_path,
            "trajectory": success_trajectory_path,
            "rgb_video": rgb,
            "tactile_video": tactile,
            "side_by_side_video": paired,
        },
        "native_success": float(np.max(success_trace["native_success"])),
        "camera_mode": "right-palm optical center, dynamically aimed at object",
        "camera_extrinsics": "rgb_camera_eye and rgb_camera_target in trajectory_env0.npz",
        "sync": "RGB, tactile, robot joints, and object 6DoF use the same simulation-step indices.",
        "max_pressure_pa": float(max(np.max(left), np.max(right))),
        "nonzero_pressure_fraction": float(np.mean(np.concatenate([left.ravel(), right.ravel()]) > 0)),
    }
    with open(os.path.join(root, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
