#!/usr/bin/env python3
"""Export a small review set: NPZ/coverage plus uniformly sampled RGB frames."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def uniform_indices(length, count):
    if length <= 0:
        return []
    return sorted(set(np.linspace(0, length - 1, min(count, length)).round().astype(int).tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--trajectories", type=int, default=5)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit("Refusing to overwrite review export: {}".format(args.output_root))

    episodes = sorted(
        episode
        for episode in args.dataset_root.glob("*/successful_episodes/episode_*")
        if (episode / "trajectory_env0.npz").is_file()
        and (episode / "pressure_grids.npz").is_file()
    )
    selected = [episodes[index] for index in uniform_indices(len(episodes), args.trajectories)]
    args.output_root.mkdir(parents=True)
    manifest = {
        "dataset_root": str(args.dataset_root),
        "available_complete_episodes": len(episodes),
        "requested_trajectories": args.trajectories,
        "frames_per_trajectory": args.frames,
        "samples": [],
    }

    for sample_index, episode in enumerate(selected):
        shard = episode.parents[1].name
        destination = args.output_root / "sample_{:02d}_{}_{}".format(
            sample_index, shard, episode.name
        )
        frame_destination = destination / "rgb_frames_5"
        frame_destination.mkdir(parents=True)
        for filename in ("trajectory_env0.npz", "pressure_grids.npz", "contact_coverage.json"):
            source = episode / filename
            if source.is_file():
                shutil.copy2(str(source), str(destination / filename))

        frames = sorted((episode / "rgb_frames").glob("frame_*.png"))
        frame_indices = uniform_indices(len(frames), args.frames)
        copied_frames = []
        for frame_index in frame_indices:
            source = frames[frame_index]
            shutil.copy2(str(source), str(frame_destination / source.name))
            copied_frames.append({"index": frame_index, "source_name": source.name})

        manifest["samples"].append({
            "sample": destination.name,
            "source_episode": str(episode),
            "source_frame_count": len(frames),
            "uniform_frames": copied_frames,
        })

    with (args.output_root / "sample_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
