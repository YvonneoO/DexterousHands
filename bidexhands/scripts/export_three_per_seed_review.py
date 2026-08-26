#!/usr/bin/env python3
"""Export three complete, uniformly spaced review episodes from every seed shard."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def uniform_indices(length, count):
    if length < count:
        raise ValueError("need at least {} episodes, found {}".format(count, length))
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_with_hash(source, destination):
    shutil.copy2(str(source), str(destination))
    return {"name": destination.name, "bytes": destination.stat().st_size,
            "sha256": sha256(destination)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_root", type=Path,
                        help="Task directory containing seed_*/successful_episodes")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--trajectories-per-seed", type=int, default=3)
    parser.add_argument("--frames-per-trajectory", type=int, default=5)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit("Refusing to overwrite review export: {}".format(args.output_root))

    seed_dirs = sorted(path for path in args.task_root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise SystemExit("No seed_* shards found in {}".format(args.task_root))

    task_manifest = {
        "source_task_root": str(args.task_root),
        "trajectories_per_seed": args.trajectories_per_seed,
        "frames_per_trajectory": args.frames_per_trajectory,
        "seeds": [],
    }
    args.output_root.mkdir(parents=True)

    for seed_dir in seed_dirs:
        episodes = sorted(
            episode for episode in (seed_dir / "successful_episodes").glob("episode_*")
            if (episode / "trajectory_env0.npz").is_file()
            and (episode / "pressure_grids.npz").is_file()
        )
        indices = uniform_indices(len(episodes), args.trajectories_per_seed)
        seed_output = args.output_root / seed_dir.name
        seed_output.mkdir()
        seed_manifest = {
            "seed_shard": seed_dir.name,
            "source": str(seed_dir),
            "available_complete_episodes": len(episodes),
            "samples": [],
        }

        for sample_index, episode_index in enumerate(indices):
            episode = episodes[episode_index]
            sample_output = seed_output / "sample_{:02d}_{}".format(
                sample_index, episode.name)
            frames_output = sample_output / "rgb_frames"
            preview_output = sample_output / "preview_5_frames"
            frames_output.mkdir(parents=True)
            preview_output.mkdir()
            files = []
            for filename in ("trajectory_env0.npz", "pressure_grids.npz",
                             "contact_coverage.json"):
                source = episode / filename
                if source.is_file():
                    files.append(copy_with_hash(source, sample_output / filename))

            frames = sorted((episode / "rgb_frames").glob("frame_*.png"))
            copied_all_frames = []
            for frame_index, source in enumerate(frames):
                metadata = copy_with_hash(source, frames_output / source.name)
                metadata["source_index"] = frame_index
                copied_all_frames.append(metadata)
            frame_indices = uniform_indices(len(frames), args.frames_per_trajectory)
            copied_preview_frames = []
            for frame_index in frame_indices:
                source = frames_output / frames[frame_index].name
                metadata = copy_with_hash(source, preview_output / source.name)
                metadata["source_index"] = frame_index
                copied_preview_frames.append(metadata)

            seed_manifest["samples"].append({
                "sample": sample_output.name,
                "source_episode": str(episode),
                "source_episode_index": episode_index,
                "source_frame_count": len(frames),
                "files": files,
                "complete_rgb_frames": copied_all_frames,
                "uniform_preview_rgb_frames": copied_preview_frames,
            })

        with (seed_output / "sample_manifest.json").open("w") as handle:
            json.dump(seed_manifest, handle, indent=2, sort_keys=True)
        task_manifest["seeds"].append(seed_manifest)

    with (args.output_root / "sample_manifest.json").open("w") as handle:
        json.dump(task_manifest, handle, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output_root),
                      "seed_count": len(task_manifest["seeds"])}, sort_keys=True))


if __name__ == "__main__":
    main()
