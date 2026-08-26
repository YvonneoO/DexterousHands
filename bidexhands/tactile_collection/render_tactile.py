#!/usr/bin/env python3
"""Render bilateral EgoTouch-layout pressure grids with a task-aware title."""

import argparse
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pressure_npz")
    parser.add_argument("output_mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    task_name = os.environ.get("BIDEX_TACTILE_TASK", "").strip()
    rollout_title = "{} tactile rollout".format(task_name) if task_name else "Tactile rollout"
    data = np.load(args.pressure_npz)
    left = data["left_pressure_grid"]
    right = data["right_pressure_grid"]
    finite = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
    positive = finite[finite > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(vmax, 1.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_mp4)), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tactile_frames_") as frame_dir:
        frame_index = 0
        for step in range(0, left.shape[0], max(1, args.stride)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#101321")
            last_image = None
            for ax, grid, title in zip(axes, (left[step], right[step]), ("Left Hand", "Right Hand")):
                ax.set_facecolor("#101321")
                last_image = ax.imshow(grid, cmap="turbo", vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_title(title + " Pressure", color="white")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
            cbar.set_label("Pressure (Pa)", color="white")
            cbar.ax.tick_params(colors="white")
            fig.suptitle("{} — step {}".format(rollout_title, step), color="white")
            fig.savefig(
                os.path.join(frame_dir, "frame_{:06d}.png".format(frame_index)),
                dpi=120,
                facecolor=fig.get_facecolor(),
            )
            plt.close(fig)
            frame_index += 1
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", os.path.join(frame_dir, "frame_%06d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "20", args.output_mp4,
        ])
    print("Wrote {} (vmax {:.6g} Pa)".format(args.output_mp4, vmax))


if __name__ == "__main__":
    main()
