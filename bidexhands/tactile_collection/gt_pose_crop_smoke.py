#!/usr/bin/env python3
"""VISION-only smoke test for gt_pose_crop.py (same directory).

Boots one real bidexhands task (default ShadowHandPen, num_envs=1), steps it
with random actions so the hands leave their exact reset pose, positions the
SAME dynamic chest camera used during the real wilor_raw_rigid data
collection (BIDEX_CAMERA_MODE=chest / BIDEX_CHEST_TARGET_MODE=workspace, see
collect_wilor_raw_rigid.sh), computes a GT-pose hand crop box via
gt_pose_crop.build_bimanual_boxes_from_task, and saves both the raw frame and
a box-overlay frame to disk per captured step for visual inspection.

This is a geometry/rendering smoke test only -- it does NOT run SAM3 (SAM3's
modern-torch/transformers stack cannot coexist in this process with
bidexhands' pinned old-torch py38 IsaacGym env). A useful follow-up
cross-check, run separately in the touchanything conda env against the saved
frame_*_raw.png files, is to run SAM3 on them and compare its box to the
"sides" recorded in gt_boxes.json -- left for a second pass, not blocking
this one.

No trained policy is loaded -- random actions are enough to get the hands
into a non-degenerate pose for a crop sanity check; this does not exercise
success/reward, only rendering + geometry. Must run inside a SLURM GPU
allocation, bidexhands_isaacgym_py38 env, --pipeline cpu (the GPU rigid-body
PhysX pipeline crashes on H200, see vision_bidexhands_h200_smoke.py).

Usage (task/algo/etc. forwarded straight to bidexhands' own get_args()), run
from the bidexhands package root (bidex_root) so `tactile_collection` and
`bidexhands` both resolve, matching collect_wilor_raw_rigid.sh's PYTHONPATH:
  python -m tactile_collection.gt_pose_crop_smoke \\
      --task ShadowHandPen --algo ppo --num_envs 1 --headless --test \\
      --seed 3204 --sim_device cuda:0 --rl_device cuda:0 \\
      --graphics_device_id 0 --pipeline cpu
"""
import json
import os

# Match the real data-collection camera recipe exactly (collect_wilor_raw_rigid.sh /
# collect_shadow_hand_pen_5_success_wilor_raw_rigid.sh) unless the caller overrides.
os.environ.setdefault("BIDEX_CAMERA_MODE", "chest")
os.environ.setdefault("BIDEX_CHEST_TARGET_MODE", "workspace")
os.environ.setdefault("BIDEX_CHEST_TARGET_CENTER", "bbox")
os.environ.setdefault("BIDEX_CHEST_TARGET_SMOOTHING", "0.0")
os.environ.setdefault("BIDEX_CHEST_EYE_OFFSET", "0.32,0.0,0.80")
os.environ.setdefault("BIDEX_CHEST_TARGET_OFFSET", "0.0,0.0,0.08")

import cv2  # noqa: E402  (pure image I/O -- no torch-version conflict with the pinned py38 env)

# Isaac Gym must be imported before torch; the Bi-DexHands config import does that.
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed  # noqa: E402
from bidexhands.utils.parse_task import parse_task  # noqa: E402
from bidexhands.utils.process_marl import get_AgentIndex  # noqa: E402
import torch  # noqa: E402

from tactile_collection.rollout_tactile_rgb_chest import apply_visual_style, capture_frame, create_camera  # noqa: E402
from tactile_collection.gt_pose_crop import build_bimanual_boxes_from_task  # noqa: E402


def draw_boxes(frame_bgr, sides):
    out = frame_bgr.copy()
    colors = {"left": (255, 80, 80), "right": (80, 220, 80)}
    for side, info in sides.items():
        x1, y1, x2, y2 = [int(round(c)) for c in info["box"]]
        color = colors.get(side, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{side}/{info['actor']}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def main():
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    task, env = parse_task(args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))
    apply_visual_style(task)

    if env.num_envs != 1:
        raise ValueError(f"this smoke test requires --num_envs=1, got {env.num_envs}")

    width = int(os.environ.get("BIDEX_VIDEO_WIDTH", "960"))
    height = int(os.environ.get("BIDEX_VIDEO_HEIGHT", "720"))
    out_dir = os.path.abspath(os.environ.get("GT_CROP_SMOKE_OUT", "gt_pose_crop_smoke_out"))
    os.makedirs(out_dir, exist_ok=True)
    n_steps = int(os.environ.get("GT_CROP_SMOKE_STEPS", "30"))
    n_frames = int(os.environ.get("GT_CROP_SMOKE_FRAMES", "5"))
    action_scale = float(os.environ.get("GT_CROP_SMOKE_ACTION_SCALE", "0.5"))

    camera, camera_palm_handle, camera_palm_name = create_camera(task, width, height)
    print(f"[setup] task={args.task} camera_palm_body={camera_palm_name} "
          f"size={width}x{height} camera_mode={os.environ.get('BIDEX_CAMERA_MODE')} "
          f"eye_offset={os.environ.get('BIDEX_CHEST_EYE_OFFSET')} "
          f"target_offset={os.environ.get('BIDEX_CHEST_TARGET_OFFSET')}", flush=True)

    env.reset()
    capture_every = max(1, n_steps // max(1, n_frames))
    results = []
    for step in range(n_steps):
        with torch.no_grad():
            actions = action_scale * (2.0 * torch.rand(env.num_envs, env.num_actions, device=env.rl_device) - 1.0)
            env.step(actions)

        if step % capture_every != 0:
            continue

        sides, eye, target = build_bimanual_boxes_from_task(task, camera, camera_palm_handle, width, height)

        raw_path = os.path.join(out_dir, f"frame_{step:04d}_raw.png")
        capture_frame(task, camera, width, height, raw_path)
        frame_bgr = cv2.imread(raw_path)
        if frame_bgr is None:
            print(f"[warn] failed to reload saved frame {raw_path}", flush=True)
            continue
        overlay_path = os.path.join(out_dir, f"frame_{step:04d}_gtbox.png")
        cv2.imwrite(overlay_path, draw_boxes(frame_bgr, sides))

        record = {
            "step": step,
            "camera_eye": [float(v) for v in eye],
            "camera_target": [float(v) for v in target],
            "sides": {k: {"box": [float(c) for c in v["box"]], "actor": v["actor"]} for k, v in sides.items()},
            "raw_frame": raw_path,
            "overlay_frame": overlay_path,
        }
        results.append(record)
        box_str = " ".join(f"{k}_box={[round(c, 1) for c in v['box']]}" for k, v in sides.items())
        print(f"[step {step}] sides={list(sides.keys())} {box_str}", flush=True)

    summary_path = os.path.join(out_dir, "gt_boxes.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    n_with_both = sum(1 for r in results if len(r["sides"]) == 2)
    n_with_any = sum(1 for r in results if len(r["sides"]) >= 1)
    print(f"\n[summary] captured {len(results)} frames, {n_with_any} with >=1 hand box, "
          f"{n_with_both} with both hands boxed. Wrote {summary_path}", flush=True)
    print("GT_POSE_CROP_SMOKE_DONE", flush=True)
    if n_with_any == 0:
        raise SystemExit("no hand ever projected into frame -- check camera convention / hand actor names")


if __name__ == "__main__":
    main()
