#!/usr/bin/env python3
"""Multi-env smoke test for multi_env_camera.py + gt_pose_crop.py's
build_bimanual_boxes_all_envs -- gt_pose_crop_smoke.py (same directory)
already validated the single-env (task.envs[0]) camera+crop path; this
exercises the num_envs>1 generalization needed for online PPO
(compute_proprio_predtac_state in shadow_hand_pen.py), BEFORE bringing in
the IPC/tactile-server layer on top -- isolates rendering/projection
correctness across envs from everything else that could go wrong.

No trained policy, random actions per env, same reasoning as
gt_pose_crop_smoke.py. Dumps one raw + one box-overlay frame per env at the
LAST step only (not a step sweep like the single-env smoke test -- the
question here is "do all N envs' cameras/boxes look right", not "is a given
env's box stable over time", already answered by the single-env test).

Usage: python -m tactile_collection.gt_pose_crop_smoke_multienv \\
    --task ShadowHandPen --algo ppo --num_envs 4 --headless --test \\
    --seed 3204 --sim_device cuda:0 --rl_device cuda:0 \\
    --graphics_device_id 0 --pipeline cpu
"""
import json
import os

os.environ.setdefault("BIDEX_CAMERA_MODE", "chest")
os.environ.setdefault("BIDEX_CHEST_TARGET_MODE", "workspace")
os.environ.setdefault("BIDEX_CHEST_TARGET_CENTER", "bbox")
os.environ.setdefault("BIDEX_CHEST_TARGET_SMOOTHING", "0.0")
os.environ.setdefault("BIDEX_CHEST_EYE_OFFSET", "0.32,0.0,0.80")
os.environ.setdefault("BIDEX_CHEST_TARGET_OFFSET", "0.0,0.0,0.08")
os.environ.setdefault("BIDEX_HAND_COLOR_SAME", "1")

import cv2  # noqa: E402

from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed  # noqa: E402
from bidexhands.utils.parse_task import parse_task  # noqa: E402
from bidexhands.utils.process_marl import get_AgentIndex  # noqa: E402
import torch  # noqa: E402

from tactile_collection.multi_env_camera import apply_visual_style_all_envs, create_cameras, render_all_and_capture  # noqa: E402
from tactile_collection.gt_pose_crop import build_bimanual_boxes_all_envs  # noqa: E402


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
    apply_visual_style_all_envs(task)

    width = int(os.environ.get("BIDEX_VIDEO_WIDTH", "960"))
    height = int(os.environ.get("BIDEX_VIDEO_HEIGHT", "720"))
    out_dir = os.path.abspath(os.environ.get("GT_CROP_SMOKE_OUT", "gt_pose_crop_smoke_multienv_out"))
    os.makedirs(out_dir, exist_ok=True)
    n_steps = int(os.environ.get("GT_CROP_SMOKE_STEPS", "30"))
    action_scale = float(os.environ.get("GT_CROP_SMOKE_ACTION_SCALE", "0.5"))

    cameras, palm_handles, palm_name = create_cameras(task, width, height)
    print(f"[setup] task={args.task} num_envs={env.num_envs} camera_palm_body={palm_name} "
          f"size={width}x{height}", flush=True)

    env.reset()
    for step in range(n_steps):
        with torch.no_grad():
            actions = action_scale * (2.0 * torch.rand(env.num_envs, env.num_actions, device=env.rl_device) - 1.0)
            env.step(actions)

    # Position cameras (and compute their boxes from the resulting view/proj
    # matrices) BEFORE rendering -- render_all_and_capture must see the SAME
    # transform build_bimanual_boxes_all_envs just set, not whatever the
    # camera's previous/default transform was.
    sides_all_envs, eyes, targets = build_bimanual_boxes_all_envs(task, cameras, palm_handles, width, height)
    frames = render_all_and_capture(task, cameras, width, height)  # (num_envs, H, W, 3) RGB

    results = []
    for i in range(env.num_envs):
        frame_bgr = frames[i][:, :, ::-1].copy()
        raw_path = os.path.join(out_dir, f"env{i:02d}_raw.png")
        overlay_path = os.path.join(out_dir, f"env{i:02d}_gtbox.png")
        cv2.imwrite(raw_path, frame_bgr)
        cv2.imwrite(overlay_path, draw_boxes(frame_bgr, sides_all_envs[i]))
        record = {
            "env": i,
            "camera_eye": [float(v) for v in eyes[i]],
            "camera_target": [float(v) for v in targets[i]],
            "sides": {k: {"box": [float(c) for c in v["box"]], "actor": v["actor"]}
                      for k, v in sides_all_envs[i].items()},
        }
        results.append(record)
        box_str = " ".join(f"{k}_box={[round(c, 1) for c in v['box']]}" for k, v in sides_all_envs[i].items())
        print(f"[env {i}] sides={list(sides_all_envs[i].keys())} {box_str}", flush=True)

    summary_path = os.path.join(out_dir, "gt_boxes_multienv.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    n_with_both = sum(1 for r in results if len(r["sides"]) == 2)
    n_with_any = sum(1 for r in results if len(r["sides"]) >= 1)
    print(f"\n[summary] {env.num_envs} envs, {n_with_any} with >=1 hand box, "
          f"{n_with_both} with both hands boxed. Wrote {summary_path}", flush=True)
    print("GT_POSE_CROP_SMOKE_MULTIENV_DONE", flush=True)
    if n_with_any < env.num_envs:
        raise SystemExit(f"{env.num_envs - n_with_any}/{env.num_envs} envs had no hand project into frame "
                          f"-- per-env camera/crop generalization is broken for those envs")


if __name__ == "__main__":
    main()
