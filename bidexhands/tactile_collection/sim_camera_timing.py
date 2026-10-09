#!/usr/bin/env python3
"""Sim-side cost of N per-env cameras for the online Pred-Tac loop (ONE sim GPU, N envs). Per tick, times each piece
of what compute_proprio_predtac_state does today and of the cheaper "downsized frame + full-res crops" payload:

  phys      env.step (N envs, random actions)
  pos_box   camera placement + GT-pose box projection (build_bimanual_boxes_all_envs, minus its internal render)
  render    step_graphics + render_all_camera_sensors (all N cameras in one call)
  readback  get_camera_image per env -> (N,H,W,3) uint8 (the python loop in multi_env_camera.render_all_and_capture)
  resize    cv2.resize of every frame to --small (the downsized full frame the DINO branch needs)
  crop      one warpAffine crop (256x256, box x rescale 2.0, like WiLoR's ViTDetDataset) per hand FROM THE FULL-RES FRAME
  ipc_full  np.savez of the full-res request (what predtac_ipc.write_request writes today)   [+ read-back time]
  ipc_small np.savez of {small frame, 2 crops, boxes, has_hand}                                [+ read-back time]

  python -m tactile_collection.sim_camera_timing --task ShadowHandPen --algo ppo --num_envs 64 --headless --test \
      --seed 3204 --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 --pipeline cpu
Env: BIDEX_VIDEO_WIDTH/HEIGHT (default 960x720), TIMING_TICKS (8), TIMING_WARMUP (3), TIMING_SMALL (224), TIMING_IPC_DIR.
"""
import os
import time

os.environ.setdefault("BIDEX_CAMERA_MODE", "chest")
os.environ.setdefault("BIDEX_CHEST_TARGET_MODE", "workspace")
os.environ.setdefault("BIDEX_CHEST_TARGET_CENTER", "bbox")
os.environ.setdefault("BIDEX_CHEST_TARGET_SMOOTHING", "0.0")
os.environ.setdefault("BIDEX_CHEST_EYE_OFFSET", "0.32,0.0,0.80")
os.environ.setdefault("BIDEX_CHEST_TARGET_OFFSET", "0.0,0.0,0.08")
os.environ.setdefault("BIDEX_HAND_COLOR_SAME", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed  # noqa: E402
from bidexhands.utils.parse_task import parse_task  # noqa: E402
from bidexhands.utils.process_marl import get_AgentIndex  # noqa: E402
import torch  # noqa: E402  (must come AFTER the isaacgym-importing bidexhands modules)

from tactile_collection.multi_env_camera import apply_visual_style_all_envs, create_cameras  # noqa: E402
from tactile_collection.gt_pose_crop import build_bimanual_boxes_all_envs  # noqa: E402


def crop_box(frame, box, out=256, rescale=2.0):
    """Square crop around the box centre, side = max(w,h) * rescale, warped to out x out (CPU, like ViTDetDataset)."""
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    s = max(x2 - x1, y2 - y1) * rescale
    sc = out / max(s, 1.0)
    M = np.array([[sc, 0, out / 2 - sc * cx], [0, sc, out / 2 - sc * cy]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (out, out), flags=cv2.INTER_LINEAR)


def main():
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    task, env = parse_task(args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))
    apply_visual_style_all_envs(task)
    W = int(os.environ.get("BIDEX_VIDEO_WIDTH", "960"))
    H = int(os.environ.get("BIDEX_VIDEO_HEIGHT", "720"))
    ticks = int(os.environ.get("TIMING_TICKS", "8"))
    warm = int(os.environ.get("TIMING_WARMUP", "3"))
    small = int(os.environ.get("TIMING_SMALL", "224"))
    ipc_dir = os.environ.get("TIMING_IPC_DIR", "/scratch/project/prj-02-phai-lab/yqq/ipc/timing_bench")
    os.makedirs(ipc_dir, exist_ok=True)
    N = env.num_envs
    from isaacgym import gymapi

    cameras, palm_handles, _ = create_cameras(task, W, H)
    env.reset()
    rows = []
    for tick in range(warm + ticks):
        tm = {}
        t0 = time.time()
        with torch.no_grad():
            actions = 0.5 * (2.0 * torch.rand(N, env.num_actions, device=env.rl_device) - 1.0)
            env.step(actions)
        tm["phys"] = time.time() - t0

        t0 = time.time()
        sides, _, _ = build_bimanual_boxes_all_envs(task, cameras, palm_handles, W, H)
        t_build = time.time() - t0

        t0 = time.time()
        task.gym.fetch_results(task.sim, True)
        task.gym.step_graphics(task.sim)
        task.gym.render_all_camera_sensors(task.sim)
        tm["render"] = time.time() - t0
        tm["pos_box"] = max(t_build - tm["render"], 0.0)

        t0 = time.time()
        frames = np.empty((N, H, W, 3), dtype=np.uint8)
        for i, cam in enumerate(cameras):
            rgba = np.asarray(task.gym.get_camera_image(task.sim, task.envs[i], cam, gymapi.IMAGE_COLOR),
                              dtype=np.uint8).reshape(H, W, 4)
            frames[i] = rgba[:, :, :3]
        tm["readback"] = time.time() - t0

        t0 = time.time()
        small_frames = np.stack([cv2.resize(frames[i], (small, small), interpolation=cv2.INTER_AREA) for i in range(N)])
        tm["resize"] = time.time() - t0

        t0 = time.time()
        crops = np.zeros((N, 2, 256, 256, 3), dtype=np.uint8)
        boxes = np.full((N, 2, 4), np.nan, dtype=np.float32)
        has = np.zeros((N, 2), dtype=bool)
        for i in range(N):
            for h, side in enumerate(("left", "right")):
                if side in sides[i]:
                    b = sides[i][side]["box"]
                    boxes[i, h] = b
                    has[i, h] = True
                    crops[i, h] = crop_box(frames[i], b)
        tm["crop"] = time.time() - t0

        for name, payload in (("full", dict(frames=frames, boxes=boxes, has_hand=has)),
                              ("small", dict(frames=small_frames, crops=crops, boxes=boxes, has_hand=has))):
            p = os.path.join(ipc_dir, f"req_{name}_{os.getpid()}")
            t0 = time.time()
            np.savez(p, **payload)
            tm[f"ipc_{name}_w"] = time.time() - t0
            t0 = time.time()
            with np.load(p + ".npz") as d:
                _ = {k: d[k] for k in d.files}
            tm[f"ipc_{name}_r"] = time.time() - t0
            tm[f"mb_{name}"] = os.path.getsize(p + ".npz") / 1e6
            os.remove(p + ".npz")
        if tick >= warm:
            rows.append(tm)
        print(f"[tick {tick}] " + " ".join(f"{k}={v * 1e3:.0f}ms" if not k.startswith("mb") else f"{k}={v:.0f}MB"
                                            for k, v in tm.items()), flush=True)

    keys = list(rows[0].keys())
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print(f"\nSUMMARY N={N} res={W}x{H} small={small} ticks={len(rows)}", flush=True)
    for k in keys:
        if k.startswith("mb"):
            print(f"  {k:12s} {mean[k]:9.1f} MB   ({mean[k] / N:.2f} MB/env)", flush=True)
        else:
            print(f"  {k:12s} {mean[k] * 1e3:9.0f} ms  ({mean[k] * 1e3 / N:.2f} ms/env)", flush=True)
    tot_full = sum(mean[k] for k in ("pos_box", "render", "readback", "ipc_full_w"))
    tot_small = sum(mean[k] for k in ("pos_box", "render", "readback", "resize", "crop", "ipc_small_w"))
    print(f"  camera+IPC path per tick: full={tot_full * 1e3:.0f} ms  small+crops={tot_small * 1e3:.0f} ms "
          f"(phys={mean['phys'] * 1e3:.0f} ms)", flush=True)
    print("SIM_CAMERA_TIMING_DONE", flush=True)


if __name__ == "__main__":
    main()
