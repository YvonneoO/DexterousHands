#!/usr/bin/env python3
"""Stage 2 v0 student eval: roll out a trained BC student (train_bc_student.py)
in the live bidexhands sim and report success rate.

Run as (mirrors rollout_tactile_rgb_chest.py's own invocation convention):
    cd <DexterousHands root>/bidexhands
    python -m tactile_collection.bc.eval_bc_student \
        --ckpt <student.pt> --episodes 100 -- --task ShadowHandPen --algo ppo

Everything after the (optional) "--" or any flag this script doesn't
recognize is forwarded to bidexhands' own arg parser (get_args/load_cfg/
parse_task), which is what actually needs --task, --cfg_train, etc. -- same
convention rollout_tactile_rgb_chest.py's collection runs used.

Camera positioning, GT-tactile projection, and the goal-hiding render helpers
are imported directly from rollout_tactile_rgb_chest.py rather than
reimplemented, so eval-time inputs match training-time inputs exactly (same
camera formula, same taxel projection). Camera offsets default to the
Pen/Scissors collection-time values (CAMERA_EYE_OFFSET=0.32,0.0,0.80 /
CAMERA_TARGET_OFFSET=0.0,0.0,0.08); pass --camera_eye_offset/
--camera_target_offset explicitly for a task collected with different values
(e.g. Door: 0.38,0.0,0.82).

The student is queried only every --frame_stride physics steps (must match
whatever the checkpoint was trained on -- see train_bc_student.py), holding
the previous action between queries via zero-order hold: that's the only
cadence the policy has ever seen supervision at. Camera repositioning is
likewise gated on the same stride, matching rollout_tactile_rgb_chest.py's
own loop exactly (not repositioned every physics step).

Not yet run anywhere -- local draft only.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from PIL import Image

# Isaac Gym must be imported before torch; bidexhands.utils.config does that.
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex

from tactile_collection.egotouch_taxels import EgoTouchTaxelMapper
from tactile_collection.rollout_tactile_rgb_chest import (
    as_numpy, env0, native_success, create_camera, position_camera,
    hide_goal_object_for_render, restore_goal_object_after_render,
)
from tactile_collection.bc.train_bc_student import BCStudent, PROP_KEYS


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student checkpoint from train_bc_student.py")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max_steps_per_episode", type=int, default=500)
    ap.add_argument("--frame_stride", type=int, default=2,
                     help="must match BIDEX_FRAME_STRIDE used for the training data")
    ap.add_argument("--camera_eye_offset", default="0.32,0.0,0.80")
    ap.add_argument("--camera_target_offset", default="0.0,0.0,0.08")
    ap.add_argument("--asset_dir", default="tactile_collection/assets")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None, help="optional json path for per-episode results")
    args, unknown = ap.parse_known_args()

    # bidexhands' own arg parser (--task, --algo, --cfg_train, ...) consumes
    # whatever's left -- forward it exactly as rollout_tactile_rgb_chest.py's
    # own invocation would pass it.
    sys.argv = [sys.argv[0]] + unknown

    os.environ["BIDEX_CAMERA_MODE"] = "chest"
    os.environ["BIDEX_CHEST_TARGET_MODE"] = "workspace"
    os.environ["BIDEX_CHEST_TARGET_CENTER"] = "bbox"
    os.environ["BIDEX_CHEST_TARGET_SMOOTHING"] = "0.0"
    os.environ["BIDEX_CHEST_EYE_OFFSET"] = args.camera_eye_offset
    os.environ["BIDEX_CHEST_TARGET_OFFSET"] = args.camera_target_offset

    set_np_formatting()
    bidex_args = get_args()
    cfg, cfg_train, logdir = load_cfg(bidex_args)
    sim_params = parse_sim_params(bidex_args, cfg, cfg_train)
    set_seed(args.seed, cfg_train.get("torch_deterministic", False))
    task, env = parse_task(bidex_args, cfg, cfg_train, sim_params, get_AgentIndex(cfg))

    if env.num_envs != 1:
        raise ValueError("Student eval currently requires --num_envs=1; got {}".format(env.num_envs))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = BCStudent(ckpt["prop_dim"], ckpt["tac_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tac_mode = ckpt["tac_mode"]
    img_size = ckpt["args"]["img_size"]
    print(f"[ckpt] tac_mode={tac_mode} prop_dim={ckpt['prop_dim']} "
          f"tac_dim={ckpt['tac_dim']} action_dim={ckpt['action_dim']} img_size={img_size}", flush=True)

    camera, camera_palm_handle, camera_palm_name = create_camera(task, 640, 480)

    left_mapper = right_mapper = net_contact_forces = None
    if tac_mode == "gt":
        from isaacgym import gymtorch
        left_mapper = EgoTouchTaxelMapper(
            task.gym, task.envs[0], "another_hand", "left",
            os.path.join(args.asset_dir, "pressure_position_mapping_left.json"),
        )
        right_mapper = EgoTouchTaxelMapper(
            task.gym, task.envs[0], "hand", "right",
            os.path.join(args.asset_dir, "pressure_position_mapping_right.json"),
        )
        net_contact_force_tensor = task.gym.acquire_net_contact_force_tensor(task.sim)
        net_contact_forces = gymtorch.wrap_tensor(net_contact_force_tensor).view(env.num_envs, -1, 3)

    obs = env.reset()
    zero_action = torch.zeros(1, ckpt["action_dim"], device=device)
    last_action = zero_action.clone()

    results = []
    episode_steps = 0
    successes = 0
    completed = 0

    while completed < args.episodes:
        if episode_steps % args.frame_stride == 0:
            position_camera(task, camera, camera_palm_handle)
            frame = capture_frame_array(task, camera, 640, 480)
            img_t = image_to_tensor(frame, img_size, device)
            prop_t = prop_from_task(task, env.num_envs, device)

            if tac_mode == "gt":
                task.gym.refresh_net_contact_force_tensor(task.sim)
                env_forces = as_numpy(net_contact_forces[0])
                left_pa, _, _ = left_mapper.project_net_forces(env_forces)
                right_pa, _, _ = right_mapper.project_net_forces(env_forces)
                tac = np.concatenate([left_pa.astype(np.float32).reshape(-1),
                                       right_pa.astype(np.float32).reshape(-1)])
                tac_t = torch.from_numpy(tac).unsqueeze(0).to(device)
            else:
                tac_t = torch.zeros(1, 0, device=device)

            with torch.no_grad():
                last_action = model(img_t, prop_t, tac_t)

        with torch.no_grad():
            next_obs, rew, done, infos = env.step(last_action)
            obs.copy_(next_obs)

        episode_steps += 1
        done_np = as_numpy(done).astype(bool)
        success_np = native_success(task, infos)

        if bool(done_np[0]) or episode_steps >= args.max_steps_per_episode:
            completed += 1
            success = bool(success_np is not None and success_np[0] > 0)
            successes += int(success)
            results.append({"episode": completed, "steps": episode_steps, "success": success})
            print(f"[eval] episode {completed}/{args.episodes}  steps={episode_steps}  "
                  f"success={success}  running_SR={successes / completed:.3f}", flush=True)
            episode_steps = 0
            last_action = zero_action.clone()
            obs = env.reset()

    sr = successes / max(completed, 1)
    print(f"[eval] DONE  episodes={completed}  successes={successes}  SR={sr:.4f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "tac_mode": tac_mode, "episodes": completed,
                       "successes": successes, "success_rate": sr, "per_episode": results}, f, indent=2)


if __name__ == "__main__":
    main()
