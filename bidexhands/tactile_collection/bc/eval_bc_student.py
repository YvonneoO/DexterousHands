#!/usr/bin/env python3
"""Stage 2 v0 student eval: roll out a trained BC student (train_bc_student.py)
in the live bidexhands sim and report a trustworthy final success rate (large
episode count). For a success-rate-vs-epoch curve during training, use
sweep_eval_bc_student.py instead -- this script is for a single checkpoint.

Run as (mirrors rollout_tactile_rgb_chest.py's own invocation convention):
    cd <DexterousHands root>/bidexhands
    python -m tactile_collection.bc.eval_bc_student \
        --ckpt <student.pt> --episodes 100 -- --task ShadowHandPen --algo ppo

Everything after the (optional) "--" or any flag this script doesn't
recognize is forwarded to bidexhands' own arg parser (get_args/load_cfg/
parse_task), which is what actually needs --task, --cfg_train, etc. -- same
convention rollout_tactile_rgb_chest.py's collection runs used.

Camera offsets default to the Pen/Scissors collection-time values
(CAMERA_EYE_OFFSET=0.32,0.0,0.80 / CAMERA_TARGET_OFFSET=0.0,0.0,0.08); pass
--camera_eye_offset/--camera_target_offset explicitly for a task collected
with different values (e.g. Door: 0.38,0.0,0.82). Rollout logic itself lives
in rollout_eval_core.py.

Not yet run anywhere -- local draft only.
"""
import argparse
import json

# rollout_eval_core pulls in isaacgym via bidexhands.utils.config; that must
# happen before `import torch` runs anywhere in the process, so it's imported
# first here even though torch is used below.
from tactile_collection.bc.rollout_eval_core import build_eval_context, run_rollout_eval

import torch

from tactile_collection.bc.train_bc_student import BCStudent


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
    # parse_known_args leaves a literal "--" separator in `unknown` if the
    # caller passed one to mark where bidexhands' own args start; bidexhands'
    # parser doesn't handle a bare "--" and rejects everything after it.
    if unknown and unknown[0] == "--":
        unknown = unknown[1:]

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tac_mode = ckpt["tac_mode"]
    img_size = ckpt["args"]["img_size"]

    ctx = build_eval_context(
        unknown, args.seed, args.camera_eye_offset, args.camera_target_offset,
        tac_mode, args.asset_dir, args.frame_stride, args.max_steps_per_episode,
    )
    model = BCStudent(ckpt["prop_dim"], ckpt["tac_dim"], ckpt["action_dim"]).to(ctx.device)
    model.load_state_dict(ckpt["model"])
    print(f"[ckpt] tac_mode={tac_mode} prop_dim={ckpt['prop_dim']} "
          f"tac_dim={ckpt['tac_dim']} action_dim={ckpt['action_dim']} img_size={img_size}", flush=True)

    sr, results = run_rollout_eval(ctx, model, tac_mode, img_size, ckpt["action_dim"], args.episodes,
                                    tac_vmax=ckpt.get("tac_vmax"))
    for r in results:
        print(f"[eval] episode {r['episode']}/{args.episodes}  steps={r['steps']}  "
              f"success={r['success']}", flush=True)
    print(f"[eval] DONE  episodes={len(results)}  SR={sr:.4f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "tac_mode": tac_mode, "episodes": len(results),
                       "success_rate": sr, "per_episode": results}, f, indent=2)


if __name__ == "__main__":
    main()
