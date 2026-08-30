#!/usr/bin/env python3
"""BC distillation Stage 2 v0: train a student policy to imitate a bidexhands PPO
teacher's actions on its own already-collected successful-episode trajectories.

Two arms for this first pass (no v2-dit / predicted tactile yet -- that's Stage 2 v1):
  --tac_mode none : proprioception + vision           -> action
  --tac_mode gt   : proprioception + vision + GT tactile (pressure_grids.npz) -> action

Both arms are supervised ONLY at the timesteps where a real RGB frame was saved
(rgb_frames/*.png, stride-2 relative to the full per-step trajectory arrays) --
interior physics steps without a frame are dropped from training, not held or
interpolated. This keeps the two arms exactly comparable (same supervised
timesteps, same effective decision rate) and matches how eval-time rollout will
have to query the student anyway (hold the previous action between frames,
since that's the only cadence the policy has ever been trained at).

"Proprioception" here means the same fixed observation for both arms --
hand DOF positions (own + other hand), previous action targets, and
object/goal pose -- so the only thing that differs between arms is the
tactile channel. Adjust PROP_KEYS if a narrower/more realistic prop-only
definition is wanted later.

Episode selection is NOT done by this script -- it comes from a fixed
manifest JSON (see generate_episode_manifest.py), the single source of truth
for exactly which successful episodes go into train/val for this task. That's
required, not just convenient: each task's episodes are split across multiple
shard directories (one per collection seed, or several ad-hoc named batches
for Pen) with ids that restart at 0 within each shard, so "episode 3" alone
doesn't identify anything -- and reusing one manifest across all three arms
(P-only, P+GT-tac, P+Pred-tac) guarantees they all see identical episodes, so
an SR difference is never confounded by which episodes each arm trained on.

Data layout expected per manifest entry (already produced by
rollout_tactile_rgb_chest.py):
  <shard_dir>/successful_episodes/<episode>/
      rgb_frames/frame_000000.png ...
      trajectory_env0.npz   (actions, shadow_hand_dof_pos, cur_targets,
                              object_pose, goal_pose, frame_index, rgb_frame_step, ...)
      pressure_grids.npz    (left_pressure_grid, right_pressure_grid, both (T,21,21))

Not yet run anywhere -- local draft only.
"""
import os
import json
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


PROP_KEYS = [
    "shadow_hand_dof_pos", "shadow_hand_another_dof_pos",
    "cur_targets", "object_pose", "goal_pose",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_episode(ep_dir):
    traj = np.load(os.path.join(ep_dir, "trajectory_env0.npz"))
    press = np.load(os.path.join(ep_dir, "pressure_grids.npz"))
    return traj, press


def _episode_frame_rows(traj):
    """Row in the per-step arrays for each saved RGB frame (0..num_rgb-1)."""
    frame_index = traj["frame_index"]
    rgb_frame_step = traj["rgb_frame_step"]
    rows = np.searchsorted(frame_index, rgb_frame_step)
    assert np.all(frame_index[rows] == rgb_frame_step), \
        "frame_index/rgb_frame_step mismatch -- trajectory_env0.npz layout changed?"
    return rows


def _load_image(path, img_size):
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H,W,3) in [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)


class BCEpisodeDataset(Dataset):
    """One sample per (episode, saved RGB frame).

    `manifest_entries` is a list of {"shard_dir": ..., "episode": ...} dicts,
    e.g. manifest["train"] or manifest["val"] from generate_episode_manifest.py.
    """

    def __init__(self, manifest_entries, tac_mode, img_size=224):
        assert tac_mode in ("none", "gt")
        self.tac_mode = tac_mode
        self.img_size = img_size

        ep_dirs = [
            os.path.join(e["shard_dir"], "successful_episodes", e["episode"])
            for e in manifest_entries
        ]

        self.samples = []  # (ep_dir, frame_idx_in_episode, row_in_traj)
        for ep_dir in ep_dirs:
            traj, _ = _load_episode(ep_dir)
            rows = _episode_frame_rows(traj)
            n_png = len(glob.glob(os.path.join(ep_dir, "rgb_frames", "*.png")))
            n = min(len(rows), n_png)
            for f in range(n):
                self.samples.append((ep_dir, f, int(rows[f])))

        self._cache = {}  # ep_dir -> (traj, press); per-worker under DataLoader multiprocessing

    def _get(self, ep_dir):
        if ep_dir not in self._cache:
            self._cache[ep_dir] = _load_episode(ep_dir)
        return self._cache[ep_dir]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_dir, f, row = self.samples[idx]
        traj, press = self._get(ep_dir)

        img = _load_image(os.path.join(ep_dir, "rgb_frames", f"frame_{f:06d}.png"), self.img_size)

        prop = np.concatenate([
            np.asarray(traj[k][row], dtype=np.float32).reshape(-1)
            for k in PROP_KEYS if k in traj.files
        ])
        prop = torch.from_numpy(prop)

        action = torch.from_numpy(np.asarray(traj["actions"][row], dtype=np.float32))

        if self.tac_mode == "gt":
            left = np.asarray(press["left_pressure_grid"][row], dtype=np.float32).reshape(-1)
            right = np.asarray(press["right_pressure_grid"][row], dtype=np.float32).reshape(-1)
            tac = torch.from_numpy(np.concatenate([left, right]))
        else:
            tac = torch.zeros(0, dtype=torch.float32)

        return img, prop, tac, action


# ---------------------------------------------------------------------------
# Model: small from-scratch CNN vision encoder + prop MLP + (optional) tac MLP
# -> actor MLP. Frames are sim renders, not natural images, and this is meant
# to be bidexhands's own standalone student -- so a from-scratch CNN avoids
# both a questionable pretraining-domain match and a cross-repo dependency on
# Ego2Contact's DINOv2 stack.
# ---------------------------------------------------------------------------

class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()

        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.GroupNorm(8, cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.proj(self.net(x))


class BCStudent(nn.Module):
    def __init__(self, prop_dim, tac_dim, action_dim, emb_dim=256, hidden=(256, 256)):
        super().__init__()
        self.vis_enc = SmallCNNEncoder(out_dim=emb_dim)
        self.prop_enc = nn.Sequential(nn.Linear(prop_dim, emb_dim), nn.ReLU(inplace=True))
        self.tac_dim = tac_dim
        joint_dim = emb_dim * 2
        if tac_dim > 0:
            self.tac_enc = nn.Sequential(nn.Linear(tac_dim, emb_dim), nn.ReLU(inplace=True))
            joint_dim += emb_dim
        else:
            self.tac_enc = None

        layers, d = [], joint_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, action_dim), nn.Tanh()]  # teacher actions live in [-1, 1]
        self.actor = nn.Sequential(*layers)

    def forward(self, img, prop, tac):
        feats = [self.vis_enc(img), self.prop_enc(prop)]
        if self.tac_enc is not None:
            feats.append(self.tac_enc(tac))
        return self.actor(torch.cat(feats, dim=-1))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    for img, prop, tac, action in loader:
        img, prop, tac, action = (img.to(device), prop.to(device),
                                   tac.to(device), action.to(device))
        with torch.set_grad_enabled(train):
            pred = model(img, prop, tac)
            loss = nn.functional.mse_loss(pred, action)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * img.shape[0]
        n += img.shape[0]
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                     help="episode manifest JSON from generate_episode_manifest.py -- "
                          "the SAME file must be used for all three arms of one task")
    ap.add_argument("--tac_mode", choices=["none", "gt"], required=True)
    ap.add_argument("--out", required=True, help="checkpoint output dir")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                     help="weight init / dataloader shuffling only -- episode "
                          "selection comes from --manifest, not this")
    ap.add_argument("--ckpt_every", type=int, default=5,
                     help="save a numbered epoch_XXX.pt checkpoint every N epochs "
                          "(plus the last epoch) for sweep_eval_bc_student.py's "
                          "success-rate-vs-epoch curve")
    ap.add_argument("--wandb_project", default="ego2contact-bc-distill")
    ap.add_argument("--wandb_group", default=None,
                     help="defaults to the manifest's task name -- groups the three "
                          "arms of one task together in the wandb UI")
    ap.add_argument("--wandb_run_name", default=None,
                     help="defaults to '<task>_<tac_mode>'")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"[data] manifest task={manifest.get('task')} "
          f"{len(manifest['train'])} train episodes, {len(manifest['val'])} val episodes", flush=True)

    train_ds = BCEpisodeDataset(manifest["train"], args.tac_mode, args.img_size)
    val_ds = BCEpisodeDataset(manifest["val"], args.tac_mode, args.img_size)
    print(f"[data] {len(train_ds)} train samples, {len(val_ds)} val samples", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    sample_img, sample_prop, sample_tac, sample_action = train_ds[0]
    prop_dim, tac_dim, action_dim = (
        sample_prop.shape[0], sample_tac.shape[0], sample_action.shape[0]
    )
    print(f"[model] prop_dim={prop_dim} tac_dim={tac_dim} action_dim={action_dim}", flush=True)

    model = BCStudent(prop_dim, tac_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    task_name = manifest.get("task", "unknown_task")
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group or task_name,
                name=args.wandb_run_name or f"{task_name}_{args.tac_mode}",
                config={**vars(args), "task": task_name,
                        "prop_dim": prop_dim, "tac_dim": tac_dim, "action_dim": action_dim},
            )
            with open(os.path.join(args.out, "wandb_run_id.txt"), "w") as f:
                f.write(wandb_run.id)
        except ImportError:
            print("[wandb] not installed -- skipping logging", flush=True)

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[epoch {epoch:3d}] train_mse={train_loss:.5f}  val_mse={val_loss:.5f}", flush=True)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        ckpt = {
            "model": model.state_dict(),
            "prop_dim": prop_dim, "tac_dim": tac_dim, "action_dim": action_dim,
            "tac_mode": args.tac_mode, "args": vars(args),
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out, "best_model.pt"))
        torch.save(ckpt, os.path.join(args.out, "last_model.pt"))
        is_last = epoch == args.epochs - 1
        if is_last or (epoch + 1) % args.ckpt_every == 0:
            torch.save(ckpt, os.path.join(args.out, f"epoch_{epoch + 1:03d}.pt"))

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()  # sweep_eval_bc_student.py resumes this same run id later


if __name__ == "__main__":
    main()
