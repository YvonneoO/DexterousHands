#!/usr/bin/env python3
"""BC distillation Stage 2: train a student policy to imitate a bidexhands PPO
teacher's actions on its own already-collected successful-episode trajectories.

Three arms:
  --tac_mode none : proprioception + vision                              -> action
  --tac_mode gt   : proprioception + vision + GT tactile (pressure_grids.npz)      -> action
  --tac_mode pred : proprioception + vision + PREDICTED tactile (pred_pressure_grids.npz,
                    written offline by Ego2Contact's scripts/sim_data_collection/dexteroushands/
                    infer_pred_tactile_offline.py -- a v2-dit sim-finetuned forward pass over the
                    episode's own rgb_frames/*.png) -> action
  tac_mode="pred" is OFFLINE-generation only here (a pre-built file, exactly like "gt"'s
  pressure_grids.npz) -- LIVE rollout eval (rollout_eval_core.py) does NOT support it yet, since
  that would need a live SAM3+WiLoR+DINO+v2-dit forward pass inside the sim step loop; out of
  scope for now (see rollout_eval_core.py's run_rollout_eval TODO).

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

TAC_FILE_BY_MODE = {"gt": "pressure_grids.npz", "pred": "pred_pressure_grids.npz"}


def _load_episode(ep_dir, tac_file="pressure_grids.npz"):
    traj = np.load(os.path.join(ep_dir, "trajectory_env0.npz"))
    press = np.load(os.path.join(ep_dir, tac_file))
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


def compute_tac_vmax(manifest_entries, tac_file="pressure_grids.npz", percentile=99.0):
    """99th-percentile scale for tactile normalization, computed from ONE task's
    own training split (raw pressure reaches into the tens/hundreds of
    thousands -- confirmed empirically for Pen GT: max ~222,724, nonzero mean
    ~11,000 -- five to six orders of magnitude larger than vision ([0,1]) or
    proprioception, which stalls training). Percentile-based (not z-score)
    because the data is non-negative, sparse (~93% zero), and heavy-tailed --
    matches the vmax convention already used elsewhere in this project's
    other tactile-prediction pipeline, not a new scheme. Computed per-task,
    NOT globally: different tasks' contact dynamics give different scales,
    and reusing one task's vmax on another would just reintroduce the same
    kind of scale mismatch this fixes.

    `tac_file` selects which per-episode npz to read -- "pressure_grids.npz"
    (GT, default) or "pred_pressure_grids.npz" (offline v2-dit predictions;
    see TAC_FILE_BY_MODE). tac_mode="pred" computes its OWN vmax from the
    predicted training values (not GT's), since the model's own output scale
    need not exactly match GT's -- same code path, just pointed at the other
    file, matching this function's SAME contract (both files share the
    left_pressure_grid/right_pressure_grid keys and units).
    """
    values = []
    for e in manifest_entries:
        ep_dir = os.path.join(e["shard_dir"], "successful_episodes", e["episode"])
        _, press = _load_episode(ep_dir, tac_file)
        left = np.nan_to_num(np.asarray(press["left_pressure_grid"], dtype=np.float32))
        right = np.nan_to_num(np.asarray(press["right_pressure_grid"], dtype=np.float32))
        values.append(left.reshape(-1))
        values.append(right.reshape(-1))
    all_values = np.concatenate(values)
    vmax = float(np.percentile(all_values, percentile))
    return max(vmax, 1e-6)  # guard against a degenerate all-zero episode set


class BCEpisodeDataset(Dataset):
    """One sample per (episode, saved RGB frame).

    `manifest_entries` is a list of {"shard_dir": ..., "episode": ...} dicts,
    e.g. manifest["train"] or manifest["val"] from generate_episode_manifest.py.
    `tac_vmax`: required when tac_mode in ("gt", "pred") -- see compute_tac_vmax().
    tac_mode="pred" reads pred_pressure_grids.npz (offline-generated by Ego2Contact's
    infer_pred_tactile_offline.py) instead of pressure_grids.npz -- see TAC_FILE_BY_MODE
    -- through the exact same NaN-sanitize + vmax-clip code path as "gt" below.
    """

    def __init__(self, manifest_entries, tac_mode, img_size=224, tac_vmax=None):
        assert tac_mode in ("none", "gt", "pred")
        if tac_mode in ("gt", "pred"):
            assert tac_vmax is not None, f"tac_vmax is required when tac_mode='{tac_mode}'"
        self.tac_mode = tac_mode
        self.tac_file = TAC_FILE_BY_MODE.get(tac_mode)
        self.img_size = img_size
        self.tac_vmax = tac_vmax

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
            # tac_mode="none" has no tac_file (unused below) -- default to the always-present
            # GT file so this load never errors regardless of tac_mode.
            self._cache[ep_dir] = _load_episode(ep_dir, self.tac_file or "pressure_grids.npz")
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

        if self.tac_mode in ("gt", "pred"):
            # left/right_pressure_grid use NaN to mark unmapped/non-sensor cells
            # (distinct from a real zero-contact reading) -- must be sanitized
            # before hitting the network, or it poisons the whole forward pass.
            # (pred_pressure_grids.npz uses 0.0, not NaN, for its unused/non-RGB-frame
            # rows -- nan_to_num is a no-op there, kept for a single shared code path.)
            left = np.nan_to_num(
                np.asarray(press["left_pressure_grid"][row], dtype=np.float32), nan=0.0
            ).reshape(-1)
            right = np.nan_to_num(
                np.asarray(press["right_pressure_grid"][row], dtype=np.float32), nan=0.0
            ).reshape(-1)
            tac_raw = np.concatenate([left, right])
            # raw pressure reaches into the tens/hundreds of thousands -- scale
            # to [0,1] against this task's own training-set vmax (see
            # compute_tac_vmax) so it doesn't dwarf vision/proprioception.
            tac = torch.from_numpy(np.clip(tac_raw / self.tac_vmax, 0.0, 1.0).astype(np.float32))
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
        # No final activation: act_inference() returns the RAW, UNCLIPPED
        # Gaussian policy mean (confirmed from real logged actions exceeding
        # +-4 in magnitude) -- the env clips to its Box(-1,1) action space
        # internally at step() time, the logged/target actions are not
        # pre-clipped. A Tanh here would structurally cap predictions at
        # +-1, guaranteeing it can never match targets beyond that range.
        layers += [nn.Linear(d, action_dim)]
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
    ap.add_argument("--tac_mode", choices=["none", "gt", "pred"], required=True)
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

    tac_vmax = None
    if args.tac_mode in ("gt", "pred"):
        tac_vmax = compute_tac_vmax(manifest["train"], tac_file=TAC_FILE_BY_MODE[args.tac_mode])
        print(f"[data] tac_vmax (99th pct, this task's train split, {TAC_FILE_BY_MODE[args.tac_mode]}) "
              f"= {tac_vmax:.2f}", flush=True)

    train_ds = BCEpisodeDataset(manifest["train"], args.tac_mode, args.img_size, tac_vmax)
    val_ds = BCEpisodeDataset(manifest["val"], args.tac_mode, args.img_size, tac_vmax)
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
            "tac_mode": args.tac_mode, "tac_vmax": tac_vmax, "args": vars(args),
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
