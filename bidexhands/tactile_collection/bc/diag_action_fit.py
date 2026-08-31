#!/usr/bin/env python3
"""Diagnostic: compare a trained BC student's predicted actions against the
true logged teacher actions on its OWN training samples (not live rollout).
No simulator needed. If predictions are close here but live rollout still
fails at SR=0, that points at the eval harness; if predictions are already
far off here, that points at data/capacity (BC didn't learn to predict
actions well, independent of any rollout mechanics).
"""
import json
import argparse

import numpy as np
import torch

from tactile_collection.bc.train_bc_student import BCStudent, BCEpisodeDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--n_samples", type=int, default=200)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tac_mode = ckpt["tac_mode"]
    img_size = ckpt["args"]["img_size"]

    ds = BCEpisodeDataset(manifest[args.split], tac_mode, img_size, tac_vmax=ckpt.get("tac_vmax"))
    model = BCStudent(ckpt["prop_dim"], ckpt["tac_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    n = min(args.n_samples, len(ds))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=n, replace=False)

    errs, pred_mags, true_mags = [], [], []
    all_preds = []
    with torch.no_grad():
        for i in idx:
            img, prop, tac, action = ds[i]
            pred = model(img.unsqueeze(0), prop.unsqueeze(0), tac.unsqueeze(0))[0]
            err = torch.nn.functional.mse_loss(pred, action).item()
            errs.append(err)
            pred_mags.append(pred.abs().mean().item())
            true_mags.append(action.abs().mean().item())
            all_preds.append(pred.numpy())

    errs = np.array(errs)
    all_preds = np.stack(all_preds)  # (n, action_dim)
    print(f"[diag] {args.split} split, {n} samples, tac_mode={tac_mode}")
    print(f"[diag] per-sample MSE: mean={errs.mean():.4f} median={np.median(errs):.4f} "
          f"min={errs.min():.4f} max={errs.max():.4f}")
    print(f"[diag] mean |pred action|: {np.mean(pred_mags):.4f}")
    print(f"[diag] mean |true action|: {np.mean(true_mags):.4f}")
    # per-dim std across the n predictions -- near-zero means the model is
    # outputting almost the SAME action regardless of input (collapse).
    print(f"[diag] mean per-action-dim std ACROSS samples: {all_preds.std(axis=0).mean():.4f}")

    print("[diag] example predictions vs true (first 6 action dims of first 3 samples):")
    with torch.no_grad():
        for i in idx[:3]:
            img, prop, tac, action = ds[i]
            pred = model(img.unsqueeze(0), prop.unsqueeze(0), tac.unsqueeze(0))[0]
            print("  pred:", np.round(pred.numpy()[:6], 3))
            print("  true:", np.round(action.numpy()[:6], 3))


if __name__ == "__main__":
    main()
