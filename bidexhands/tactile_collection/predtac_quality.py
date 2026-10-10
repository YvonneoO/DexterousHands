"""Online Pred-Tac input-quality monitor (env PREDTAC_QUALITY_LOG=1): compares the tactile the policy is actually fed (the
predictor's latest response) with the simulator's ground-truth per-link contact, in the same envs at the same time, and
estimates the effective time lag of the response.

Ground truth: per hand, per link force magnitude from vec_sensor_tensor (same extraction as the noisy-GT path), palm /
lfmetacarpal overlap handled as in noisy_tactile.py, converted to the predictor's normalised pressure g = clamp(F * scale):
scale: analytic |F|/(W*L)/PREDTAC_QUALITY_VMAX (vmax = the served predictor's per-task normaliser), or "scale"[17] of a
PREDTAC_QUALITY_PARAMS json (calibrate_noisy_tactile.py) times PREDTAC_QUALITY_VMAX_RATIO.

Every PREDTAC_QUALITY_EVERY (200) env steps one line is printed, over that window, with the current response (c, b) compared to
the GT g of k env steps ago for k = 0..K:
  corr[k]  Pearson correlation of c with g_(t-k) over all (env, hand, link) entries
  best_k   the k with the highest correlation = effective lag of the tactile in the observation, in env steps
  tpr/fpr  of the binary channel against 1[g_(t-k) > thr], at k = 0 and at best_k
  gt_pos / pred_pos  rate of contact in GT and in the prediction
"""
import json
import os
from collections import deque

import torch

from tactile_collection.noisy_tactile import LINK_NAMES, NUM_LINKS, PALM_IDX, LFMETA_IDX, PALM_OVERLAP_RATIO

K = 10


class QualityMonitor:
    def __init__(self, task):
        self.task = task
        self.device = task.device
        params = {}
        path = os.environ.get("PREDTAC_QUALITY_PARAMS") or os.environ.get("NOISY_TACTILE_PARAMS")
        if path:
            with open(path) as f:
                params = json.load(f)
        if "scale" in params:
            scale = torch.as_tensor(params["scale"], dtype=torch.float32, device=self.device).reshape(NUM_LINKS)
            scale = scale * float(os.environ.get("PREDTAC_QUALITY_VMAX_RATIO", "1.0"))
        else:
            # analytic force (N) -> normalised pooled pressure: |F| / (W*L) / vmax, vmax = the served predictor's per-task
            # normaliser (calibrate_noisy_tactile.py's analytic_scale; palm / lfmetacarpal share the palm area)
            vmax = float(os.environ.get("PREDTAC_QUALITY_VMAX", "0"))
            assert vmax > 0, "set PREDTAC_QUALITY_VMAX (the task's pressure normaliser) or PREDTAC_QUALITY_PARAMS"
            WL = {"distal": 3.666e-4, "middle": 4.025e-4, "proximal": 9.0e-4,
                  "thdistal": 5.049e-4, "thmiddle": 7.04e-4, "thproximal": 9.88e-4}
            palm_area = 0.064 * 0.098 + 0.022 * 0.050
            sc = []
            for n in LINK_NAMES:
                a = palm_area if n in ("palm", "lfmetacarpal") else (WL[n] if n.startswith("th") else WL[n[2:]])
                sc.append(1.0 / (a * vmax))
            scale = torch.as_tensor(sc, dtype=torch.float32, device=self.device)
        self.scale = scale
        self.thr = float(params.get("threshold", 0.1))
        gt_names = [n.split(":")[1] for n in task.fingertips] + [n.split(":")[1] for n in task.tactile_extra_links]
        self.perm = torch.as_tensor([gt_names.index(n) for n in LINK_NAMES], device=self.device, dtype=torch.long)
        self.every = int(os.environ.get("PREDTAC_QUALITY_EVERY", "200"))
        self.hist = deque(maxlen=K + 1)
        self.steps = 0
        self._reset_acc()
        print(f"[predtac][quality] monitor on: params={path} thr={self.thr} scale[0]={self.scale[0].item():.4g} "
              f"every={self.every} steps, K={K}", flush=True)

    def _reset_acc(self):
        z = lambda: torch.zeros(K + 1, dtype=torch.float64, device=self.device)
        self.n, self.sx, self.sy, self.sxx, self.syy, self.sxy = z(), z(), z(), z(), z(), z()
        self.tp, self.pos, self.fp, self.neg = z(), z(), z(), z()
        self.gt_pos = torch.zeros((), dtype=torch.float64, device=self.device)
        self.pred_pos = torch.zeros_like(self.gt_pos)
        self.cnt = 0

    def _gt(self):
        t, n = self.task, self.task.num_envs

        def hand_forces(off):
            tips = t.vec_sensor_tensor[:, off:off + 30].view(n, len(t.fingertips), 6)[:, :, :3]
            extra = t.vec_sensor_tensor[:, off + 30:off + 102].view(n, t.num_tactile_extra, 6)[:, :, :3]
            return torch.cat([torch.norm(tips, dim=-1), torch.norm(extra, dim=-1)], dim=-1)[:, self.perm]

        force = torch.stack([hand_forces(102), hand_forces(0)], dim=1).clone()      # (N, 2, 17): left, right
        combined = force[..., PALM_IDX] + PALM_OVERLAP_RATIO * force[..., LFMETA_IDX]
        force[..., PALM_IDX] = combined
        force[..., LFMETA_IDX] = combined
        return (force * self.scale).clamp(0.0, 1.0)

    @torch.no_grad()
    def update(self, cont_np, bin_np, stale_ticks, capture_every):
        g = self._gt()
        self.hist.append(g)
        c = torch.as_tensor(cont_np, device=self.device, dtype=torch.float32)
        b = torch.as_tensor(bin_np, device=self.device, dtype=torch.float32)
        self.steps += 1
        if float(c.abs().sum()) == 0.0 and float(b.sum()) == 0.0:
            return                                              # no response yet (or none non-zero): nothing to compare
        self.cnt += 1
        self.gt_pos += (g > self.thr).double().mean()
        self.pred_pos += (b > 0.5).double().mean()
        cf = c.flatten().double()
        for k in range(len(self.hist)):
            gk = self.hist[-1 - k]
            y = gk.flatten().double()
            self.n[k] += y.numel()
            self.sx[k] += cf.sum(); self.sy[k] += y.sum()
            self.sxx[k] += (cf * cf).sum(); self.syy[k] += (y * y).sum(); self.sxy[k] += (cf * y).sum()
            gpos = gk > self.thr
            self.pos[k] += gpos.sum(); self.neg[k] += (~gpos).sum()
            self.tp[k] += ((b > 0.5) & gpos).sum(); self.fp[k] += ((b > 0.5) & ~gpos).sum()
        if self.cnt % self.every == 0:
            self._report(stale_ticks, capture_every)
            self._reset_acc()

    def _report(self, stale_ticks, capture_every):
        n = self.n.clamp_min(1)
        mx, my = self.sx / n, self.sy / n
        cov = self.sxy / n - mx * my
        var = (self.sxx / n - mx * mx).clamp_min(1e-12) * (self.syy / n - my * my).clamp_min(1e-12)
        corr = (cov / var.sqrt()).cpu().numpy()
        valid = self.n.cpu().numpy() > 0
        best = int(max(range(len(corr)), key=lambda k: corr[k] if valid[k] else -9))
        tpr = (self.tp / self.pos.clamp_min(1)).cpu().numpy()
        fpr = (self.fp / self.neg.clamp_min(1)).cpu().numpy()
        c = max(self.cnt, 1)
        print(f"[predtac][quality] steps={self.steps} stale_ticks={stale_ticks}(x{capture_every} env steps) "
              f"gt_pos={float(self.gt_pos) / c:.4f} pred_pos={float(self.pred_pos) / c:.4f} | "
              f"corr@lag(env steps): " + " ".join(f"{k}:{corr[k]:.3f}" for k in range(len(corr)) if valid[k]) +
              f" | best_lag={best} | k=0 tpr={tpr[0]:.3f} fpr={fpr[0]:.4f} | k=best tpr={tpr[best]:.3f} fpr={fpr[best]:.4f}",
              flush=True)


_DUMP = {"n": 0}


def dump_frames(frames, sides_all_envs, step, extra=None):
    """Diagnostic (env PREDTAC_DUMP_DIR): save a few of the frames the predictor server is actually served, together with the
    ground-truth-pose hand boxes keyed as sent (left/right), so scripts/.../analyze_predtac_boxes.py can compare them with the
    SAM3 boxes the training cache was built from. Every PREDTAC_DUMP_EVERY (40) submitted steps, envs 0..PREDTAC_DUMP_ENVS-1,
    at most PREDTAC_DUMP_MAX (40) dumps."""
    import json
    import cv2
    d = os.environ.get("PREDTAC_DUMP_DIR")
    if not d or _DUMP["n"] >= int(os.environ.get("PREDTAC_DUMP_MAX", "40")):
        return
    if step % int(os.environ.get("PREDTAC_DUMP_EVERY", "40")) != 0:
        return
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    rows = []
    for i in range(int(os.environ.get("PREDTAC_DUMP_ENVS", "3"))):
        name = f"s{step:06d}_e{i}.png"
        cv2.imwrite(os.path.join(d, "frames", name), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        rows.append({"file": name, "step": int(step), "env": i,
                     "boxes": {k: [float(x) for x in v["box"]] for k, v in sides_all_envs[i].items()},
                     "hw": [int(frames[i].shape[0]), int(frames[i].shape[1])]})
    with open(os.path.join(d, "meta.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _DUMP["n"] += 1
