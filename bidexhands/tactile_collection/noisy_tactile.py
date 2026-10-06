"""Noisy-GT tactile: simulate the online tactile PREDICTOR's output from ground-truth link forces,
so PPO can train at thousands of envs without a camera/predictor in the loop.

The real predictor (predtac_server.py) returns, per env/hand, 17 anatomical-link values:
  continuous[l] = max over link l's taxels of the predicted pressure (0..1, predictor units)
  binary[l]     = OR over link l's taxels of (taxel > pressure_threshold)  ==  (continuous[l] > threshold)
so ONE noisy continuous value per link determines both channels (consistent by construction here too).

Model, per link l (all (N, 2, 17) tensors, slot 0 = left, 1 = right, links in the server's
sorted-name order):
  g      = clamp(force_l * scale_l, 0, 1)            ground-truth pooled pressure in predictor units
  detect = persistent Bernoulli: P(detect | g > thr) = tpr_l, P(detect | g <= thr) = fpr_l
  mag    = g * exp(sigma_l * z - sigma_l^2 / 2)      multiplicative lognormal distortion
  cont   = detect ? max(mag, thr*(1.05+0.5u)) : min(mag, 0.95*thr)
Errors persist across consecutive steps via an AR(1) latent (rho), like a visual predictor's errors.

!! The default parameters are PLACEHOLDERS (smoke-test values), not measured. Calibrate scale / tpr /
!! fpr / sigma / rho from predictor-vs-GT statistics on held-out sim episodes and pass them via a JSON
!! file (cfg env.noisyTactileParams): {"threshold":.., "scale":[17], "tpr":[17], "fpr":[17], "sigma":[17], "rho":..}.
"""
import json
import math

import torch

NUM_LINKS = 17

DEFAULT_PARAMS = {
    "threshold": 0.1,   # the server's pressure_threshold
    "scale": 0.02,      # PLACEHOLDER: raw link force (N) -> pooled predictor-unit pressure
    "tpr": 0.8,         # PLACEHOLDER
    "fpr": 0.01,        # PLACEHOLDER
    "sigma": 0.5,       # PLACEHOLDER
    "rho": 0.8,         # PLACEHOLDER temporal persistence of errors
    "calibrated": False,
}


def _per_link(v, device):
    t = torch.as_tensor(v, dtype=torch.float32, device=device)
    return t.expand(NUM_LINKS).clone() if t.ndim == 0 else t.reshape(NUM_LINKS)


class NoisyTactileModel:
    def __init__(self, num_envs, device, params_path=None):
        p = dict(DEFAULT_PARAMS)
        if params_path:
            with open(params_path) as f:
                p.update(json.load(f))
            p["calibrated"] = True
        self.device = device
        self.thr = float(p["threshold"])
        self.rho = float(p["rho"])
        self.scale = _per_link(p["scale"], device)
        self.tpr = _per_link(p["tpr"], device)
        self.fpr = _per_link(p["fpr"], device)
        self.sigma = _per_link(p["sigma"], device)
        self.z_ev = torch.randn(num_envs, 2, NUM_LINKS, device=device)
        self.z_mag = torch.randn(num_envs, 2, NUM_LINKS, device=device)
        tag = "calibrated params from " + str(params_path) if p["calibrated"] else "PLACEHOLDER params (NOT measured)"
        print(f"[noisy_tactile] {tag}: thr={self.thr} rho={self.rho} "
              f"scale[0]={self.scale[0].item():.4f} tpr[0]={self.tpr[0].item():.2f} "
              f"fpr[0]={self.fpr[0].item():.3f} sigma[0]={self.sigma[0].item():.2f}", flush=True)

    def _ar1(self, z):
        return self.rho * z + math.sqrt(max(0.0, 1.0 - self.rho ** 2)) * torch.randn_like(z)

    def __call__(self, force):
        """force: (N, 2, 17) raw link force magnitudes (sim units, N). Returns (continuous, binary), both (N, 2, 17)."""
        g = (force * self.scale).clamp(0.0, 1.0)
        self.z_ev = self._ar1(self.z_ev)
        self.z_mag = self._ar1(self.z_mag)
        u = 0.5 * (1.0 + torch.erf(self.z_ev / math.sqrt(2.0)))  # uniform(0,1), AR-correlated
        true_contact = g > self.thr
        detect = torch.where(true_contact, u < self.tpr, u < self.fpr)
        mag = g * torch.exp(self.sigma * self.z_mag - 0.5 * self.sigma ** 2)
        hit = torch.maximum(mag, self.thr * (1.05 + 0.5 * u))
        miss = torch.minimum(mag, torch.full_like(mag, 0.95 * self.thr))
        cont = torch.where(detect, hit, miss).clamp(0.0, 1.0)
        binary = (cont > self.thr).float()
        return cont, binary
