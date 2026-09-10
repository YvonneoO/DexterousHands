#!/usr/bin/env python3
"""bidexhands-side client for the online Pred-Tac IPC bridge (predtac_ipc.py,
same directory). Owns the per-env "last known" tactile reading and updates it
opportunistically from whatever the server's newest response is -- see
predtac_ipc.py's docstring for why this is asynchronous/stale-tolerant
rather than a per-tick blocking round trip.

Hand-slot convention throughout: index 0 = left, index 1 = right -- matches
the tactile-prediction model's own input/output convention (src/models/
pose_encoder.py's WiLoRFeatEncoder docstring, src/objectives/flow_matching.py's
_fold_grid, and pressure_grids.npz's left_pressure_grid/right_pressure_grid
naming), NOT sim_cond_cache.py's raw SAM3 extraction order (right, left) --
see Ego2Contact's infer_pred_tactile_offline.py for the same handedness
caveat on the offline side. gt_pose_crop.py's `sides` dict uses string keys
("left"/"right") so this client is the one place that commits to a fixed
slot order for the wire format.
"""
import numpy as np

from tactile_collection import predtac_ipc

NUM_LINKS = 17  # egotouch_taxels._layout() anatomical groups per hand


class PredTacClient:
    def __init__(self, run_id, num_envs, num_links=NUM_LINKS):
        self.run_id = run_id
        self.num_envs = num_envs
        self.num_links = num_links
        self.last_tick_seen = -1
        self.continuous = np.zeros((num_envs, 2, num_links), dtype=np.float32)
        self.binary = np.zeros((num_envs, 2, num_links), dtype=np.float32)
        self._tick = 0

    def submit(self, frames_uint8, sides_all_envs):
        """frames_uint8: (num_envs,H,W,3) uint8, one rendered frame per env
        (multi_env_camera.render_all_and_capture's output). sides_all_envs:
        list of length num_envs of gt_pose_crop "sides" dicts
        ({"left": {"box": [x1,y1,x2,y2]}, "right": {...}}, either key
        possibly absent)."""
        boxes = np.full((self.num_envs, 2, 4), np.nan, dtype=np.float32)
        has_hand = np.zeros((self.num_envs, 2), dtype=bool)
        for i, sides in enumerate(sides_all_envs):
            for h, key in enumerate(("left", "right")):
                info = sides.get(key)
                if info is not None:
                    boxes[i, h] = info["box"]
                    has_hand[i, h] = True
        predtac_ipc.write_request(self.run_id, self._tick, frames_uint8, boxes, has_hand)
        self._tick += 1

    def poll(self):
        """Non-blocking. Updates self.continuous/self.binary in place if a
        newer server response is available; always returns the current
        (possibly stale, possibly still all-zero before the first response)
        (continuous, binary) pair, each (num_envs, 2, num_links)."""
        resp = predtac_ipc.read_response(self.run_id)
        if resp is not None and resp["tick"] > self.last_tick_seen:
            self.last_tick_seen = resp["tick"]
            self.continuous = resp["continuous"]
            self.binary = resp["binary"]
        return self.continuous, self.binary

    def staleness_ticks(self):
        """How many client ticks old the most recent response is, right now
        -- (last submitted tick) - (tick embedded in the freshest response
        ever received). self._tick is the NEXT tick to submit, so the most
        recently submitted one is self._tick - 1. -1 (a submit not yet made)
        or a very large number (no response ever received, still all-zero)
        both mean "no real staleness measurement yet", not "zero lag" --
        callers should treat those as a distinct case, not average them in."""
        if self._tick == 0:
            return -1
        if self.last_tick_seen < 0:
            return None  # no response ever received -- still serving all-zero fallback
        return (self._tick - 1) - self.last_tick_seen
