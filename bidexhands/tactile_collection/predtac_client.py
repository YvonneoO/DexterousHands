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
import time

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
        # Cached from the last submit() call, so poll_blocking() can
        # re-send the SAME frame under a fresh tick number without the
        # caller re-rendering -- see poll_blocking's docstring for why this
        # is necessary (server-side frame_interval decimation).
        self._last_frames = None
        self._last_boxes = None
        self._last_has_hand = None
        # Clear any request/response left over from a PREVIOUS process that
        # used this exact run_id -- see predtac_ipc.reset_run's docstring.
        # Without this, a stale leftover response can hand a brand-new
        # session a bogus high-water-mark tick and silently freeze it on one
        # stale frame for the rest of the run (found live 2026-09-14).
        predtac_ipc.reset_run(run_id)

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
        self._last_frames = frames_uint8
        self._last_boxes = boxes
        self._last_has_hand = has_hand

    def poll(self):
        """Non-blocking. Updates self.continuous/self.binary in place if a
        newer server response is available; always returns the current
        (possibly stale, possibly still all-zero before the first response)
        (continuous, binary) pair, each (num_envs, 2, num_links)."""
        resp = predtac_ipc.read_response(self.run_id)
        if resp is not None and resp["tick"] > self.last_tick_seen:
            self.last_tick_seen = resp["tick"]
            # Defensive: a degenerate frame (e.g. no hand detected, a
            # transient WiLoR/DINO failure) can in principle leak NaN out of
            # the server's pooling -- found live 2026-09-14 on VTDexManip's
            # Handover (job 539404, iteration 267/4400): a MultivariateNormal
            # ValueError traced back to NaN in the policy's action
            # distribution, consistent with an unsanitized NaN tactile
            # reading silently propagating through the obs buffer into the
            # actor. Zero is the same safe fallback poll() already uses
            # before any response has ever arrived, so replacing NaN with 0
            # here is consistent with that existing convention, not a new
            # semantics.
            self.continuous = np.nan_to_num(resp["continuous"], nan=0.0)
            self.binary = np.nan_to_num(resp["binary"], nan=0.0)
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

    def poll_blocking(self, timeout_s=15.0, poll_interval_s=0.02, resubmit_interval_s=1.0):
        """Blocking variant of poll(): spins until a genuinely NEW response
        (any tick newer than whatever was known when this was called) has
        arrived, instead of taking whatever's freshest so far like poll()
        does. This trades wall-clock time for near-zero staleness, so it's
        only appropriate where the caller can afford to stall the sim loop:
        eval (real-time throughput doesn't matter there), or a deliberate
        training-time throttle experiment (see PREDTAC_BLOCKING in the task
        classes) -- never the default high-throughput training path, which
        stays async via poll().

        Deliberately waits for "any newer response", NOT "a response tagged
        with exactly self._tick - 1": the server decimates to match training's
        own frame_interval (only every 2nd raw client tick actually advances
        its window and gets a fresh response -- see predtac_server.py), so a
        response for the EXACT most-recently-submitted tick may never arrive
        at all. Waiting for an exact match instead of "any newer" was tried
        first and burned the full timeout on every single call (see commit
        history) -- always falling back one tick short of the target instead
        of ever really blocking.

        ⚠️ RESUBMISSION (found live 2026-09-14): frame_interval decimation
        means a SINGLE submitted tick has only a 1-in-frame_interval chance
        of ever getting a response at all -- the server's main loop marks a
        skipped tick "seen" and moves on WITHOUT ever writing a response for
        it (see predtac_server.py's `continue` under the decimation check),
        so passively waiting on one unlucky submission can deterministically
        burn the full timeout_s -- confirmed live: at frame_interval=2,
        every odd-numbered submitted tick got zero response, exactly
        matching a 63s/iteration blocking-training slowdown that PERSISTED
        even after forcing the server and training processes onto the same
        physical node (ruling out any filesystem/network cause -- see
        vision_ppo_train_predtac_colocated.sbatch). Fix: if no newer
        response has appeared after resubmit_interval_s, re-send the SAME
        frame (cached by submit()) under a fresh tick number -- physics
        hasn't advanced while we've been waiting, so the frame content is
        still correct, and this gives the server's decimation counter
        another, differently-pared chance to keep it. Never disable this by
        raising resubmit_interval_s above timeout_s "to save requests" --
        frame_interval is a fixed model-training convention (NOT a
        throughput knob, see predtac_server.py's --frame_interval help), so
        without resubmission roughly 1-in-frame_interval calls are
        guaranteed dead on arrival.

        Falls back to whatever's freshest and prints a warning if timeout_s
        elapses first, rather than hanging forever on a dead or
        permanently-behind server."""
        if self._tick == 0:
            return self.continuous, self.binary
        seen_before = self.last_tick_seen
        deadline = time.time() + timeout_s
        next_resubmit = time.time() + resubmit_interval_s
        while self.last_tick_seen <= seen_before:
            now = time.time()
            if now > deadline:
                print(f"[predtac][blocking] timeout after {timeout_s}s waiting for a response "
                      f"newer than tick {seen_before} (submitted tick={self._tick - 1}) -- "
                      f"falling back to stale value", flush=True)
                break
            if now > next_resubmit and self._last_frames is not None:
                predtac_ipc.write_request(self.run_id, self._tick, self._last_frames,
                                           self._last_boxes, self._last_has_hand)
                self._tick += 1
                next_resubmit = now + resubmit_interval_s
            time.sleep(poll_interval_s)
            self.poll()
        return self.continuous, self.binary
