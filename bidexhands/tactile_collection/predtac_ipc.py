#!/usr/bin/env python3
"""File-based IPC between the online-PPO process (bidexhands_isaacgym_py38,
pinned old torch) and the tactile-prediction server process (touchanything
env, modern torch/transformers) -- these two stacks cannot coexist in one
process (SAM3/WiLoR need modern torch; Isaac Gym's bindings here need the
pinned old one), so P+Pred-Tac training needs two processes talking over
something version-agnostic. Plain numpy + os is importable unmodified from
either environment, so this module is the ONLY thing shared by both sides
(predtac_client.py on the bidexhands side, predtac_server.py in Ego2Contact
on the touchanything side) -- no torch import here at all.

Design is deliberately ASYNCHRONOUS / stale-tolerant, not a lockstep
request/response per tick: the client overwrites one request file every PPO
tick (fire-and-forget, never blocks on the server), and separately reads
whatever the newest response file says, carrying the last-known tactile
reading forward across ticks until a newer response appears (starting from
all-zeros before the server's first response). This decouples PPO's physics
tick rate from the server's WiLoR+DiT compute rate entirely -- exactly the
tradeoff already agreed for this ablation (slower/lagged tactile signal is
fine; a synchronous per-tick stall is not, and was never benchmarked as fast
enough for this to begin with -- see benchmark_online_tactile_pred.py).

Both request and response are written atomically (write to a temp path in
the SAME directory, then os.replace) so the reader never observes a
partially-written file -- os.replace is atomic on a single POSIX filesystem,
which scratch is.
"""
import os

import numpy as np

YQQ = "/scratch/project/prj-02-phai-lab/yqq"
IPC_ROOT = os.environ.get("PREDTAC_IPC_ROOT", os.path.join(YQQ, "ipc", "predtac"))


def run_dir(run_id):
    d = os.path.join(IPC_ROOT, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_write_npz(path, **arrays):
    tmp_path = path + ".tmp"
    np.savez(tmp_path, **arrays)
    # np.savez appends .npz if the path doesn't already end with it.
    if not tmp_path.endswith(".npz"):
        tmp_path += ".npz"
    os.replace(tmp_path, path)


def request_path(run_id):
    return os.path.join(run_dir(run_id), "request.npz")


def response_path(run_id):
    return os.path.join(run_dir(run_id), "response.npz")


def write_request(run_id, tick, frames_uint8, boxes, has_hand):
    """frames_uint8: (N,H,W,3) uint8. boxes: (N,2,4) float32, hand order
    [left, right], NaN where has_hand is False. has_hand: (N,2) bool."""
    _atomic_write_npz(
        request_path(run_id),
        tick=np.asarray(tick, dtype=np.int64),
        frames=np.asarray(frames_uint8, dtype=np.uint8),
        boxes=np.asarray(boxes, dtype=np.float32),
        has_hand=np.asarray(has_hand, dtype=bool),
    )


def read_request(run_id):
    """Returns None if no request has been written yet, or if it's mid-write
    on this exact call (rare race, self-heals next poll -- os.replace makes
    a torn READ impossible, but the file can simply not exist yet)."""
    path = request_path(run_id)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            return {
                "tick": int(data["tick"]),
                "frames": data["frames"],
                "boxes": data["boxes"],
                "has_hand": data["has_hand"],
            }
    except (OSError, ValueError, EOFError):
        return None


def write_response(run_id, tick, continuous, binary):
    """continuous, binary: (N,2,17) float32 -- per env, per hand ([left,
    right]), per anatomical link group (egotouch_taxels._layout order)."""
    _atomic_write_npz(
        response_path(run_id),
        tick=np.asarray(tick, dtype=np.int64),
        continuous=np.asarray(continuous, dtype=np.float32),
        binary=np.asarray(binary, dtype=np.float32),
    )


def read_response(run_id):
    path = response_path(run_id)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            return {
                "tick": int(data["tick"]),
                "continuous": data["continuous"],
                "binary": data["binary"],
            }
    except (OSError, ValueError, EOFError):
        return None
