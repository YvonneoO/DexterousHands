#!/usr/bin/env python3
"""Run one BottleCap-v2 smoke test and promote it to its configured full run."""

import argparse
import json
import os
import re
import subprocess
import time


REPO = "/lp-dev/qianqian/DexterousHands"
CODE = os.path.join(REPO, "bidexhands")
PYTHON = "/lp-dev/qianqian/envs/rlgpu/bin/python"
ROOT = os.path.join(REPO, "runs", "bottlecap_v2")


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_status(path, **values):
    payload = {}
    if os.path.exists(path):
        try:
            with open(path) as handle:
                payload.update(json.load(handle))
        except Exception:
            pass
    payload.update(values)
    payload["updated_at"] = timestamp()
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def run_logged(command, log_path, environment, status_path, phase):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_status(status_path, phase=phase, command=command, log=log_path)
    with open(log_path, "a") as handle:
        handle.write("{} START {}\n".format(timestamp(), phase))
        handle.flush()
        process = subprocess.Popen(command, cwd=CODE, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        write_status(status_path, pid=process.pid)
        return_code = process.wait()
        handle.write("{} END {} rc={}\n".format(timestamp(), phase, return_code))
    write_status(status_path, pid=None, return_code=return_code)
    return return_code


def log_is_healthy(path):
    with open(path, errors="replace") as handle:
        text = handle.read()
    fatal = re.compile(r"Traceback|CUDA error|out of memory|nan(?:\s|$)|segmentation fault", re.IGNORECASE)
    has_iteration = "Learning iteration" in text
    return has_iteration and fatal.search(text) is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--graphics-gpu", type=int, default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cfg-env", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()

    run_root = os.path.join(ROOT, args.task)
    os.makedirs(run_root, exist_ok=True)
    status_path = os.path.join(run_root, "status.json")
    environment = os.environ.copy()
    environment.update({
        "LD_LIBRARY_PATH": "/lp-dev/qianqian/envs/rlgpu/lib",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "ISAAC_GRAPHICS_DEVICE_ID": str(
            args.gpu if args.graphics_gpu is None else args.graphics_gpu),
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "PYTHONUNBUFFERED": "1",
    })

    common = [
        PYTHON,
        "train.py",
        "--task={}".format(args.task),
        "--algo=ppo",
        "--headless",
        "--sim_device=cuda:0",
        "--rl_device=cuda:0",
        "--num_threads=4",
        "--seed={}".format(args.seed),
        "--cfg_env={}".format(args.cfg_env),
        "--cfg_train=cfg/ppo/bottlecap_v2.yaml",
    ]
    if args.resume > 0:
        common.append("--resume={}".format(args.resume))

    if not args.full_only:
        smoke_root = os.path.join(run_root, "smoke_256")
        smoke_command = common + [
            "--num_envs=256",
            "--max_iterations=40",
            "--logdir={}".format(smoke_root),
        ]
        smoke_log = os.path.join(smoke_root, "train.log")
        smoke_rc = run_logged(smoke_command, smoke_log, environment, status_path, "smoke_256")
        if smoke_rc != 0 or not log_is_healthy(smoke_log):
            write_status(status_path, phase="smoke_failed", smoke_rc=smoke_rc)
            return 1

    full_root = os.path.join(run_root, "full_2048")
    full_command = common + [
        "--num_envs=2048",
        "--max_iterations=3500",
        "--logdir={}".format(full_root),
    ]
    full_log = os.path.join(full_root, "train.log")
    write_status(status_path, smoke_rc=0, smoke_validated=True,
                 resumed_from=args.resume if args.resume > 0 else None,
                 graphics_device=(args.gpu if args.graphics_gpu is None
                                  else args.graphics_gpu))
    full_rc = run_logged(full_command, full_log, environment, status_path, "full_2048")
    checkpoint = os.path.join(full_root + "_seed{}".format(args.seed), "model_3500.pt")
    write_status(
        status_path,
        phase="complete" if full_rc == 0 else "full_failed",
        full_rc=full_rc,
        expected_checkpoint=checkpoint,
        checkpoint_exists=os.path.exists(checkpoint),
    )
    return full_rc


if __name__ == "__main__":
    raise SystemExit(main())
