#!/usr/bin/env python3
"""Start an additional pipeline queue after the current GPU worker releases its lock."""

import argparse
import os
import subprocess
import sys
import time


CODE = "/lp-dev/qianqian/DexterousHands/bidexhands"
PYTHON = "/lp-dev/qianqian/envs/rlgpu/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    command = [
        PYTHON,
        os.path.join(CODE, "pipeline_worker.py"),
        "--gpu", str(args.gpu),
        "--seed-base", str(args.seed_base),
        "--tasks",
    ] + args.tasks
    while True:
        rc = subprocess.call(command, cwd=CODE)
        if rc != 2:
            return rc
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
