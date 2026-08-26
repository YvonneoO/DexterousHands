#!/usr/bin/env python3
"""One durable DexterousHands training/evaluation queue for a physical GPU."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time


REPO = "/lp-dev/qianqian/DexterousHands"
CODE = os.path.join(REPO, "bidexhands")
PYTHON = "/lp-dev/qianqian/envs/rlgpu/bin/python"
RUNS = os.path.join(REPO, "runs", "task_pipeline")
LD_LIBRARY_PATH = "/lp-dev/qianqian/envs/rlgpu/lib"


def slug(task):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", task).lower()


def task_cfg_args(task):
    """Select repository-owned task/PPO YAMLs without overriding their values."""
    train_config = "cfg/ppo/config.yaml"
    if task == "ShadowHandLiftUnderarm":
        train_config = "cfg/ppo/lift_config.yaml"
    elif task == "ShadowHandBlockStack":
        train_config = "cfg/ppo/stack_block_config.yaml"
    elif task == "ShadowHandReOrientation":
        # retrieve_cfg catches this task in the generic branch, so select the
        # repository's dedicated YAML explicitly.
        train_config = "cfg/ppo/re_orientation_config.yaml"
    return [
        "--cfg_env=cfg/{}.yaml".format(task),
        "--cfg_train={}".format(train_config),
    ]


def process_running(logdir):
    output = subprocess.check_output(["ps", "-eo", "args"], universal_newlines=True)
    needle = "--logdir={}".format(logdir)
    return any("train.py" in line and needle in line for line in output.splitlines())


class Worker:
    def __init__(self, gpu, seed_base):
        self.gpu = gpu
        self.seed_base = seed_base
        self.root = os.path.join(RUNS, "gpu{}".format(gpu))
        os.makedirs(self.root, exist_ok=True)
        self.worker_log = os.path.join(self.root, "worker.log")
        self.state_path = os.path.join(self.root, "state.json")

    def log(self, message):
        line = "{} {}\n".format(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message)
        with open(self.worker_log, "a") as handle:
            handle.write(line)
        print(line, end="", flush=True)

    def state(self, **values):
        payload = {}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as handle:
                    payload.update(json.load(handle))
            except Exception:
                pass
        payload.update({"gpu": self.gpu, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        payload.update(values)
        temporary = self.state_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, self.state_path)

    def environment(self):
        env = os.environ.copy()
        env.update({
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
            "CUDA_VISIBLE_DEVICES": str(self.gpu),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        })
        return env

    def command(self, task, seed, num_envs, logdir, max_iterations=None):
        cmd = [
            PYTHON, "train.py", "--task={}".format(task), "--algo=ppo",
            "--num_envs={}".format(num_envs), "--headless", "--sim_device=cuda:0",
            "--rl_device=cuda:0", "--num_threads=4", "--seed={}".format(seed),
            "--logdir={}".format(logdir),
        ]
        cmd.extend(task_cfg_args(task))
        if max_iterations is not None:
            cmd.append("--max_iterations={}".format(max_iterations))
        return cmd

    def run_logged(self, cmd, log_path, phase, task):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.state(phase=phase, task=task, command=cmd, log=log_path)
        self.log("START {} {} -> {}".format(phase, task, log_path))
        with open(log_path, "a") as handle:
            proc = subprocess.Popen(cmd, cwd=CODE, env=self.environment(), stdout=handle, stderr=subprocess.STDOUT)
            self.state(pid=proc.pid)
            rc = proc.wait()
        self.log("END {} {} rc={}".format(phase, task, rc))
        self.state(pid=None, return_code=rc)
        return rc

    def wait_for_adopted(self, task, logdir, final_checkpoint):
        self.log("ADOPT {} logdir={}".format(task, logdir))
        self.state(phase="training_adopted", task=task, log=logdir + "/train.log")
        while process_running(logdir):
            time.sleep(60)
        if os.path.exists(final_checkpoint):
            self.log("ADOPTED COMPLETE {} checkpoint={}".format(task, final_checkpoint))
            return True
        self.log("ADOPTED FAILED {} missing={}".format(task, final_checkpoint))
        self.state(phase="failed", failure="missing final checkpoint", checkpoint=final_checkpoint)
        return False

    def evaluate(self, task, seed, checkpoint):
        result_root = os.path.join(self.root, slug(task), "evaluation")
        metrics_dir = os.path.join(result_root, "metrics")
        video_dir = os.path.join(result_root, "video")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        base = [
            PYTHON, "rollout_validate.py", "--task={}".format(task), "--algo=ppo",
            "--headless", "--sim_device=cuda:0", "--rl_device=cuda:0", "--num_threads=2",
            "--seed={}".format(seed + 1000), "--model_dir={}".format(checkpoint),
        ] + task_cfg_args(task)

        metrics_env = self.environment()
        metrics_env.update({"BIDEX_EVAL_STEPS": "1200", "BIDEX_RESULT_DIR": metrics_dir})
        metrics_log = os.path.join(metrics_dir, "rollout.log")
        self.state(phase="evaluation", task=task, checkpoint=checkpoint)
        self.log("EVAL {} checkpoint={}".format(task, checkpoint))
        with open(metrics_log, "w") as handle:
            metrics_rc = subprocess.call(base + ["--num_envs=128"], cwd=CODE, env=metrics_env,
                                         stdout=handle, stderr=subprocess.STDOUT)

        video_env = self.environment()
        video_env.update({
            "BIDEX_EVAL_STEPS": "600", "BIDEX_RECORD_DIR": video_dir,
            "BIDEX_FRAME_STRIDE": "2", "BIDEX_VIDEO_WIDTH": "640",
            "BIDEX_VIDEO_HEIGHT": "480", "BIDEX_CAMERA_POS": "0.75,-0.95,0.90",
            "BIDEX_CAMERA_TARGET": "-0.25,-0.25,0.55",
        })
        video_log = os.path.join(video_dir, "rollout.log")
        with open(video_log, "w") as handle:
            video_rc = subprocess.call(base + ["--num_envs=1"], cwd=CODE, env=video_env,
                                       stdout=handle, stderr=subprocess.STDOUT)

        summary_path = os.path.join(metrics_dir, "summary.json")
        summary = None
        if os.path.exists(summary_path):
            with open(summary_path) as handle:
                summary = json.load(handle)
        self.log("EVAL END {} metrics_rc={} video_rc={} success_rate={}".format(
            task, metrics_rc, video_rc, None if summary is None else summary.get("episode_success_rate")))
        self.state(phase="evaluated", metrics_rc=metrics_rc, video_rc=video_rc,
                   metrics_summary=summary_path, video=os.path.join(video_dir, "rollout.mp4"),
                   episode_success_rate=None if summary is None else summary.get("episode_success_rate"))

    def train(self, task, seed):
        task_root = os.path.join(self.root, slug(task))
        burnin_logdir = os.path.join(task_root, "burnin_1024")
        burnin_log = os.path.join(burnin_logdir, "train.log")
        rc = self.run_logged(self.command(task, seed, 1024, burnin_logdir, 20), burnin_log,
                             "burnin_1024", task)
        bad = False
        if os.path.exists(burnin_log):
            with open(burnin_log, errors="replace") as handle:
                text = handle.read()
            bad = bool(re.search(r"Traceback|RuntimeError|CUDA error|illegal memory|Segmentation fault", text, re.I))
        if rc != 0 or bad:
            self.log("BURNIN FAILED {} rc={} error_marker={}".format(task, rc, bad))
            self.state(phase="burnin_failed", task=task)
            return None

        full_logdir = os.path.join(task_root, "full_2048")
        full_log = os.path.join(full_logdir, "train.log")
        rc = self.run_logged(self.command(task, seed, 2048, full_logdir), full_log,
                             "training_2048", task)
        final_iteration = 6501 if task == "ShadowHandLiftUnderarm" else 6500
        checkpoint = "{}_seed{}/model_{}.pt".format(full_logdir, seed, final_iteration)
        if rc != 0 or not os.path.exists(checkpoint):
            self.log("TRAIN FAILED {} rc={} checkpoint_exists={}".format(task, rc, os.path.exists(checkpoint)))
            self.state(phase="training_failed", task=task, checkpoint=checkpoint)
            return None
        self.log("TRAIN COMPLETE {} checkpoint={}".format(task, checkpoint))
        return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--adopt-task", default="")
    parser.add_argument("--adopt-logdir", default="")
    parser.add_argument("--adopt-checkpoint", default="")
    parser.add_argument("--adopt-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    lock_path = os.path.join(RUNS, "gpu{}.lock".format(args.gpu))
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("A worker already owns GPU {}".format(args.gpu), file=sys.stderr)
        return 2

    worker = Worker(args.gpu, args.seed_base)
    worker.log("WORKER START tasks={}".format(args.tasks))
    if args.adopt_task:
        if worker.wait_for_adopted(args.adopt_task, args.adopt_logdir, args.adopt_checkpoint):
            worker.evaluate(args.adopt_task, args.adopt_seed, args.adopt_checkpoint)

    for offset, task in enumerate(args.tasks):
        seed = args.seed_base + offset
        checkpoint = worker.train(task, seed)
        if checkpoint:
            worker.evaluate(task, seed, checkpoint)
    worker.state(phase="complete", task=None, pid=None)
    worker.log("WORKER COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
