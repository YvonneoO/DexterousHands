#!/usr/bin/env python3
"""Generate a self-contained HTML status dashboard for the Bi-DexHands pipeline."""

import argparse
import datetime as dt
import glob
import html
import json
import os
import re
import subprocess
import time


ROOT = "/lp-dev/qianqian/DexterousHands"
PIPELINE = os.path.join(ROOT, "runs", "task_pipeline")
HORA_ROOT = "/lp-dev/qianqian/hora"
CUBE_ROOT = os.path.join(HORA_ROOT, "outputs", "ShadowHandCubeRotation")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def pipeline_spec(task, gpu, seed, family):
    logdir = os.path.join(PIPELINE, "gpu{}".format(gpu), snake(task), "full_2048")
    return {
        "task": task, "gpu": gpu, "seed": seed, "family": family,
        "log": os.path.join(logdir, "train.log"),
        "checkpoint": "{}_seed{}/model_6500.pt".format(logdir, seed),
        "eval_root": os.path.join(PIPELINE, "gpu{}".format(gpu), snake(task), "evaluation"),
    }


def repo_configs(task):
    train = "cfg/ppo/config.yaml"
    if task == "ShadowHandLiftUnderarm":
        train = "cfg/ppo/lift_config.yaml"
    elif task == "ShadowHandBlockStack":
        train = "cfg/ppo/stack_block_config.yaml"
    elif task == "ShadowHandReOrientation":
        train = "cfg/ppo/re_orientation_config.yaml"
    return "cfg/{}.yaml".format(task), train


TASKS = [
    {"task": "ShadowHandOver", "gpu": 0, "seed": 0, "family": "handover",
     "log": os.path.join(ROOT, "runs/formal_2048/over/train.log"),
     "checkpoint": os.path.join(ROOT, "runs/formal_2048/over_seed0/model_6500.pt"),
     "eval_root": os.path.join(PIPELINE, "gpu0/shadow_hand_over/evaluation")},
    {"task": "ShadowHandLiftUnderarm", "gpu": 1, "seed": 1, "family": "in-hand / lift",
     "log": os.path.join(ROOT, "runs/formal_2048/lift/train.log"),
     "checkpoint": os.path.join(ROOT, "runs/formal_2048/lift_seed1/model_6501.pt"),
     "eval_root": os.path.join(PIPELINE, "gpu1/shadow_hand_lift_underarm/evaluation")},
    {"task": "ShadowHandReOrientation", "gpu": 2, "seed": 2, "family": "in-hand / pose",
     "log": os.path.join(ROOT, "runs/formal_2048/reorient/train.log"),
     "checkpoint": os.path.join(ROOT, "runs/formal_2048/reorient_seed2/model_6500.pt"),
     "eval_root": os.path.join(PIPELINE, "gpu2/shadow_hand_re_orientation/evaluation")},
    {"task": "ShadowHandBottleCap", "gpu": 3, "seed": 3, "family": "articulated",
     "log": os.path.join(ROOT, "runs/formal_2048/bottlecap/train.log"),
     "checkpoint": os.path.join(ROOT, "runs/formal_2048/bottlecap_seed3/model_6500.pt"),
     "eval_root": os.path.join(PIPELINE, "gpu3/shadow_hand_bottle_cap/evaluation")},
    pipeline_spec("ShadowHandDoorOpenInward", 5, 50, "articulated"),
    pipeline_spec("ShadowHandScissors", 6, 60, "articulated"),
    pipeline_spec("ShadowHandCatchUnderarm", 0, 10, "catch"),
    pipeline_spec("ShadowHandCatchOver2Underarm", 0, 11, "catch"),
    pipeline_spec("ShadowHandDoorCloseInward", 0, 70, "articulated"),
    pipeline_spec("ShadowHandGraspAndPlace", 1, 20, "grasp / place"),
    pipeline_spec("ShadowHandBlockStack", 1, 21, "stacking"),
    pipeline_spec("ShadowHandSwingCup", 1, 71, "articulated"),
    pipeline_spec("ShadowHandDoorCloseOutward", 2, 30, "articulated"),
    pipeline_spec("ShadowHandPushBlock", 2, 31, "push"),
    pipeline_spec("ShadowHandPen", 2, 32, "in-hand / articulated"),
    pipeline_spec("ShadowHandTwoCatchUnderarm", 2, 72, "catch"),
    pipeline_spec("ShadowHandSwitch", 3, 40, "articulated"),
    pipeline_spec("ShadowHandDoorOpenOutward", 3, 41, "articulated"),
    pipeline_spec("ShadowHandKettle", 3, 42, "articulated"),
    pipeline_spec("ShadowHandCatchAbreast", 2, 73, "catch"),
]


def last_float(text, label):
    matches = re.findall(re.escape(label) + r"\s*([-+0-9.eE]+)", text)
    return float(matches[-1]) if matches else None


def parse_log(path):
    result = {"iteration": None, "max_iteration": None, "reward": None,
              "reward_step": None, "train_success": None, "consecutive": None,
              "eta_seconds": None, "error": None}
    if not os.path.exists(path):
        return result
    with open(path, "r", errors="replace") as handle:
        text = ANSI.sub("", handle.read())
    iterations = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
    if iterations:
        result["iteration"], result["max_iteration"] = [int(x) for x in iterations[-1]]
    result["reward"] = last_float(text, "Mean reward:")
    result["reward_step"] = last_float(text, "Mean reward/step:")
    result["train_success"] = last_float(text, "Mean episode successes:")
    result["consecutive"] = last_float(text, "Mean episode consecutive_successes:")
    result["eta_seconds"] = last_float(text, "ETA:")
    error = re.search(r"Traceback|RuntimeError|CUDA error|illegal memory|Segmentation fault", text, re.I)
    result["error"] = error.group(0) if error else None
    return result


def read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return None


def format_number(value, digits=3):
    return "—" if value is None else ("{:.%df}" % digits).format(value)


def format_percent(value):
    return "—" if value is None else "{:.1f}%".format(100.0 * value)


def format_eta(seconds):
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return "{}h {:02d}m".format(hours, minutes) if hours else "{}m".format(minutes)


def relative_href(path):
    return os.path.relpath(path, PIPELINE).replace(os.sep, "/")


def collect():
    try:
        processes = subprocess.check_output(["ps", "-eo", "args"], universal_newlines=True)
    except Exception:
        processes = ""
    rows = []
    for spec in TASKS:
        row = dict(spec)
        row["cfg_env"], row["cfg_train"] = repo_configs(spec["task"])
        row.update(parse_log(spec["log"]))
        final_metrics_path = os.path.join(spec["eval_root"], "metrics", "summary.json")
        final_video_path = os.path.join(spec["eval_root"], "video", "rollout.mp4")
        final_trajectory_path = os.path.join(spec["eval_root"], "metrics", "trajectory_env0.npz")
        evaluation = read_json(final_metrics_path)
        evaluation_kind = "final" if evaluation else None
        metrics_path = final_metrics_path
        video_path = final_video_path
        trajectory_path = final_trajectory_path
        # Keep the existing Over model_1000 smoke visible until final evaluation
        # replaces it. It is labeled interim and never counts as task completion.
        if not evaluation and spec["task"] == "ShadowHandOver":
            smoke_root = os.path.join(ROOT, "runs", "rollout_smoke", "over_1000")
            smoke_metrics = os.path.join(smoke_root, "summary.json")
            smoke_evaluation = read_json(smoke_metrics)
            if smoke_evaluation:
                evaluation = smoke_evaluation
                evaluation_kind = "interim model_1000"
                metrics_path = smoke_metrics
                video_path = os.path.join(smoke_root, "rollout.mp4")
                trajectory_path = os.path.join(smoke_root, "trajectory_env0.npz")
        running = "train.py" in processes and "--task={}".format(spec["task"]) in processes
        if running:
            status = "training"
        elif evaluation_kind == "final":
            status = "evaluated"
        elif os.path.exists(spec["checkpoint"]):
            status = "trained"
        elif row["error"]:
            status = "failed"
        elif row["iteration"] is not None:
            status = "paused"
        else:
            status = "queued"
        row.update({
            "status": status,
            "metrics": metrics_path,
            "video": video_path,
            "trajectory": trajectory_path,
            "evaluation_kind": evaluation_kind,
            "metrics_exists": os.path.exists(metrics_path),
            "video_exists": os.path.exists(video_path),
            "trajectory_exists": os.path.exists(trajectory_path),
            "eval_success": None if not evaluation else evaluation.get("episode_success_rate"),
            "eval_reward_step": None if not evaluation else evaluation.get("mean_reward_per_step"),
            "position_error": None if not evaluation else evaluation.get("position_error_mean_m"),
            "rotation_error": None if not evaluation else evaluation.get("rotation_error_mean_rad"),
        })
        if spec["task"] == "ShadowHandOver":
            tactile_root = os.path.join(
                spec["eval_root"], "tactile_pa", "collection_1200_seed1000")
            tactile_summary_path = os.path.join(tactile_root, "summary.json")
            tactile_summary = read_json(tactile_summary_path)
            row["tactile"] = {
                "root": tactile_root,
                "summary": tactile_summary_path,
                "pressure_grids": os.path.join(tactile_root, "pressure_grids.npz"),
                "trajectory": os.path.join(tactile_root, "trajectory_env0.npz"),
                "video": os.path.join(tactile_root, "tactile.mp4"),
                "summary_exists": os.path.exists(tactile_summary_path),
                "pressure_grids_exists": os.path.exists(os.path.join(tactile_root, "pressure_grids.npz")),
                "trajectory_exists": os.path.exists(os.path.join(tactile_root, "trajectory_env0.npz")),
                "video_exists": os.path.exists(os.path.join(tactile_root, "tactile.mp4")),
                "metrics": tactile_summary or {},
            }
            paired_root = os.path.join(
                spec["eval_root"], "tactile_pa", "paired_rgb_tactile_1200_seed1000_v3")
            paired_summary_path = os.path.join(paired_root, "summary.json")
            paired_summary = read_json(paired_summary_path)
            row["paired_tactile"] = {
                "root": paired_root,
                "summary": paired_summary_path,
                "pressure_grids": os.path.join(paired_root, "pressure_grids.npz"),
                "trajectory": os.path.join(paired_root, "trajectory_env0.npz"),
                "rgb_video": os.path.join(paired_root, "rgb.mp4"),
                "tactile_video": os.path.join(paired_root, "tactile.mp4"),
                "side_by_side_video": os.path.join(paired_root, "rgb_tactile_side_by_side.mp4"),
                "summary_exists": os.path.exists(paired_summary_path),
                "pressure_grids_exists": os.path.exists(os.path.join(paired_root, "pressure_grids.npz")),
                "trajectory_exists": os.path.exists(os.path.join(paired_root, "trajectory_env0.npz")),
                "rgb_video_exists": os.path.exists(os.path.join(paired_root, "rgb.mp4")),
                "tactile_video_exists": os.path.exists(os.path.join(paired_root, "tactile.mp4")),
                "side_by_side_video_exists": os.path.exists(os.path.join(paired_root, "rgb_tactile_side_by_side.mp4")),
                "metrics": paired_summary or {},
            }
        # Prefer the verified native-success-only paired collection whenever it
        # exists.  Unlike the original Over-only demo, this layout is shared by
        # every task and stores media/trajectory files inside successful_episode.
        success_root = os.path.join(
            spec["eval_root"], "tactile_pa", "paired_ego_success_v1")
        success_artifact_root = os.path.join(success_root, "successful_episode")
        success_summary_path = os.path.join(success_root, "summary.json")
        success_summary = read_json(success_summary_path)
        if success_summary and os.path.exists(
                os.path.join(success_artifact_root, "trajectory_env0.npz")):
            row["paired_tactile"] = {
                "root": success_root,
                "summary": success_summary_path,
                "pressure_grids": os.path.join(success_artifact_root, "pressure_grids.npz"),
                "trajectory": os.path.join(success_artifact_root, "trajectory_env0.npz"),
                "rgb_video": os.path.join(success_artifact_root, "rgb.mp4"),
                "tactile_video": os.path.join(success_artifact_root, "tactile.mp4"),
                "side_by_side_video": os.path.join(
                    success_artifact_root, "rgb_tactile_side_by_side.mp4"),
                "summary_exists": True,
                "pressure_grids_exists": os.path.exists(
                    os.path.join(success_artifact_root, "pressure_grids.npz")),
                "trajectory_exists": True,
                "rgb_video_exists": os.path.exists(
                    os.path.join(success_artifact_root, "rgb.mp4")),
                "tactile_video_exists": os.path.exists(
                    os.path.join(success_artifact_root, "tactile.mp4")),
                "side_by_side_video_exists": os.path.exists(
                    os.path.join(success_artifact_root, "rgb_tactile_side_by_side.mp4")),
                "metrics": success_summary,
                "success_only": True,
            }
        if spec["task"] == "ShadowHandReOrientation":
            hora_root = os.path.join(spec["eval_root"], "hora_v4_demo")
            textured_summary_path = os.path.join(hora_root, "textured_full_summary.json")
            physical_summary_path = os.path.join(hora_root, "physical_camera_v2_summary.json")
            validation_path = os.path.join(hora_root, "physical_validation.json")
            config_path = os.path.join(hora_root, "hora_cube_stage1_config.yaml")
            row["hora_demo"] = {
                "root": hora_root,
                "primary_video": os.path.join(hora_root, "hora_v4_textured_full.mp4"),
                "alternate_video": os.path.join(hora_root, "hora_v4_physical_camera_v2.mp4"),
                "textured_summary": textured_summary_path,
                "physical_summary": physical_summary_path,
                "validation": validation_path,
                "config": config_path,
                "primary_video_exists": os.path.exists(os.path.join(hora_root, "hora_v4_textured_full.mp4")),
                "alternate_video_exists": os.path.exists(os.path.join(hora_root, "hora_v4_physical_camera_v2.mp4")),
                "textured_summary_exists": os.path.exists(textured_summary_path),
                "physical_summary_exists": os.path.exists(physical_summary_path),
                "validation_exists": os.path.exists(validation_path),
                "config_exists": os.path.exists(config_path),
                "metrics": read_json(textured_summary_path) or {},
                "validation_metrics": read_json(validation_path) or {},
            }
        rows.append(row)
    return rows


def collect_cube():
    status_path = os.path.join(CUBE_ROOT, "pipeline", "status.txt")
    log_path = os.path.join(CUBE_ROOT, "pipeline", "pipeline.log")
    try:
        with open(status_path, "r", errors="replace") as handle:
            status_text = handle.read().strip()
    except OSError:
        status_text = "not installed"
    stage = status_text.split()[0] if status_text else "unknown"
    error = None
    if os.path.exists(log_path):
        with open(log_path, "r", errors="replace") as handle:
            log_text = ANSI.sub("", handle.read())
        match = re.search(
            r"Traceback|RuntimeError|CUDA error|illegal memory|Segmentation fault|AssertionError",
            log_text, re.I)
        error = match.group(0) if match else None
    if error:
        stage = "failed"

    metrics = {}
    event_files = sorted(
        glob.glob(os.path.join(CUBE_ROOT, "cube_z_stage1", "stage1_tb", "events*")),
        key=os.path.getmtime,
    )
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = EventAccumulator(event_files[-1], size_guidance={"scalars": 0})
            events.Reload()
            tags = set(events.Tags().get("scalars", []))
            for tag in ["rotation_reward", "full_rotation_success", "drop_fraction",
                        "episode_rewards/step", "episode_lengths/step"]:
                if tag in tags and events.Scalars(tag):
                    item = events.Scalars(tag)[-1]
                    metrics[tag] = item.value
                    metrics["agent_steps"] = item.step
        except Exception as exc:
            metrics["tensorboard_error"] = str(exc)

    cache = os.path.join(HORA_ROOT, "cache", "shadow_cube_grasp_50k_s07.npy")
    checkpoint = os.path.join(CUBE_ROOT, "cube_z_stage1", "stage1_nn", "best.pth")
    return {
        "task": "ShadowHandCubeRotation",
        "family": "single-hand in-hand rotation",
        "gpu": 0,
        "stage": stage,
        "status_text": status_text,
        "error": error,
        "metrics": metrics,
        "grasp_cache": cache,
        "warm_checkpoint": os.path.join(HORA_ROOT, "outputs", "ShadowHandRWS", "rws_v1", "stage1_nn", "best.pth"),
        "checkpoint": checkpoint,
        "log": log_path,
        "grasp_cache_exists": os.path.exists(cache),
        "checkpoint_exists": os.path.exists(checkpoint),
    }


def render(rows, cube):
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    counts = {name: sum(row["status"] == name for row in rows)
              for name in ["training", "evaluated", "trained", "queued", "paused", "failed"]}
    complete = counts["evaluated"] + counts["trained"]
    leftovers = len(rows) - complete
    cards = [
        ("20", "total tasks"),
        (str(counts["training"]), "training now"),
        (str(complete), "trained / evaluated"),
        (str(leftovers), "leftovers"),
    ]
    card_html = "".join('<div class="metric"><strong>{}</strong><span>{}</span></div>'.format(v, l) for v, l in cards)
    cube_metrics = cube.get("metrics", {})
    cube_panel = """
    <section class="cube-panel"><div><div class="eyebrow">Priority experiment · HORA/RWS</div>
      <h2>ShadowHand cube rotation</h2><p>{status}</p></div>
      <div class="cube-grid"><div><span>GPU / stage</span><strong>GPU {gpu} · {stage}</strong></div>
      <div><span>Agent steps</span><strong>{steps}</strong></div>
      <div><span>Rotation signal</span><strong>{rotation}</strong></div>
      <div><span>Full 2π success</span><strong>{success}</strong></div>
      <div><span>Episode length</span><strong>{length}</strong></div>
      <div><span>Drop fraction</span><strong>{drop}</strong></div></div>
      <div class="cube-paths"><code>{cache}</code><code>{checkpoint}</code><code>{log}</code></div>
    </section>""".format(
        status=html.escape(cube.get("status_text", "unknown")), gpu=cube.get("gpu", 0),
        stage=html.escape(cube.get("stage", "unknown")),
        steps="—" if cube_metrics.get("agent_steps") is None else "{:,}".format(cube_metrics["agent_steps"]),
        rotation=format_number(cube_metrics.get("rotation_reward"), 3),
        success=format_percent(cube_metrics.get("full_rotation_success")),
        length=format_number(cube_metrics.get("episode_lengths/step"), 1),
        drop=format_percent(cube_metrics.get("drop_fraction")),
        cache=html.escape(cube["grasp_cache"]), checkpoint=html.escape(cube["checkpoint"]),
        log=html.escape(cube["log"]),
    )
    body_rows = []
    for row in rows:
        iteration = "—" if row["iteration"] is None else "{:,} / {:,}".format(row["iteration"], row["max_iteration"])
        eval_success = format_percent(row["eval_success"])
        train_success = format_percent(row["train_success"])
        reward = format_number(row["reward"], 2)
        artifact_lines = [
            '<div><span>env</span><code>{}</code></div>'.format(html.escape(row["cfg_env"])),
            '<div><span>ppo</span><code>{}</code></div>'.format(html.escape(row["cfg_train"])),
            '<div><span>ckpt</span><code>{}</code></div>'.format(html.escape(row["checkpoint"])),
            '<div><span>log</span><code>{}</code></div>'.format(html.escape(row["log"])),
        ]
        evaluation_panel = ""
        if row["evaluation_kind"]:
            artifact_lines.extend([
                '<div><span>traj</span><code>{}</code></div>'.format(html.escape(row["trajectory"])),
                '<div><span>video</span><code>{}</code></div>'.format(html.escape(row["video"])),
            ])
            links = []
            if row["metrics_exists"]:
                links.append('<a href="{}" target="_blank">summary.json</a>'.format(html.escape(relative_href(row["metrics"]))))
            if row["trajectory_exists"]:
                links.append('<a href="{}" download>trajectory.npz</a>'.format(html.escape(relative_href(row["trajectory"]))))
            if row["video_exists"]:
                links.append('<a href="{}" target="_blank">open video</a>'.format(html.escape(relative_href(row["video"]))))
            video = ""
            if row["video_exists"]:
                video = '<video controls playsinline preload="metadata" src="{}">Your browser cannot play this MP4.</video>'.format(
                    html.escape(relative_href(row["video"])))
            evaluation_panel = """
            <details class="evaluation"><summary><span class="eval-kind">{kind}</span> View evaluation &amp; video</summary>
              <div class="eval-grid"><div><span>Success</span><strong>{success}</strong></div><div><span>Reward / step</span><strong>{reward}</strong></div><div><span>Position error</span><strong>{pos}</strong></div><div><span>Rotation error</span><strong>{rot}</strong></div></div>
              <div class="artifact-links">{links}</div>{video}
            </details>""".format(
                kind=html.escape(row["evaluation_kind"]), success=format_percent(row["eval_success"]),
                reward=format_number(row["eval_reward_step"], 3),
                pos="—" if row["position_error"] is None else "{:.3f} m".format(row["position_error"]),
                rot="—" if row["rotation_error"] is None else "{:.3f} rad".format(row["rotation_error"]),
                links="".join(links), video=video)
        tactile = row.get("tactile", {})
        if tactile.get("summary_exists"):
            metrics = tactile.get("metrics", {})
            tactile_links = [
                '<a href="{}" target="_blank">tactile summary.json</a>'.format(
                    html.escape(relative_href(tactile["summary"])))
            ]
            if tactile.get("pressure_grids_exists"):
                tactile_links.append('<a href="{}" download>pressure_grids.npz</a>'.format(
                    html.escape(relative_href(tactile["pressure_grids"]))))
            if tactile.get("trajectory_exists"):
                tactile_links.append('<a href="{}" download>tactile trajectory.npz</a>'.format(
                    html.escape(relative_href(tactile["trajectory"]))))
            tactile_video = ""
            if tactile.get("video_exists"):
                tactile_links.append('<a href="{}" target="_blank">open tactile video</a>'.format(
                    html.escape(relative_href(tactile["video"]))))
                tactile_video = '<video controls playsinline preload="metadata" src="{}">Your browser cannot play this MP4.</video>'.format(
                    html.escape(relative_href(tactile["video"])))
            evaluation_panel += """
            <details class="evaluation tactile-panel"><summary><span class="eval-kind">tactile · Pa</span> View EgoTouch-layout pressure data &amp; video</summary>
              <div class="eval-grid"><div><span>Rollout success</span><strong>{success}</strong></div><div><span>Frames</span><strong>{frames}</strong></div><div><span>Contacts</span><strong>{contacts}</strong></div><div><span>Peak pressure</span><strong>{peak}</strong></div></div>
              <div class="artifact-links">{links}</div>{video}
            </details>""".format(
                success=format_percent(metrics.get("episode_success_rate")),
                frames="{:,}".format(metrics.get("steps", 0)),
                contacts="{:,}".format(metrics.get("total_object_hand_contacts", 0)),
                peak="{:.1f} kPa".format(metrics.get("max_pressure_pa", 0.0) / 1000.0),
                links="".join(tactile_links), video=tactile_video)
        paired = row.get("paired_tactile", {})
        if paired.get("summary_exists"):
            metrics = paired.get("metrics", {})
            paired_links = [
                '<a href="{}" target="_blank">paired summary.json</a>'.format(
                    html.escape(relative_href(paired["summary"])))
            ]
            if paired.get("side_by_side_video_exists"):
                paired_links.append('<a href="{}" target="_blank">open side-by-side video</a>'.format(
                    html.escape(relative_href(paired["side_by_side_video"]))))
            if paired.get("rgb_video_exists"):
                paired_links.append('<a href="{}" target="_blank">rgb.mp4</a>'.format(
                    html.escape(relative_href(paired["rgb_video"]))))
            if paired.get("tactile_video_exists"):
                paired_links.append('<a href="{}" target="_blank">tactile.mp4</a>'.format(
                    html.escape(relative_href(paired["tactile_video"]))))
            if paired.get("pressure_grids_exists"):
                paired_links.append('<a href="{}" download>pressure_grids.npz</a>'.format(
                    html.escape(relative_href(paired["pressure_grids"]))))
            if paired.get("trajectory_exists"):
                paired_links.append('<a href="{}" download>trajectory.npz</a>'.format(
                    html.escape(relative_href(paired["trajectory"]))))
            paired_video = ""
            if paired.get("side_by_side_video_exists"):
                paired_video = '<video controls playsinline preload="metadata" style="max-width:100%;aspect-ratio:10/3" src="{}">Your browser cannot play this MP4.</video>'.format(
                    html.escape(relative_href(paired["side_by_side_video"])))
            evaluation_panel += """
            <details class="evaluation tactile-panel"><summary><span class="eval-kind">rgb + tactile</span> View synchronized paired rollout</summary>
              <div class="eval-grid"><div><span>Rollout success</span><strong>{success}</strong></div><div><span>Video frames</span><strong>{frames}</strong></div><div><span>Contacts</span><strong>{contacts}</strong></div><div><span>Peak pressure</span><strong>{peak}</strong></div></div>
              <div class="artifact-links">{links}</div>{video}
            </details>""".format(
                success=format_percent(metrics.get("episode_success_rate")),
                frames="{:,}".format(metrics.get("video_frames", 0)),
                contacts="{:,}".format(metrics.get("total_object_hand_contacts", 0)),
                peak="{:.1f} kPa".format(metrics.get("max_pressure_pa", 0.0) / 1000.0),
                links="".join(paired_links), video=paired_video)
        hora = row.get("hora_demo", {})
        if hora.get("primary_video_exists"):
            metrics = hora.get("metrics", {})
            validation = hora.get("validation_metrics", {})
            batch = validation.get("batch_evaluation", {}) if isinstance(validation, dict) else {}
            hora_links = [
                '<a href="{}" target="_blank">open textured demo</a>'.format(
                    html.escape(relative_href(hora["primary_video"])))
            ]
            if hora.get("alternate_video_exists"):
                hora_links.append('<a href="{}" target="_blank">physical-camera demo</a>'.format(
                    html.escape(relative_href(hora["alternate_video"]))))
            if hora.get("textured_summary_exists"):
                hora_links.append('<a href="{}" target="_blank">textured summary.json</a>'.format(
                    html.escape(relative_href(hora["textured_summary"]))))
            if hora.get("physical_summary_exists"):
                hora_links.append('<a href="{}" target="_blank">physical-camera summary.json</a>'.format(
                    html.escape(relative_href(hora["physical_summary"]))))
            if hora.get("validation_exists"):
                hora_links.append('<a href="{}" target="_blank">physical validation.json</a>'.format(
                    html.escape(relative_href(hora["validation"]))))
            if hora.get("config_exists"):
                hora_links.append('<a href="{}" target="_blank">HORA config.yaml</a>'.format(
                    html.escape(relative_href(hora["config"]))))
            hora_video = '<video controls playsinline preload="metadata" style="max-width:100%;aspect-ratio:119/32" src="{}">Your browser cannot play this MP4.</video>'.format(
                html.escape(relative_href(hora["primary_video"])))
            evaluation_panel += """
            <details class="evaluation tactile-panel"><summary><span class="eval-kind">HORA v4 demo</span> View stronger cube in-hand rotation reference</summary>
              <div class="eval-grid"><div><span>Repo / recipe</span><strong>HORA + RWS warm start</strong></div><div><span>Object</span><strong>ShadowHand cube</strong></div><div><span>No-drop rollout</span><strong>{stable}</strong></div><div><span>Verified 2π</span><strong>{verified}</strong></div></div>
              <p style="color:#b8c9d8;margin:8px 0 10px">Reference demo from the separate HORA ShadowHandCubeRotation run: 2048-env PPO, torque control, cube-only scale 0.7, grasp cache <code>shadow_cube_grasp_50k_s07.npy</code>, initialized from the RWS warm checkpoint. Qualitatively it holds/rotates the cube better than the Bi-DexHands ReOrientation policy, but current summaries mark it as stable partial rotation rather than a verified full 2π success under the corrected physical metric. Best rollout: {rotation} rad cumulative rotation; batch audit: {successes}/{envs} full-turn successes, max {max_rotation} rad, drops {drops}.</p>
              <div class="artifact-links">{links}</div>{video}
            </details>""".format(
                stable="yes" if metrics.get("stable_no_drop") else "unknown",
                verified="yes" if metrics.get("verified_success") else "not yet",
                rotation=format_number(metrics.get("max_cumulative_rotation_rad"), 3),
                successes=batch.get("successes", 0),
                envs=batch.get("num_envs", 0),
                max_rotation=format_number(batch.get("max_rotation_rad"), 3),
                drops=batch.get("drops", 0),
                links="".join(hora_links),
                video=hora_video)
        body_rows.append("""
        <tr data-status="{status}" data-search="{search}">
          <td><div class="task">{task}</div><div class="family">{family}</div></td>
          <td><span class="gpu">GPU {gpu}</span></td>
          <td><span class="status {status}">{status}</span></td>
          <td>{iteration}</td><td>{reward}</td><td>{train_success}</td><td>{eval_success}</td>
          <td>{eta}</td><td class="paths">{paths}{evaluation_panel}</td>
        </tr>""".format(
            status=row["status"], search=html.escape((row["task"] + " " + row["family"]).lower()),
            task=html.escape(row["task"]), family=html.escape(row["family"]), gpu=row["gpu"],
            iteration=iteration, reward=reward, train_success=train_success,
            eval_success=eval_success, eta=format_eta(row["eta_seconds"]), paths="".join(artifact_lines),
            evaluation_panel=evaluation_panel))
    css = """
    :root{color-scheme:dark;--bg:#090d12;--panel:#111823;--line:#243044;--text:#eef3f8;--muted:#8fa0b5;--cyan:#59d8e6;--green:#6ce3a0;--amber:#f3c969;--red:#ff7e84}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0%,#142334 0,transparent 34%),var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
    main{max-width:1560px;margin:auto;padding:42px 28px 72px}header{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:28px}h1{font-size:clamp(28px,4vw,52px);letter-spacing:-.045em;margin:0}.eyebrow{color:var(--cyan);font-weight:700;letter-spacing:.13em;text-transform:uppercase;font-size:12px}.subtitle{color:var(--muted);max-width:720px;margin:10px 0 0}.policy{border:1px solid #295167;background:#10212c;padding:12px 16px;border-radius:12px;color:#bfeef3;white-space:nowrap}.metrics{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px;margin:26px 0}.metric{background:linear-gradient(145deg,#151e2a,#0f151e);border:1px solid var(--line);border-radius:16px;padding:18px}.metric strong{display:block;font-size:30px}.metric span{color:var(--muted)}.cube-panel{display:grid;grid-template-columns:minmax(260px,.7fr) minmax(560px,1.3fr);gap:20px;border:1px solid #31566a;background:linear-gradient(135deg,#10232d,#111823);border-radius:18px;padding:20px;margin:24px 0}.cube-panel h2{margin:4px 0;font-size:25px}.cube-panel p{color:#b8c9d8;margin:8px 0}.cube-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.cube-grid div{background:#0b151e;border:1px solid #24394b;border-radius:10px;padding:10px}.cube-grid span{display:block;color:#7f92a7;font-size:10px;text-transform:uppercase}.cube-grid strong{font-size:13px}.cube-paths{grid-column:1/-1;display:grid;gap:4px}.cube-paths code{font:10px/1.35 ui-monospace,monospace;color:#8faabd;overflow-wrap:anywhere}
    .toolbar{display:flex;gap:10px;align-items:center;margin:22px 0 12px}.toolbar input{flex:1;min-width:180px;background:#0e151e;border:1px solid var(--line);border-radius:10px;padding:11px 13px;color:var(--text)}button{border:1px solid var(--line);background:#121b26;color:var(--muted);padding:10px 12px;border-radius:10px;cursor:pointer}button.active{color:#081013;background:var(--cyan);border-color:var(--cyan)}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:16px;background:rgba(14,20,29,.9)}table{width:100%;border-collapse:collapse;min-width:1320px}th{position:sticky;top:0;background:#121a25;color:#9eb0c4;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.08em;padding:13px 12px;border-bottom:1px solid var(--line)}td{padding:14px 12px;border-bottom:1px solid #1c2736;vertical-align:top}.task{font-weight:720}.family{font-size:12px;color:var(--muted);margin-top:3px}.gpu{font-family:ui-monospace,monospace;color:#c5d3e2}.status{display:inline-block;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.05em}.status.training{color:#061517;background:var(--cyan)}.status.evaluated{color:#07130d;background:var(--green)}.status.trained{color:#101409;background:#b9de7a}.status.queued{color:#d4deeb;background:#2a3545}.status.paused{color:#181204;background:var(--amber)}.status.failed{color:#190608;background:var(--red)}
    .paths{min-width:520px}.paths>div{display:grid;grid-template-columns:38px 1fr;gap:6px;margin-bottom:5px}.paths>div>span{color:#6f8298;font-size:10px;text-transform:uppercase;padding-top:2px}.paths code{font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;color:#b9c8d8;overflow-wrap:anywhere}.evaluation{margin-top:12px;border:1px solid #294058;border-radius:12px;background:#0b121b;padding:10px}.evaluation summary{cursor:pointer;color:#cdeaf0;font-weight:650}.eval-kind{display:inline-block;margin-right:8px;padding:3px 7px;border-radius:999px;background:#284456;color:#8ce7f1;font-size:10px;text-transform:uppercase}.eval-grid{display:grid!important;grid-template-columns:repeat(4,1fr)!important;gap:7px!important;margin:12px 0!important}.eval-grid div{background:#111d29;border-radius:8px;padding:9px}.eval-grid span{display:block;color:#7f92a7;font-size:10px;text-transform:uppercase}.eval-grid strong{display:block;margin-top:3px;font-size:13px}.artifact-links{display:flex!important;grid-template-columns:none!important;gap:8px!important;flex-wrap:wrap;margin:8px 0!important}.artifact-links a{color:#8ee8f1;text-decoration:none;border:1px solid #31566a;border-radius:8px;padding:6px 9px;font-size:11px}.evaluation video{display:block;width:100%;max-width:480px;aspect-ratio:4/3;background:#000;border-radius:10px;margin-top:10px}.tactile-panel{border-color:#31566a}.tactile-panel video{max-width:720px;aspect-ratio:2/1}.foot{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:14px}@media(max-width:760px){main{padding:28px 14px}header{display:block}.policy{margin-top:16px;white-space:normal}.metrics{grid-template-columns:repeat(2,1fr)}.toolbar{flex-wrap:wrap}.eval-grid{grid-template-columns:repeat(2,1fr)!important}}
    """
    data = json.dumps({"tasks": rows, "inhand_rotation": cube}, sort_keys=True)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>Bi-DexHands Simulator Data Pipeline</title><style>{css}</style></head>
    <body><main><header><div><div class="eyebrow">Simulator data collection</div><h1>Bi-DexHands pipeline</h1><p class="subtitle">PPO training → deterministic rollout → native task success + joint/object audit → RGB, trajectory, and EgoTouch-layout pressure (Pa) export.</p></div><div class="policy">After the current GPU 5/6 jobs: <strong>maximum 4 GPUs</strong> (0–3)</div></header>
    {cube_panel}<section class="metrics">{cards}</section><div class="toolbar"><input id="search" aria-label="Filter tasks" placeholder="Filter by task or family…"><button class="active" data-filter="all">All</button><button data-filter="training">Training</button><button data-filter="evaluated">Evaluated</button><button data-filter="queued">Leftovers</button></div>
    <div class="table-wrap"><table><thead><tr><th>Task</th><th>GPU</th><th>Status</th><th>Iteration</th><th>Mean reward</th><th>Train success</th><th>Eval success</th><th>ETA</th><th>Files</th></tr></thead><tbody>{rows}</tbody></table></div>
    <div class="foot"><span>Updated {now}</span><span>Auto-refresh: 60 s · source: training logs + evaluation summaries</span></div></main>
    <script>const rows=[...document.querySelectorAll('tbody tr')],q=document.querySelector('#search'),buttons=[...document.querySelectorAll('button[data-filter]')];let filter='all';function apply(){{const s=q.value.toLowerCase();rows.forEach(r=>{{const status=r.dataset.status,matchFilter=filter==='all'||(filter==='queued'?['queued','paused','failed'].includes(status):status===filter),matchText=r.dataset.search.includes(s);r.hidden=!(matchFilter&&matchText)}})}}q.addEventListener('input',apply);buttons.forEach(b=>b.addEventListener('click',()=>{{filter=b.dataset.filter;buttons.forEach(x=>x.classList.toggle('active',x===b));apply()}}));window.PIPELINE_DATA={data};</script></body></html>""".format(css=css, cube_panel=cube_panel, cards=card_html, rows="".join(body_rows), now=now, data=data)


def write_once(output):
    rows = collect()
    cube = collect_cube()
    document = render(rows, cube)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w") as handle:
        handle.write(document)
    os.replace(temporary, output)
    with open(os.path.splitext(output)[0] + ".json", "w") as handle:
        json.dump({"generated_at": dt.datetime.utcnow().isoformat() + "Z", "tasks": rows,
                   "inhand_rotation": cube}, handle,
                  indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(PIPELINE, "pipeline_status.html"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    while True:
        write_once(args.output)
        if not args.watch:
            break
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
