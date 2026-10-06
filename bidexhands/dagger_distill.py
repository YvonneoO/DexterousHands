"""DAgger distillation: a GT-tactile PPO teacher -> a student that sees a different observation
(online predicted tactile / P-only / GT-tactile) in the SAME simulator.

Why: online-PPO with the tactile predictor in the loop is ~700x lower throughput than the GT-tac
arm (predictor + camera per env), so RL from scratch is sample-starved. Imitation needs far fewer
on-policy samples: the student acts, the teacher (which sees GT tactile, computed by the task into
`task.teacher_obs` via cfg env.teacherObs) labels every visited state, labelled states are
aggregated, and the student regresses the teacher's actions.

Run like train.py (same CLI via get_args), configured through env vars:
  DAGGER_TEACHER_CKPT  (required) GT-tac PPO checkpoint (362-dim actor_critic state_dict, e.g. model_6500.pt)
  DAGGER_OUT           (required) output dir for student checkpoints / resume state
  DAGGER_ROUNDS        total rounds (default 100)
  DAGGER_STEPS         env steps per round (default = env episodeLength)
  DAGGER_BETA_ROUNDS   beta (prob. of executing the teacher's action) decays linearly 1 -> 0 over this many rounds (default 10)
  DAGGER_EPOCHS        gradient epochs over the aggregate after each round (default 4)
  DAGGER_LR / DAGGER_BATCH / DAGGER_MAX_SAMPLES   (5e-4 / 4096 / 400000)
  DAGGER_SAVE_EVERY    rounds between full resume-state saves incl. the dataset (default 5)
Student checkpoints (model_<round>.pt) are plain ActorCritic state_dicts, i.e. loadable by
`train.py --test --model_dir=...` / eval_ppo_sweep.py with the student's own cfg.
"""
import os
import sys
import time

import numpy as np

from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex

import torch  # must come after the isaacgym-importing modules above
import torch.nn as nn

from bidexhands.algorithms.rl.ppo import ActorCritic

TEACHER_OBS_DIM = 362  # GT-tac layout: 338 proprio + 2 hands x 12 links


def envf(name, default):
    return float(os.environ.get(name, default))


def envi(name, default):
    return int(float(os.environ.get(name, default)))


def build_actor_critic(obs_dim, env, cfg_train, device):
    learn_cfg = cfg_train["learn"]
    return ActorCritic((obs_dim,), env.state_space.shape, env.action_space.shape,
                       learn_cfg.get("init_noise_std", 0.3), cfg_train["policy"], asymmetric=False).to(device)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    cfg["env"]["teacherObs"] = True
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    device = env.rl_device

    teacher_ckpt = os.environ["DAGGER_TEACHER_CKPT"]
    out_dir = os.environ["DAGGER_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    rounds = envi("DAGGER_ROUNDS", 100)
    steps_per_round = envi("DAGGER_STEPS", cfg["env"]["episodeLength"])
    beta_rounds = envi("DAGGER_BETA_ROUNDS", 10)
    epochs = envi("DAGGER_EPOCHS", 4)
    lr = envf("DAGGER_LR", 5e-4)
    batch = envi("DAGGER_BATCH", 4096)
    cap = envi("DAGGER_MAX_SAMPLES", 400000)
    save_every = envi("DAGGER_SAVE_EVERY", 5)

    N = task.num_envs
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    clip_obs, clip_act = env.clip_obs, env.clip_actions
    print(f"[dagger] envs={N} student_obs_dim={obs_dim} act_dim={act_dim} steps/round={steps_per_round} "
          f"rounds={rounds} beta_rounds={beta_rounds} teacher={teacher_ckpt}", flush=True)

    teacher = build_actor_critic(TEACHER_OBS_DIM, env, cfg_train, device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    student = build_actor_critic(obs_dim, env, cfg_train, device)
    opt = torch.optim.Adam(student.actor.parameters(), lr=lr)

    X = torch.zeros(cap, obs_dim, device=device)
    Y = torch.zeros(cap, act_dim, device=device)
    size, ptr, round0 = 0, 0, 0
    state_path = os.path.join(out_dir, "dagger_state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device)
        student.load_state_dict(st["student"])
        opt.load_state_dict(st["opt"])
        size = int(st["size"])
        ptr = int(st["ptr"])
        X[:size] = st["X"].to(device)
        Y[:size] = st["Y"].to(device)
        round0 = int(st["round"]) + 1
        print(f"[dagger] resumed from round {st['round']} ({size} samples)", flush=True)

    def teacher_obs():
        return torch.clamp(task.teacher_obs, -clip_obs, clip_obs).to(device)

    obs = env.reset()
    if obs_dim == TEACHER_OBS_DIM:
        # GT-tac student: its own obs must equal the teacher obs (plumbing check)
        err = (obs - teacher_obs()).abs().max().item()
        print(f"[dagger] student-vs-teacher obs max abs diff after reset = {err:.3e}", flush=True)
        assert err < 1e-5, "teacher obs does not reproduce the GT-tac observation"

    t_start = time.time()
    for r in range(round0, rounds):
        beta = max(0.0, 1.0 - r / beta_rounds) if beta_rounds > 0 else 0.0
        new_x, new_y = [], []
        ep_done, ep_succ = 0, 0
        for _ in range(steps_per_round):
            with torch.no_grad():
                a_teacher = teacher.act_inference(teacher_obs())
                a_student = student.act_inference(obs)
                use_teacher = (torch.rand(N, device=device) < beta).unsqueeze(-1)
                actions = torch.where(use_teacher, a_teacher, a_student)
            new_x.append(obs.clone())
            new_y.append(torch.clamp(a_teacher, -clip_act, clip_act))
            next_obs, rews, dones, infos = env.step(actions)
            done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0 and isinstance(infos, dict) and "successes" in infos:
                s = infos["successes"][done_ids.to(infos["successes"].device)]
                ep_done += int(done_ids.numel())
                ep_succ += int(s.sum().item())
            obs = next_obs
        bx = torch.cat(new_x)
        by = torch.cat(new_y)
        with torch.no_grad():
            pre_loss = ((student.act_inference(bx) - by) ** 2).mean().item()  # on-policy imitation error, before training on it
        n_new = bx.shape[0]
        idx = (ptr + torch.arange(n_new, device=device)) % cap
        X[idx] = bx
        Y[idx] = by
        ptr = (ptr + n_new) % cap
        size = min(size + n_new, cap)

        student.train()
        last_loss = float("nan")
        for _ in range(epochs):
            perm = torch.randperm(size, device=device)
            for i in range(0, size, batch):
                bi = perm[i:i + batch]
                loss = ((student.act_inference(X[bi]) - Y[bi]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
                opt.step()
                last_loss = loss.item()
        student.eval()

        sr = ep_succ / ep_done if ep_done > 0 else float("nan")
        print(f"[dagger] round={r} beta={beta:.2f} samples={size} onpolicy_mse={pre_loss:.5f} train_mse={last_loss:.5f} "
              f"episodes={ep_done} success={sr:.3f} elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)

        atomic_save(student.state_dict(), os.path.join(out_dir, f"model_{r}.pt"))
        if (r + 1) % save_every == 0 or r == rounds - 1:
            atomic_save({"student": student.state_dict(), "opt": opt.state_dict(), "size": size, "ptr": ptr,
                         "X": X[:size].cpu(), "Y": Y[:size].cpu(), "round": r}, state_path)
    print("[dagger] DONE", flush=True)


if __name__ == "__main__":
    main()
