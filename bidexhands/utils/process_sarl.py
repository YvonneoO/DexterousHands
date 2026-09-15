from bidexhands.algorithms.rl.ppo import PPO
from bidexhands.algorithms.rl.sac import SAC
from bidexhands.algorithms.rl.td3 import TD3
from bidexhands.algorithms.rl.ddpg import DDPG
from bidexhands.algorithms.rl.trpo import TRPO
import os

def process_sarl(args, env, cfg_train, logdir):
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        # ⚠️ Must mirror args.test, NOT be forced True (found live 2026-09-15):
        # is_testing=True is passed straight into PPO(..., is_testing=...), and
        # PPO.run() branches on self.is_testing into an UNBOUNDED "while True"
        # inference loop with no training, no logging, no checkpoint saves --
        # see ppo.py's run(). Forcing is_testing=True just because --model_dir
        # was set (the normal way to RESUME training, not just to eval) meant
        # every resumed/auto-resumed job across the whole Pen/Scissors
        # PREDTAC_BLOCKING chain silently ran an infinite inference loop
        # instead of training for its entire walltime (confirmed: tfevents
        # stuck at the 88-byte header, zero "Learning iteration" log lines,
        # zero checkpoints beyond model_0.pt, across 11 separate job attempts
        # spanning Sep 14-15). This is a second, independent instance of the
        # same "any --model_dir forces eval-only" bug already fixed once in
        # train.py's own branch (commit f797bb4) -- that fix alone did NOT
        # cover this file, since process_sarl() builds the PPO object (with
        # is_testing baked in at construction time) before train.py's branch
        # ever runs.
        is_testing = args.test
        chkpt_path = args.model_dir

    if args.max_iterations != -1:
        cfg_train["learn"]["max_iterations"] = args.max_iterations

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

    """Set up the algo system for training or inferencing."""
    model = eval(args.algo.upper())(vec_env=env,
              cfg_train = cfg_train,
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if args.resume > 0 and args.model_dir == "":
        chkpt_path = os.path.join(logdir, "model_{}.pt".format(args.resume))
        if not os.path.isfile(chkpt_path):
            raise FileNotFoundError("Resume checkpoint not found: {}".format(chkpt_path))
        print("Resuming training from {}".format(chkpt_path))
        model.load(chkpt_path)

    # ppo.test("/home/hp-3070/logs/demo/scissors/ppo_seed0/model_6000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.load(chkpt_path)

    return model
