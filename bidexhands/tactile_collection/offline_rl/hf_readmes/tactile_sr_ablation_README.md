# Tactile-SR ablation: does tactile input help policy success rate?

Robot-policy success-rate ablation across two training paradigms (online PPO,
offline IQL) and two simulators (bidexhands, VTDexManip), asking one question:
does adding tactile input to the observation improve task success rate, and
does that hold up when the tactile channel is *predicted* rather than ground
truth?

Arms compared: **P-only** (proprioception alone) / **P+GT-tac** (proprioception
+ ground-truth simulator contact pressure) / **P+Pred-Tac** (proprioception +
tactile predicted offline by the v2-dit tactile-prediction model from RGB —
not yet trained for any task, see below).

## ⚠️ Known issue affecting `offline_rl_iql/*/*_p_gt_tac/` — check before using

**The `p_gt_tac` IQL checkpoints currently in this folder for `pen` and
`scissors` were trained WITHOUT tactile input normalization** and are
confirmed broken: Scissors peaks at 63% success then collapses to 0% by the
end of training; Pen never learns at all (flat 0-3%). Root cause (confirmed
2026-09-07): `train_iql.py` concatenated raw-Pascal contact pressure
(task-dependent vmax ranging ~255-19087 Pa) directly onto radian/meter-scale
proprioception with no `observation_scaler` ever configured on d3rlpy's
`IQLConfig` — a 2-4 order of magnitude input-scale mismatch that destabilizes
IQL's value-function optimization. **This is an optimization artifact, not
evidence that tactile input is unhelpful.**

Fix (commit `1f8e8a8`, `dexteroushands_fork`/`qianqian-tactile-overlay`):
`IQLConfig(observation_scaler=StandardObservationScaler())`, which auto-fits
a z-score normalizer from each training dataset's own statistics. Retraining
for both tasks was in progress as of 2026-09-07 — **check the
`"observation_scaler"` field in each run's `eval_sweep_result.json`** (e.g.
`"StandardObservationScaler"` = post-fix; `null` = pre-fix, unreliable,
don't trust the curve) **before using any `p_gt_tac` checkpoint from this
folder.** `p_only` checkpoints are unaffected (no tactile channel to
normalize) and are already final.

This fix has only been verified for the offline IQL pipeline
(`build_mdp_dataset.py`/`train_iql.py`). Whether bidexhands' or VTDexManip's
**online PPO** observation construction has an analogous unnormalized-scale
issue has NOT been checked — PPO implementations commonly include their own
running-mean/std observation normalization by default, but this has not been
confirmed true (or false) for these specific training scripts. Treat PPO
results as a separate, so-far-unaudited question.

## Layout

### `offline_rl_iql/<task>/<task>_<arm>/`
D3rlpy IQL trained on pre-collected successful-episode manifests (not live
env interaction). `<arm>` is `p_only` or `p_gt_tac` (`p_pred_tac` not yet
trained for any task — blocked on incomplete offline tactile-prediction
generation for Pen/Scissors). Each folder: per-checkpoint `model_<step>.d3`
sweep + `eval_sweep_result.json` (success-rate-vs-training-step curve, live
rollout eval, 30 episodes/checkpoint — a quick/biased estimate, not a final
number) + `iql_final.d3`/`iql_final_policy.pt` (last checkpoint).
Peak-success checkpoints are also uploaded separately as
`peak_model_<step>.d3` (or `peak_model_iql_final_all_zero.d3` where the
whole curve is flat 0%, e.g. pre-fix Pen `p_gt_tac`).

### `bidexhands/<task>_<arm>_seed<N>/`
Online PPO (bidexhands sim), tasks `pen`/`scissors`/`over`/`door`. `<arm>` is
`gttac` or `ponly`; canonical seeds per task are the ones actually reported
in the ablation table (`gttac` always `seed42`; `ponly` is `seed13` except
Pen's `seed0`). Other `ppo_seed<N>` directories that exist on VISION but are
NOT mirrored here are data-collection seeds (gathering diverse successful
episodes for the offline manifests above), not separate trained arms.

### `vtdexmanip/`
Online PPO (VTDexManip sim) across BottleCap/HandOver/ReorientDown/
ReorientUp/ScrewFaucet/Sliding tasks. `checkpoints/` currently has BOTH a
flat naming (`checkpoints/<task>-<arm>_seed<N>_model_<step>.pt`) and a nested
one (`checkpoints/<task>/<task>-<arm>_seed<N>_model_<step>.pt`) for the same
files — this is accidental duplication from an in-progress reorganization
toward "one folder per task", not two different checkpoints. **Prefer the
nested path**; the flat copies are slated for removal, don't rely on them
sticking around.

## Provenance

Checkpoints come from VISION (`/scratch/project/prj-02-phai-lab/yqq`), pushed
via `hf upload`/`upload_folder` under HF account `qqyang`. Code:
`github.com/YvonneoO/DexterousHands` (`qianqian-tactile-overlay` branch,
offline-RL pipeline) and `github.com/YvonneoO/Ego2Contact` (`main`,
sim-data-collection + v2-dit tactile-prediction pipeline).
