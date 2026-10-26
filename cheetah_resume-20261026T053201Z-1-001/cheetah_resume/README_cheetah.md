# Cheetah-Run — 6-Method Experiment (review + resume version)
## Theory: AUG+CURL+RL > CURL+RL > AUG+RL > PPO

This is the CarRacing-v3 `carracing_review_changes` version (equal budgets, 3 seeds,
deterministic evaluation, crash-safe resume) ported to DeepMind Control Suite `cheetah:run`.
**Only the environment layer (`dmc_common.py` Section 4) differs from the CarRacing `common.py`.**
Everything else — [REVIEW-1..5] and [RESUME] — is the same code.

## File layout
```
dmc_common.py            shared infrastructure (CheetahPixelEnv + all review/resume logic)
train_cr_ppo.py          Method 1: PPO
train_cr_aug_rl.py       Method 2: AUG + RL
train_cr_ssl_rl.py       Method 3: AUTO (denoising AE) + RL
train_cr_aug_ssl_rl.py   Method 4: AUTO + RL + AUG
train_cr_curl_rl.py      Method 5: CURL + RL
train_cr_aug_curl_rl.py  Method 6: AUG + CURL + RL
cheetah_aggregate.py     across-seed + matched-step tables
run_cheetah.ps1 / .sh    all 6 methods x seeds 42,123,2024, resumable
diagnose_mujoco_gl.py    finds a working MUJOCO_GL backend
```
Results: `cheetah_vscoderesult/seed_<SEED>/`

## Setup
```
pip install dm_control "stable-baselines3[extra]" "imageio[ffmpeg]" opencv-python-headless torch torchvision
```
(`dmc_common.py` no longer auto-pip-installs; it was a CarRacing/box2d leftover.)
MuJoCo GL backend: Windows -> `glfw`, Linux/WSL -> `egl`. Override by setting `MUJOCO_GL`
yourself first (e.g. `set MUJOCO_GL=osmesa`). If rendering fails, run `python diagnose_mujoco_gl.py`.

## Run
```
python train_cr_ppo.py --seed 42          # one method, one seed (default seed 42)
.\run_cheetah.ps1                         # Windows: everything, 3 seeds, 2 jobs at a time
./run_cheetah.sh                          # Linux
python cheetah_aggregate.py               # tables after training
```
Stop any time (Ctrl+C / crash) and start the same command again: finished (seed, method)
pairs are skipped (`done_<M>.json`), unfinished ones continue from `resume_<M>.zip`
(auto-saved every 250k RL steps, `RESUME_EVERY` in `dmc_common.py`). Pretraining is not repeated.

## What changed versus your old Cheetah files
| | old Cheetah code | now |
|---|---|---|
| Budgets | 2.5M / 3.5M / 4M / 4M / 5.5M / 6.5M | `COMMON_STEPS = 6_500_000` for all (pretraining frames subtracted), warm-up = 46 % of RL steps |
| Fine-tune LR | 1e-6 for AUG+CURL+RL, 5e-6 others | 5e-6 for all |
| Seeds | fixed 2024, one folder | `--seed`, folder per seed |
| Augmentation rule | AUTO+RL+AUG and AUG+CURL+RL used their own heavier pretraining; CURL+RL used CURL views in RL | pretraining identical inside each pair; AUG = random shift on RL observations only; CURL+RL RL phase is clean |
| Evaluation | training-episode rewards only | + `eval_<M>.json`: 10 deterministic fixed-seed episodes at 1M/2.5M/4M total steps and at the end |
| Extras | — | `run_info_<M>.json`, `hyperparameters.json`, resume system; old ad-hoc resume block in `train_cr_curl_rl.py` replaced |
| Env | unchanged `CheetahPixelEnv` (dm_control, 96x96 -> 64x64), with one tweak: the old suite env is closed before a `reset(seed=...)` reload so GL contexts are not leaked |

Cheetah-specific notes: episodes are 1000 steps, rewards in [0, 1000], no early termination
(every episode ends by time-limit truncation). The `>=700` and `rolling-20 >= 500` thresholds in
`cheetah_aggregate.py` are kept from CarRacing.

## Not tested / known issues (carried over unchanged from CarRacing so results stay comparable)
1. `SSLExtractorCNN.forward` reshapes obs as (B,64,64,12) but SB3 delivers channel-first (B,12,64,64) -> pretrained encoder sees scrambled pixels in the four AUTO/CURL methods.
2. `UnfreezeEncoderCB` sets the optimizer LR, but SB3 resets it every update, so `FINETUNE_LR` has no effect.
3. `aug_random_shift` has no scale term (slight zoom even at zero shift).
These were NOT changed. Fixing them changes results vs CarRacing; ask for patches if you want them.
