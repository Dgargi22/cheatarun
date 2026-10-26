"""
dmc_common.py — Shared infrastructure for Cheetah-Run 6-Method Experiment
============================================================================
Cheetah-Run port of the CarRacing-v3 `common.py` INCLUDING all review/resume
changes ([REVIEW-1..5] equal budgets, per-seed folders, deterministic
evaluation, identical-pretraining rule, hyperparameters.json, and [RESUME]
crash-safe training). Only the environment layer (Section 4) differs: it
uses DeepMind Control Suite `cheetah:run` via suite.load() + pixels.Wrapper()
(Tassa et al. 2018; Tunyasuvunakool et al. 2020).
"""
import sys
import io
import os  
# Force UTF-8 on Windows before anything else
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Offscreen OpenGL backend for MuJoCo. Windows -> glfw, Linux/WSL -> egl.
# To override, set MUJOCO_GL yourself before running (e.g. `set MUJOCO_GL=osmesa`).
# Run diagnose_mujoco_gl.py if rendering fails.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

# SECTION 0 — Dependencies are NOT auto-installed (dm_control needs a matching mujoco):
#   pip install dm_control "stable-baselines3[extra]" "imageio[ffmpeg]" opencv-python-headless torch torchvision


# ───────────────────────────────────────────────────────────────────
# SECTION 1 — Imports
# ───────────────────────────────────────────────────────────────────
import copy, random, os, json, csv, datetime, platform, shutil, zipfile
from collections import deque
import gymnasium as gym
import numpy as np
import cv2, imageio, warnings
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dm_control import suite
from dm_control.suite.wrappers import pixels

warnings.filterwarnings("ignore")
print(" Imports done")


# ───────────────────────────────────────────────────────────────────
# SECTION 2 — Global seed
# ───────────────────────────────────────────────────────────────────
# [REVIEW-2] Seed is now chosen on the command line:  python train_cr_ppo.py --seed 123
#            Use the SAME three seeds (42, 123, 2024) for EVERY method. Default stays 42.
import argparse
_seed_parser = argparse.ArgumentParser(add_help=False)
_seed_parser.add_argument("--seed", type=int, default=42)
GLOBAL_SEED = _seed_parser.parse_known_args()[0].seed

def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"   Global seed = {seed}")

set_global_seed()


# ───────────────────────────────────────────────────────────────────
# SECTION 3 — Global config  [TUNED]
# ───────────────────────────────────────────────────────────────────
DOMAIN_NAME   = "cheetah"   # dm_control Control Suite domain
TASK_NAME     = "run"
ENV_ID        = f"dm_control/{DOMAIN_NAME}-{TASK_NAME}"   # label only (used in hyperparameters.json)
IMG_H, IMG_W  = 64, 64
N_ENVS        = 8
VIDEO_EPS     = 5
SOLVE_THRESH  = 700


N_STACK   = 4
ENC_IN_CH = 3 * N_STACK   # 12-channel input to all encoders


LATENT_DIM = 256   # was 128

# CURL model hyper-params 
CURL_TAU      = 0.997   # was 0.995 — more stable momentum encoder
CURL_TEMP     = 0.03    # was 0.05 — sharper contrastive distribution
CURL_PROJ_DIM = 256     # was 128  — bigger projection head

# All results land here — shared across all method scripts
# [REVIEW-2] one sub-folder per seed -> seeds can never overwrite each other
RESULTS_BASE = "cheetah_vscoderesult"
RESULTS_DIR  = os.path.join(RESULTS_BASE, f"seed_{GLOBAL_SEED}")

# Pretraining budgets  
AE_FRAME_BUF   = 80_000   # was 60k
CURL_FRAME_BUF = 80_000   # was 60k
AE_EPOCHS      = 350      # was 250
AE_EPOCHS_AUG  = 400      # was 300
CURL_EPOCHS    = 800      # was 500
CURL_EPOCHS_HV = 1000      # was 700
CURL_BATCH     = 512

# [REVIEW-1] SAME training budget for ALL six methods (was 2.5M / 3.5M / 4M / 4M / 5.5M / 6.5M).
#   COMMON_STEPS = total environment transitions per run INCLUDING the random-policy
#   pretraining frames (the pretraining frames are collected with env steps too).
#   RL steps = COMMON_STEPS - pretraining frames, so every method sees the same total
#   amount of environment experience.  (PPO's rollout rounding adds < 25k steps.)
#   6.5M = the largest budget used before, i.e. PPO is now also trained for 6.5M.
COMMON_STEPS = 6_500_000

PRETRAIN_FRAMES = {
    "PPO":          0,
    "AUG+RL":       0,
    "AUTO+RL":      AE_FRAME_BUF,
    "AUTO+RL+AUG":  AE_FRAME_BUF,
    "CURL+RL":      CURL_FRAME_BUF,
    "AUG+CURL+RL":  CURL_FRAME_BUF,
}
N_STEPS = {m: COMMON_STEPS - PRETRAIN_FRAMES[m] for m in PRETRAIN_FRAMES}   # RL steps

# [REVIEW-1] Encoder warm-up (frozen) is the SAME fraction of the RL budget for all SSL methods
#            (was 1.8M / 1.8M / 2.5M / 3.0M, i.e. 45-46 % of each different budget).
WARMUP_FRACTION = 0.46
N_WARMUP_STEPS = {m: int(WARMUP_FRACTION * N_STEPS[m])
                  for m in ("AUTO+RL", "AUTO+RL+AUG", "CURL+RL", "AUG+CURL+RL")}

# [REVIEW-1] Fine-tune LR after unfreeze: SAME for all SSL methods
#            (AUG+CURL+RL used 1e-6, the others 5e-6) so that CURL+RL and AUG+CURL+RL
#            differ ONLY in the augmentation.
FINETUNE_LR = {m: 5e-6 for m in N_WARMUP_STEPS}

#

AUG_JITTER_ROLLOUT  = 0.00   # no jitter at rollout time
AUG_JITTER_PRETRAIN = 0.05   # brightness-only during CURL pretrain
AUG_CUTOUT_SZ       = 0      # cutout DISABLED — hides road boundaries


PPO_KW_CNN = dict(
    learning_rate  = 3e-4,
    n_steps        = 1024,
    batch_size     = 256,
    n_epochs       = 15,
    gamma          = 0.99,
    gae_lambda     = 0.97,
    ent_coef       = 0.005,
    clip_range     = 0.2,
    max_grad_norm  = 0.5,
    vf_coef        = 0.5,
    seed           = GLOBAL_SEED,
    verbose        = 0,
)


PPO_KW_MLP = dict(
    learning_rate  = 1.5e-4,        # (was 2e-4)
    n_steps        = 3072,          # (was 2048)
    batch_size     = 256,           # (was 512)
    n_epochs       = 20,            # (was 15)
    gamma          = 0.99,
    gae_lambda     = 0.98,          # (was 0.97)
    ent_coef       = 0.002,         # (was 0.003)
    clip_range     = 0.15,          # (was 0.2)
    max_grad_norm  = 0.3,           # (was 0.5)
    vf_coef        = 0.7,           # (was 0.5)
    seed           = GLOBAL_SEED,
    verbose        = 0,
)

# Method colours
COL = {
    "PPO":         dict(hex="#388bfd", bgr=(232, 139,  56)),
    "AUG+RL":      dict(hex="#f5a623", bgr=( 35, 166, 245)),
    "AUTO+RL":      dict(hex="#2ea043", bgr=( 33, 160,  46)),
    "AUTO+RL+AUG":  dict(hex="#b44fdc", bgr=(220,  79, 180)),
    "CURL+RL":     dict(hex="#e05c5c", bgr=( 60,  92, 224)),
    "AUG+CURL+RL": dict(hex="#20c9a6", bgr=( 30, 200, 150)),
}

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(" Config ready")


# ───────────────────────────────────────────────────────────────────
# SECTION 4 — Cheetah-Run base environment (dm_control) + resize wrapper + frame stacker
# ───────────────────────────────────────────────────────────────────
# Native render resolution before downsizing to IMG_H x IMG_W — mirrors
# CarRacing-v3's native ~96x96 frame being resized down to 64x64.
NATIVE_RENDER = 96
CAMERA_ID     = 0


class CheetahPixelEnv(gym.Env):
    """
    Minimal gymnasium.Env adapter around dm_control's `cheetah:run` task.
    Follows the RL API documented in the reference papers: action_spec() /
    observation_spec() -> TimeStep(step_type, reward, discount, observation).
    Emits HWC uint8 RGB frames at NATIVE_RENDER resolution (downsized to
    IMG_H x IMG_W by ResizeWrapper below, exactly like CarRacing-v3's own
    native-resolution-then-resize pipeline).
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, seed=0):
        super().__init__()
        self.render_mode = render_mode
        self._env = suite.load(
            domain_name=DOMAIN_NAME, task_name=TASK_NAME,
            task_kwargs={"random": seed},
        )
        self._env = pixels.Wrapper(
            self._env, pixels_only=True,
            render_kwargs={"height": NATIVE_RENDER, "width": NATIVE_RENDER,
                          "camera_id": CAMERA_ID},
        )
        spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=spec.minimum.astype(np.float32),
            high=spec.maximum.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(NATIVE_RENDER, NATIVE_RENDER, 3), dtype=np.uint8,
        )
        self._time_step = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # dm_control's RNG is fixed at suite.load() time, not at
            # reset() time — SB3's make_vec_env calls env.reset(seed=...)
            # once per sub-env right after creation to differentiate the
            # n_envs parallel copies, so we honour that by reloading here.
            try:
                self._env.close()          # free the old physics / GL context first
            except Exception:
                pass
            self._env = suite.load(
                domain_name=DOMAIN_NAME, task_name=TASK_NAME,
                task_kwargs={"random": seed},
            )
            self._env = pixels.Wrapper(
                self._env, pixels_only=True,
                render_kwargs={"height": NATIVE_RENDER, "width": NATIVE_RENDER,
                              "camera_id": CAMERA_ID},
            )
        ts = self._env.reset()
        self._time_step = ts
        return ts.observation["pixels"], {}

    def step(self, action):
        action = np.clip(np.asarray(action).flatten(),
                         self.action_space.low, self.action_space.high)
        ts = self._env.step(action)
        self._time_step = ts
        reward = float(ts.reward) if ts.reward is not None else 0.0
        # dm_control convention: discount==0 at LAST is a true terminal
        # state; discount==1 (or None) at LAST is a time-limit truncation.
        terminated = bool(ts.last() and ts.discount is not None and ts.discount == 0.0)
        truncated  = bool(ts.last() and not terminated)
        return ts.observation["pixels"], reward, terminated, truncated, {}

    def render(self):
        """Separate, higher-quality frame for video recording — CarRacing-
        v3's env.render() likewise returns the env's native raw frame,
        independent of whatever resize/stack wrappers are applied to the
        training observation."""
        if self.render_mode != "rgb_array":
            return None
        return self._env.physics.render(height=240, width=320, camera_id=CAMERA_ID)

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


class ResizeWrapper(gym.ObservationWrapper):
    """Resize every observation to IMG_H × IMG_W RGB (HWC uint8)."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8)

    def observation(self, obs):
        return cv2.resize(obs, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


class FrameStackWrapper(gym.Wrapper):
    
    def __init__(self, env, n_stack=N_STACK):
        super().__init__(env)
        self.n_stack = n_stack
        self._buf    = deque(maxlen=n_stack)
        H, W, C      = env.observation_space.shape   # (64, 64, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(H, W, C * n_stack),              # (64, 64, 12)
            dtype=np.uint8)

    def _get_obs(self):
        # Concatenate along channel axis → (64, 64, 12)
        return np.concatenate(list(self._buf), axis=-1)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        for _ in range(self.n_stack):
            self._buf.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._buf.append(obs)
        return self._get_obs(), rew, term, trunc, info


# ── Internal factory: single-frame base env (used inside aug wrappers) ──
def _make_raw_env(render_mode=None):
    """Single-frame (H, W, 3) base env — used internally by aug wrappers."""
    env = CheetahPixelEnv(render_mode=render_mode, seed=GLOBAL_SEED)
    return ResizeWrapper(env)


# ── Public factory: frame-stacked base env (used by PPO and record_video) ──
def make_base_env(render_mode=None):
    """Frame-stacked (H, W, 12) base env — public interface."""
    return FrameStackWrapper(_make_raw_env(render_mode))


def obs_to_tensor(obs_uint8):
    """HWC uint8 → CHW float32 in [0,1]. Works for any channel count."""
    return torch.FloatTensor(obs_uint8).permute(2, 0, 1) / 255.0


def tensor_to_uint8(t):
    """CHW float32 → HWC uint8. Works for any channel count."""
    return (t.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


print("  Base env ready")


# ───────────────────────────────────────────────────────────────────
# SECTION 5 — Augmentation functions  [DrQ-v2 random-shift, unchanged from CarRacing-v3]
# ───────────────────────────────────────────────────────────────────


DRQ_PAD = 4   # pad each side by 4 px → random shift ± 4 px 


def aug_random_shift(img_t: torch.Tensor, pad: int = DRQ_PAD) -> torch.Tensor:
    
    C, H, W = img_t.shape
    # Step 1 — replicate-pad (avoids black border artefacts)
    x = img_t.unsqueeze(0)                                 # (1, C, H, W)
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")   # (1, C, H+2p, W+2p)
    _, _, Hp, Wp = x.shape

    # Step 2 — random translation in pixel units → normalised grid coords
    # Shift range: uniform over [-pad, pad] for both axes independently.
    tx = np.random.randint(-pad, pad + 1) / (Wp / 2.0)
    ty = np.random.randint(-pad, pad + 1) / (Hp / 2.0)

    # Build affine theta for pure translation (no rotation/scale)
    # theta = [[1, 0, tx], [0, 1, ty]]
    theta = torch.tensor([[1.0, 0.0, tx],
                           [0.0, 1.0, ty]],
                          dtype=torch.float32, device=img_t.device).unsqueeze(0)  

    # Step 3 — grid_sample with bilinear + align_corners=False 
    grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)
    out  = F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)
    return out.squeeze(0)   # (C, H, W)


def aug_random_shift_batch(imgs: torch.Tensor, pad: int = DRQ_PAD) -> torch.Tensor:
   
    B, C, H, W = imgs.shape
    # Replicate-pad the whole batch at once
    x  = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")  # (B, C, H+2p, W+2p)
    _, _, Hp, Wp = x.shape

    # Independent per-image shifts
    tx = (torch.randint(-pad, pad + 1, (B,)).float() / (Wp / 2.0))
    ty = (torch.randint(-pad, pad + 1, (B,)).float() / (Hp / 2.0))

    # Build per-image affine matrices  (B, 2, 3)
    zeros = torch.zeros(B)
    ones  = torch.ones(B)
    theta = torch.stack([
        torch.stack([ones,  zeros, tx], dim=1),
        torch.stack([zeros, ones,  ty], dim=1),
    ], dim=1)   # (B, 2, 3)

    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    grid = grid.to(imgs.device)
    out  = F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)
    return out   # (B, C, H, W)


# ── Additional augmentations used in CURL views ─────────

def aug_color_jitter(img_t, strength=0.05):
    
    img_t = TF.adjust_brightness(img_t, 1 + np.random.uniform(-strength, strength))
    return img_t.clamp(0, 1)


def aug_color_jitter_stacked(img_t, strength=0.05):
    
    C = img_t.shape[0]
    if C == 3:
        return aug_color_jitter(img_t, strength)
    n    = C // 3
    H, W = img_t.shape[1], img_t.shape[2]
    slcs = img_t.reshape(n, 3, H, W)
    out  = torch.stack([aug_color_jitter(slcs[i], strength) for i in range(n)])
    return out.reshape(C, H, W)


def aug_random_cutout(img_t, patch=AUG_CUTOUT_SZ):
    
    if patch <= 0:
        return img_t          
    C, H, W = img_t.shape
    x = np.random.randint(0, max(1, W - patch))
    y = np.random.randint(0, max(1, H - patch))
    img_t = img_t.clone()
    img_t[:, y:y + patch, x:x + patch] = 0.0
    return img_t


def aug_random_flip(img_t):
   
    return img_t 



def aug_view_A(img_t):
    
    return aug_color_jitter_stacked(
        aug_random_shift(img_t),
        strength=0.025)

def aug_view_B(img_t):
    
    return aug_color_jitter_stacked(
        aug_random_shift(img_t),
        strength=0.025)


def aug_heavy_A(img_t):
   
    return aug_color_jitter_stacked(
        aug_random_shift(img_t),
        strength=AUG_JITTER_PRETRAIN)   # 0.05 — brightness only

def aug_heavy_B(img_t):
    
    return aug_color_jitter_stacked(
        aug_random_shift(img_t),
        strength=AUG_JITTER_PRETRAIN)   # 0.05 — brightness only

def aug_heavy(img_t):
    
    return aug_random_shift(img_t)


print("  Augmentation functions ready (DrQ-v2 random-shift, CarRacing-style light-aug)")


# ───────────────────────────────────────────────────────────────────
# SECTION 6 — Augmentation environment wrappers
# ───────────────────────────────────────────────────────────────────
# NOTE: These wrappers operate on SINGLE frames (H, W, 3).
#       Frame stacking is applied by the outer factory lambdas below.
#       This keeps augmentation-per-step and temporal stacking independent.

class AugEnv(gym.Wrapper):
    

    def __init__(self):
        super().__init__(_make_raw_env())

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return tensor_to_uint8(aug_heavy(obs_to_tensor(obs))), info

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        return tensor_to_uint8(aug_heavy(obs_to_tensor(obs))), r, term, trunc, info


class CURLEnv(gym.Wrapper):
    
    def __init__(self):
        super().__init__(_make_raw_env())

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return tensor_to_uint8(aug_view_A(obs_to_tensor(obs))), info

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        return tensor_to_uint8(aug_view_A(obs_to_tensor(obs))), r, term, trunc, info


class AugCURLEnv(gym.Wrapper):
    
    def __init__(self):
        super().__init__(_make_raw_env())

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return tensor_to_uint8(aug_heavy_A(obs_to_tensor(obs))), info

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        return tensor_to_uint8(aug_heavy_A(obs_to_tensor(obs))), r, term, trunc, info


# Public factories — each returns a FRAME-STACKED env
make_aug_env      = lambda: FrameStackWrapper(AugEnv())
make_curl_env     = lambda: FrameStackWrapper(CURLEnv())
make_aug_curl_env = lambda: FrameStackWrapper(AugCURLEnv())

print("  Augmentation wrappers ready")


# ───────────────────────────────────────────────────────────────────
# SECTION 7 — CNN Encoder / Decoder / Autoencoder / CURL model  [T-2, T-3, T-6]
# ───────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    
    def __init__(self, in_ch=ENC_IN_CH, latent=LATENT_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32,  8, stride=4),           nn.ReLU(),
            nn.Conv2d(32,    64,  4, stride=2),            nn.ReLU(),
            nn.Conv2d(64,   128,  3, stride=1),            nn.ReLU(),
            nn.Conv2d(128,  128,  3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, in_ch, IMG_H, IMG_W)
        flat  = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(),
            nn.LayerNorm(512),               
            nn.Linear(512, latent),
        )

    def forward(self, x):
        return self.fc(self.cnn(x))


class CNNDecoder(nn.Module):
   
    def __init__(self, latent=LATENT_DIM, out_ch=ENC_IN_CH):
        super().__init__()
        h0, w0 = IMG_H // 8, IMG_W // 8
        self.h0, self.w0 = h0, w0
        self.fc = nn.Sequential(
            nn.Linear(latent, 512), nn.ReLU(),
            nn.Linear(512, 64 * h0 * w0), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64,    64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,    32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 64, self.h0, self.w0))


class CNNAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.enc = CNNEncoder(in_ch=ENC_IN_CH)
        self.dec = CNNDecoder(out_ch=ENC_IN_CH)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class CURLModel(nn.Module):
    
    def __init__(self, latent=LATENT_DIM, proj=CURL_PROJ_DIM,
                 tau=CURL_TAU, in_ch=ENC_IN_CH):
        super().__init__()
        self.tau  = tau
        self.temp = CURL_TEMP
        self.enc_o = CNNEncoder(in_ch=in_ch, latent=latent)
        self.prj_o = nn.Sequential(
            nn.Linear(latent, proj), nn.ReLU(),
            nn.Linear(proj, proj))
        self.enc_m = CNNEncoder(in_ch=in_ch, latent=latent)
        self.prj_m = nn.Sequential(
            nn.Linear(latent, proj), nn.ReLU(),
            nn.Linear(proj, proj))
        self._sync(init=True)

    @torch.no_grad()
    def _sync(self, init=False):
        for src, tgt in [(self.enc_o, self.enc_m), (self.prj_o, self.prj_m)]:
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                if init:
                    pt.data.copy_(ps.data)
                else:
                    pt.data.mul_(self.tau).add_((1 - self.tau) * ps.data)
        for m in [self.enc_m, self.prj_m]:
            for p in m.parameters():
                p.requires_grad_(False)

    def forward(self, va, vb):
        za = F.normalize(self.prj_o(self.enc_o(va)), dim=-1)
        with torch.no_grad():
            zb = F.normalize(self.prj_m(self.enc_m(vb)), dim=-1)
        sim    = torch.mm(za, zb.T) / self.temp   # sharper with temp=0.05
        labels = torch.arange(za.shape[0], device=za.device)
        loss   = F.cross_entropy(sim, labels)
        self._sync()
        return loss

    def get_encoder(self):
        return self.enc_o


print("  CNN models ready")


# ───────────────────────────────────────────────────────────────────
# SECTION 8 — SSL Feature Extractor for SB3  [T-2, T-10]
# ───────────────────────────────────────────────────────────────────
class SSLExtractorCNN(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, encoder):
        feat_dim = LATENT_DIM + 128     # 384; was 128+64=192
        super().__init__(observation_space, features_dim=feat_dim)
        self.enc        = encoder
        self._is_frozen = True
        for p in self.enc.parameters():
            p.requires_grad_(False)

        # [T-10] Wider live branch — more capacity for frame-stacked input
        self.live = nn.Sequential(
            nn.Conv2d(ENC_IN_CH, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32,        64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64,        64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        dummy     = torch.zeros(1, ENC_IN_CH, IMG_H, IMG_W)
        live_flat = self.live(dummy).shape[1]
        self.live_fc = nn.Linear(live_flat, 128)   # was 64

    def unfreeze_encoder(self):
        for p in self.enc.parameters():
            p.requires_grad_(True)
        self._is_frozen = False
        print("  [SSLExtractor] Encoder UNFROZEN for fine-tuning")

    def forward(self, obs):
       
       
        B = obs.shape[0]
        x = obs.reshape(B, IMG_H, IMG_W, ENC_IN_CH).permute(0, 3, 1, 2).contiguous()
        if self._is_frozen:
            with torch.no_grad():
                z = self.enc(x)
        else:
            z = self.enc(x)
        lf = F.relu(self.live_fc(self.live(x)))
        return torch.cat([z, lf], dim=-1)


def make_ssl_policy_kwargs(encoder):
    
    return dict(
        features_extractor_class  = SSLExtractorCNN,
        features_extractor_kwargs = dict(encoder=encoder),
        net_arch = [512, 256],   # was [256, 256]
    )


print("  SSLExtractorCNN ready")


# ───────────────────────────────────────────────────────────────────
# SECTION 9 — Pretraining routines
# ───────────────────────────────────────────────────────────────────
def _collect_frames(factory, n):
    
    env = factory() if callable(factory) else make_base_env()
    env.action_space.seed(GLOBAL_SEED)        # [REVIEW-2] random-policy data depends on the seed
    obs, _ = env.reset(seed=GLOBAL_SEED)
    # Pre-allocate uint8 array: 80k × 64 × 64 × 12 ≈ 3.7 GB  (vs 14.9 GB float32)
    buf = np.empty((n, IMG_H, IMG_W, ENC_IN_CH), dtype=np.uint8)
    for i in range(n):
        act = env.action_space.sample()
        obs_n, _, term, trunc, _ = env.step(act)
        buf[i] = obs_n                   
        if term or trunc:
            obs_n, _ = env.reset()
        obs = obs_n
        if (i + 1) % 20_000 == 0:
            print(f"    collected {i+1:,} / {n:,} frames...")
    env.close()
    return buf  

def pretrain_ae(use_aug=False, noise_level=0.05, epochs=AE_EPOCHS):
    
    label   = "augmented" if use_aug else "clean"
    print(f"\n  Collecting {AE_FRAME_BUF:,} {label} frames for AE pretraining...")
    
    frames = _collect_frames(make_base_env, AE_FRAME_BUF) 
    dev    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.empty_cache()   
    ae     = CNNAE().to(dev)
    opt    = torch.optim.Adam(ae.parameters(), lr=3e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    print(f"  Training AE ({epochs} epochs, noise={noise_level}) on {dev}...")
    for ep in range(epochs):
        idx      = torch.randperm(len(frames))[:256]
         
        x        = torch.from_numpy(frames[idx]).float().div(255.0).permute(0, 3, 1, 2).to(dev)
        
        if use_aug:
            x = aug_random_shift_batch(x)
        xn       = (x + noise_level * torch.randn_like(x)).clamp(0, 1)
        recon, _ = ae(xn)
        loss     = F.mse_loss(recon, x)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0) ;opt.step(); sched.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>3}/{epochs}  loss={loss.item():.5f}"
                  f"  lr={sched.get_last_lr()[0]:.2e}")
    ae.enc.cpu()
    print("   AE pretraining done")
    return ae.enc


def pretrain_curl(use_heavy=False, epochs=CURL_EPOCHS, batch=CURL_BATCH):
    
    label = "heavy" if use_heavy else "standard"
    print(f"\n  Collecting {CURL_FRAME_BUF:,} frames for CURL ({label})...")
    frames = _collect_frames(make_base_env, CURL_FRAME_BUF)  # uint8 numpy (n,H,W,C)
    dev    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.empty_cache()   # free any leftover allocations before pretraining
    model  = CURLModel().to(dev)
    opt    = torch.optim.Adam(
        list(model.enc_o.parameters()) + list(model.prj_o.parameters()),
        lr=1e-3, weight_decay=1e-4)   # small L2 regularisation added
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    fn_a   = aug_heavy_A if use_heavy else aug_view_A
    fn_b   = aug_heavy_B if use_heavy else aug_view_B
    print(f"  Training CURL ({epochs} epochs, batch={batch}) on {dev}...")
    best_loss = float("inf")
    best_enc  = None
    fn_a   = aug_heavy_A if use_heavy else aug_view_A
    fn_b   = aug_heavy_B if use_heavy else aug_view_B
    for ep in range(epochs):
        idx  = torch.randint(0, len(frames), (batch,))
        base = torch.from_numpy(frames[idx]).float().div(255.0).permute(0, 3, 1, 2).to(dev)
    
        va = torch.stack([fn_a(base[i]) for i in range(batch)])
        vb   = torch.stack([fn_b(base[i]) for i in range(batch)])
        
        loss = model(va, vb)  
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_enc  = copy.deepcopy(model.enc_o.state_dict())
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:>3}/{epochs}  loss={loss.item():.5f}"
                  f"  best={best_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}")
    model.enc_o.load_state_dict(best_enc)
    enc = model.get_encoder().cpu()
    print(f"    CURL pretraining done  (best loss={best_loss:.5f})")
    return enc


print(" Pretraining routines ready")


# ───────────────────────────────────────────────────────────────────
# SECTION 10 — Callbacks
# ───────────────────────────────────────────────────────────────────
class UnfreezeEncoderCB(BaseCallback):
    
    def __init__(self, unfreeze_at, finetune_lr=1.5e-5):
        super().__init__()
        self.unfreeze_at = unfreeze_at
        self.finetune_lr = finetune_lr
        self._unfrozen   = False

    def _on_step(self):
        if not self._unfrozen and self.num_timesteps >= self.unfreeze_at:
            fe = self.model.policy.features_extractor
            if hasattr(fe, "unfreeze_encoder"):
                fe.unfreeze_encoder()
                for pg in self.model.policy.optimizer.param_groups:
                    pg["lr"] = self.finetune_lr
                print(f"\n  [UnfreezeEncoderCB] step {self.num_timesteps:,}"
                      f" — encoder unfrozen, LR → {self.finetune_lr:.1e}\n")
            self._unfrozen = True
        return True


class LiveCB(BaseCallback):
    
    def __init__(self, method):
        super().__init__()
        self.method   = method
        self.ep_rews  = []
        self.ep_steps = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = len(self.ep_rews) + 1
                r  = info["episode"]["r"]
                s  = info["episode"]["l"]
                self.ep_rews.append(r)
                self.ep_steps.append(s)
                bar = int(np.clip(r / 950 * 30, 0, 30))
                # Use only ASCII symbols — avoid Unicode
                sym = "[OK]" if r >= 900 else ("OK" if r >= 700 else ("--" if r >= 200 else "XX"))
                try:
                    print(f"[{self.method}] Ep{ep:>5} "
                      f"|{'#'*bar+'.'*(30-bar)}| {r:+8.1f} | {s:>5}s {sym}",
                      flush=True)
                except UnicodeEncodeError:
                # Fallback for Windows cp1252 encoding
                    sys.stdout.buffer.write(
                        f"[{self.method}] Ep{ep:>5} "
                        f"|{'#'*bar+'.'*(30-bar)}| {r:+8.1f} | {s:>5}s {sym}\n".encode('utf-8', errors='replace')
                    )
                if ep % 50 == 0:
                    self._save_plot()
        return True
        

    def _save_plot(self):
        rews = np.array(self.ep_rews)
        fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        hc = COL[self.method]["hex"]
        ax.bar(range(len(rews)), rews, color=hc, alpha=0.4, width=1)
        w = min(50, len(rews))
        if len(rews) >= w:
            roll = np.convolve(rews, np.ones(w) / w, "valid")
            ax.plot(range(w - 1, len(rews)), roll, color=hc, lw=2, label=f"MA-{w}")
        ax.axhline(700,  color="#2ea043", ls="--",  lw=1.3, label="solved ~700")
        ax.axhline(900,  color="#f5a623", ls=":",   lw=1.3, label="target ~900")
        ax.set_title(
            f"{self.method} | Ep {len(rews)} | last-50 avg: {np.mean(rews[-50:]):.1f}",
            color="white")
        ax.set_xlabel("Episode", color="#8b949e")
        ax.set_ylabel("Score",   color="#8b949e")
        ax.legend(); plt.tight_layout()
        fname = os.path.join(
            RESULTS_DIR,
            f"training_curve_{self.method.replace('+', '_')}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)


print(" Callbacks ready")


# ───────────────────────────────────────────────────────────────────
# SECTION 11 — Video recorder
# ───────────────────────────────────────────────────────────────────
def _put(img, text, x, y, color=(210, 210, 210), scale=0.42, bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)


def overlay_frame(frame_rgb, step, reward, total_reward, action,
                  ep_num, done, method):
    img     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    col_bgr = COL[method]["bgr"]
    bw      = len(method) * 9 + 14
    cv2.rectangle(img, (5, 5), (5 + bw, 23), col_bgr, -1)
    _put(img, method, 9, 19, (8, 8, 8), 0.38, True)
    _put(img, f"EP {ep_num}", 8, 40, (100, 200, 255), 0.46, True)
    _put(img, f"step {step:05d}", 68, 40, (130, 145, 170), 0.34)
    rc = (50, 210, 100) if reward >= 0 else (80, 80, 220)
    _put(img, f"r:{reward:+.2f}", 5, 54, rc, 0.36)
    _put(img, f"total:{total_reward:+.0f}", 5, 68, rc, 0.36)
    if done:
        lbl = f"  DONE  {total_reward:+.0f}  "
        oc  = (28, 180, 55) if total_reward > 600 else (38, 38, 200)
        H, W = img.shape[:2]
        cv2.rectangle(img, (W // 2 - 110, H // 2 - 20),
                      (W // 2 + 110, H // 2 + 20), oc, -1)
        _put(img, lbl, W // 2 - 100, H // 2 + 7, (255, 255, 255), 0.60, True)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def record_video(model, method, n_eps=VIDEO_EPS):
    
    video_path = os.path.join(RESULTS_DIR, f"video_{method}.mp4")
    env    = make_base_env(render_mode="rgb_array")
    frames = []
    log    = []
    print(f"\n{'─'*52}")
    print(f"  Recording {method} — {n_eps} episodes")
    print(f"{'─'*52}")
    for ep in range(1, n_eps + 1):
        obs, _ = env.reset()
        done = False; tot = 0.0; step = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            act    = np.array(act).flatten()
            obs_n, rew, term, trunc, _ = env.step(act)
            done = term or trunc; tot += rew; step += 1
            raw = env.render()
            if raw is not None:
                frames.append(overlay_frame(raw, step, rew, tot, act, ep, done, method))
            obs = obs_n
        log.append((ep, tot, step))
        print(f"  {ep:>3}  {tot:>+10.1f}  {step:>6}")
    env.close()
    writer = imageio.get_writer(video_path, fps=30, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for f in frames:
        writer.append_data(f)
    writer.close()
    mr = np.mean([r for _, r, _ in log])
    print(f"    Saved {video_path}  |  mean score: {mr:+.1f}")
    return log


# ───────────────────────────────────────────────────────────────────
# SECTION 12 — Per-method result saver
# ───────────────────────────────────────────────────────────────────
def save_method_results(method, ep_rews, ep_steps, video_log):
    
    safe = method.replace("+", "_")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rewards_path = os.path.join(RESULTS_DIR, f"rewards_{safe}.json")
    with open(rewards_path, "w") as f:
        json.dump([float(r) for r in ep_rews], f)

    steps_path = os.path.join(RESULTS_DIR, f"ep_steps_{safe}.json")
    with open(steps_path, "w") as f:
        json.dump([int(s) for s in ep_steps], f)

    vlog_path = os.path.join(RESULTS_DIR, f"video_log_{safe}.json")
    with open(vlog_path, "w") as f:
        json.dump([[int(ep), float(sc), int(st)] for ep, sc, st in video_log], f)

    print(f"    [{method}] Results saved to {RESULTS_DIR}/")


# ───────────────────────────────────────────────────────────────────
# SECTION 13 — [REVIEW-3,5] Deterministic evaluation, budget accounting, hyper-parameter table
# ───────────────────────────────────────────────────────────────────
# Reviewer: "rewards appear to come from TRAINING episodes; separate deterministic evaluation
# episodes should be reported" and "results at matched environment steps incl. pretraining frames".
# Evaluation uses a SEPARATE clean env (no augmentation), deterministic actions, the SAME
# fixed episode seeds for every method, and never updates the model.
EVAL_EPISODES       = 10
EVAL_SEED_BASE      = 100_000                       # eval episode i -> env.reset(seed=EVAL_SEED_BASE+i)
EVAL_AT_TOTAL_STEPS = [1_000_000, 2_500_000, 4_000_000]   # total steps = pretraining frames + RL steps


def _safe(method):
    return method.replace("+", "_")


def _eval_path(method):
    return os.path.join(RESULTS_DIR, f"eval_{_safe(method)}.json")


def _load_eval(method):
    try:
        with open(_eval_path(method)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def evaluate_deterministic(model, n_eps=EVAL_EPISODES, seed_base=EVAL_SEED_BASE):
    """Returns (list of episode returns, number of env steps used). Does not touch training RNG/envs."""
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    env = make_base_env()                              # clean observations, separate from training envs
    rets, n_steps = [], 0
    try:
        for i in range(n_eps):
            obs, _ = env.reset(seed=seed_base + i)
            done, tot = False, 0.0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, term, trunc, _ = env.step(np.array(act).flatten())
                done = term or trunc
                tot += float(rew)
                n_steps += 1
            rets.append(tot)
    finally:
        env.close()
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
    return rets, n_steps


def _record_eval(model, method, nominal, final=False):
    rets, n_steps = evaluate_deterministic(model)
    pre = PRETRAIN_FRAMES[method]
    rec = dict(method=method, seed=GLOBAL_SEED, nominal_total_steps=int(nominal), final=bool(final),
               pretraining_env_steps=pre, rl_env_steps=int(model.num_timesteps),
               total_env_steps=int(model.num_timesteps) + pre,
               n_episodes=len(rets), episode_returns=rets,
               mean_return=float(np.mean(rets)), std_return=float(np.std(rets)),
               evaluation_env_steps=n_steps)          # evaluation steps are NOT part of the training budget
    hist = [h for h in _load_eval(method) if h["nominal_total_steps"] != rec["nominal_total_steps"]]
    hist.append(rec)
    hist.sort(key=lambda h: h["nominal_total_steps"])
    with open(_eval_path(method), "w") as f:
        json.dump(hist, f, indent=1)
    print(f"  [EVAL {method} seed={GLOBAL_SEED}] total_steps={rec['total_env_steps']:,}  "
          f"mean={rec['mean_return']:.1f} ± {rec['std_return']:.1f}  ({len(rets)} deterministic episodes)", flush=True)
    return rec


class EvalCB(BaseCallback):
    """Deterministic evaluation each time the TOTAL step count (pretrain + RL) passes a target."""

    def __init__(self, method, targets=EVAL_AT_TOTAL_STEPS):
        super().__init__()
        self.method  = method
        self.pre     = PRETRAIN_FRAMES[method]
        done         = {h["nominal_total_steps"] for h in _load_eval(method)}   # resume-safe
        self.pending = [t for t in targets if t not in done]

    def _on_step(self):
        if self.pending and self.num_timesteps + self.pre >= self.pending[0]:
            _record_eval(self.model, self.method, self.pending.pop(0))
        return True


def final_evaluate(model, method):
    """Call once after training: final deterministic evaluation + budget accounting file."""
    rec = _record_eval(model, method, COMMON_STEPS, final=True)
    info = dict(method=method, seed=GLOBAL_SEED, pretraining_env_steps=PRETRAIN_FRAMES[method],
                rl_env_steps=int(model.num_timesteps),
                total_training_env_steps=PRETRAIN_FRAMES[method] + int(model.num_timesteps),
                evaluation_env_steps=rec["evaluation_env_steps"], completed=True)
    with open(os.path.join(RESULTS_DIR, f"run_info_{_safe(method)}.json"), "w") as f:
        json.dump(info, f, indent=1)
    return rec


def save_hyperparameters():
    """[REVIEW-5] every setting the reviewers said was missing, written next to the results."""
    path = os.path.join(RESULTS_DIR, "hyperparameters.json")
    if os.path.exists(path):
        return
    hp = dict(
        seed=GLOBAL_SEED, env=ENV_ID, obs="64x64 RGB, 4 stacked frames (12 channels)", n_envs=N_ENVS,
        common_total_steps=COMMON_STEPS, rl_steps=N_STEPS, pretraining_frames=PRETRAIN_FRAMES,
        warmup_steps_encoder_frozen=N_WARMUP_STEPS, finetune_lr_after_unfreeze=FINETUNE_LR,
        ppo_cnn_policy_for_PPO_and_AUG_RL=dict(PPO_KW_CNN),
        ppo_mlp_policy_for_AUTO_and_CURL_methods=dict(PPO_KW_MLP),
        rollout_length_per_update={"PPO/AUG+RL": PPO_KW_CNN["n_steps"] * N_ENVS,
                                   "AUTO/CURL methods": PPO_KW_MLP["n_steps"] * N_ENVS},
        encoder=f"Conv(12,32,8,s4)-Conv(32,64,4,s2)-Conv(64,128,3)-Conv(128,128,3)-FC512-LayerNorm-FC{LATENT_DIM}",
        ssl_policy_features=f"frozen/fine-tuned pretrained encoder ({LATENT_DIM}) + live CNN branch (128); MLP [512,256]",
        pretraining_data="frames collected with a UNIFORM-RANDOM policy on the clean env",
        pretraining_epoch_definition="one 'epoch' = ONE mini-batch gradient step "
                                     "(AE batch 256, CURL batch %d)" % CURL_BATCH,
        pretraining_epochs=dict(AE=AE_EPOCHS, CURL=CURL_EPOCHS),
        ae_noise_level=0.05, curl_temperature=CURL_TEMP, curl_tau=CURL_TAU, curl_projection_dim=CURL_PROJ_DIM,
        augmentation=dict(
            AUG_factor="random shift +-%d px (aug_heavy) applied to RL observations ONLY" % DRQ_PAD,
            PPO="no augmentation", AUG_RL="AUG factor in RL",
            AUTO_RL="denoising AE pretraining, clean RL", AUTO_RL_AUG="same as AUTO+RL + AUG factor in RL",
            CURL_RL="CURL pretraining (its own positive-pair views: shift + brightness 0.025), clean RL",
            AUG_CURL_RL="same CURL pretraining as CURL+RL + AUG factor in RL"),
        evaluation=dict(episodes=EVAL_EPISODES, deterministic=True, seed_base=EVAL_SEED_BASE,
                        at_total_steps=EVAL_AT_TOTAL_STEPS + [COMMON_STEPS]))
    with open(path, "w") as f:
        json.dump(hp, f, indent=1, default=str)


save_hyperparameters()


# ───────────────────────────────────────────────────────────────────
# SECTION 14 — [RESUME] crash-safe training: continue from where a run stopped
# ───────────────────────────────────────────────────────────────────
# What is saved every RESUME_EVERY RL steps (at a rollout boundary, so the state is consistent):
#   resume_<M>.zip         model + optimizer state (the pretrained encoder lives inside the model)
#   resume_state_<M>.json  episode rewards / episode lengths logged so far
# When a script is started again it loads these and continues (pretraining is NOT repeated).
# When a method has fully finished, done_<M>.json is written and the script is skipped next time.
# Files are written to a temp name first and then renamed, so a crash while saving
# can never leave a half-written checkpoint.
# Note: random-number-generator state and the episode that was running at the moment of the
# crash are not restored, so a resumed run is statistically equivalent but not bit-identical.
RESUME_EVERY = 250_000          # RL steps between auto-saves (about 5 min of training on a fast GPU)


def _rp(method, stem, ext=""):
    return os.path.join(RESULTS_DIR, f"{stem}_{_safe(method)}{ext}")


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def is_complete(method):
    return os.path.exists(_rp(method, "done", ".json"))


def skip_if_complete(method):
    """Call at the top of every train_*.py: exit immediately if this seed/method already finished."""
    if is_complete(method):
        print(f"\n[{method} seed={GLOBAL_SEED}] already finished ({_rp(method, 'done', '.json')}) - skipping.")
        sys.exit(0)


def mark_complete(method):
    """Call as the LAST step of every train_*.py."""
    _atomic_json(dict(method=method, seed=GLOBAL_SEED,
                      finished=datetime.datetime.now().isoformat(timespec="seconds")),
                 _rp(method, "done", ".json"))
    for p in (_rp(method, "resume", ".zip"), _rp(method, "resume_state", ".json")):
        try:
            os.remove(p)            # not needed any more (final model is ckpt_<M>.zip)
        except OSError:
            pass


class AutoSaveCB(BaseCallback):
    """Saves model + logs every `every` RL steps, always at the start of a rollout."""

    def __init__(self, method, live_cb, every=RESUME_EVERY):
        super().__init__()
        self.method, self.live, self.every = method, live_cb, every
        self._last = 0

    def _on_training_start(self):
        self._last = self.model.num_timesteps

    def _on_rollout_start(self):
        if self.model.num_timesteps - self._last >= self.every:
            self.save()

    def _on_step(self):
        return True

    def save(self, model=None):
        model = model if model is not None else self.model
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tmp = _rp(self.method, "resume_tmp")
        model.save(tmp)                                               # writes resume_tmp_<M>.zip
        os.replace(tmp + ".zip", _rp(self.method, "resume", ".zip"))  # atomic swap
        _atomic_json(dict(num_timesteps=int(model.num_timesteps),
                          rewards=[float(r) for r in self.live.ep_rews],
                          steps=[int(s) for s in self.live.ep_steps]),
                     _rp(self.method, "resume_state", ".json"))
        self._last = model.num_timesteps
        print(f"  [AutoSave {self.method} seed={GLOBAL_SEED}] saved at {model.num_timesteps:,} RL steps", flush=True)


def _load_resume(method, make_env):
    """Returns (model, rewards, steps) or None when there is nothing (usable) to resume."""
    zpath = _rp(method, "resume", ".zip")
    if not os.path.exists(zpath):
        return None
    try:
        model = PPO.load(zpath, env=make_env())
        rews, stps = [], []
        try:
            with open(_rp(method, "resume_state", ".json")) as f:
                st = json.load(f)
            rews, stps = st.get("rewards", []), st.get("steps", [])
        except (OSError, ValueError):
            print("  (episode log missing - continuing with an empty log)")
        return model, rews, stps
    except Exception as e:
        print(f"  Could not load {zpath}: {e}\n  -> starting this run from the beginning")
        return None


def train_resumable(method, make_env, build_model, extra_callbacks=()):
    """
    Train `method` for N_STEPS[method] RL steps; resume automatically if a previous run was cut off.
      make_env()          -> new vec env
      build_model(make_env) -> NEW PPO model (do pretraining here; only called when NOT resuming)
      extra_callbacks     -> e.g. [UnfreezeEncoderCB(...)]
    Returns (model, live_cb).
    """
    total = N_STEPS[method]
    live = LiveCB(method)
    res = _load_resume(method, make_env)
    resumed = res is not None
    if resumed:
        model, live.ep_rews, live.ep_steps = res
        live.ep_rews, live.ep_steps = list(live.ep_rews), list(live.ep_steps)
        print(f"\n  >>> RESUMING {method} seed={GLOBAL_SEED} from {model.num_timesteps:,} / {total:,} RL steps"
              f"  ({len(live.ep_rews)} episodes already logged)\n", flush=True)
    else:
        model = build_model(make_env)

    auto = AutoSaveCB(method, live)
    cbs = [live, *extra_callbacks, EvalCB(method), auto]
    remaining = total - model.num_timesteps
    if remaining > 0:
        try:
            model.learn(remaining, callback=cbs, reset_num_timesteps=not resumed)
        except BaseException:
            print(f"\n  !! {method} seed={GLOBAL_SEED} stopped at {model.num_timesteps:,} RL steps. "
                  f"Run the same command again - it continues from the last auto-save.", flush=True)
            raise
    else:
        print(f"  {method}: training already complete ({model.num_timesteps:,} steps) - doing final steps only")
    auto.save(model)     # final state, so a crash during evaluation/video does not re-train anything
    return model, live


print("  dmc_common.py fully loaded (Cheetah-Run, review + resume version)")
