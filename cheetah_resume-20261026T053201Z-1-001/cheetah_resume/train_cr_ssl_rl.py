"""
train_cr_ssl_rl.py — Method 3: AUTO+RL (Denoising Autoencoder + RL)  (Cheetah-Run)
==============================================================
Run independently:  python train_cr_ssl_rl.py [--seed 42]
Results saved to:   cheetah_vscoderesult/seed_<SEED>/
[RESUME] If the run is interrupted, start the same command again: it continues from the last auto-save.
"""
from dmc_common import *

METHOD = "AUTO+RL"
skip_if_complete(METHOD)         # [RESUME] already finished for this seed -> exit
set_global_seed()
print(f"\n{'='*58}\n  Training {METHOD}  ({N_STEPS[METHOD]:,} steps)\n{'='*58}")

# Phase 0 (AE pretraining) + Phase 1-2 (PPO, encoder frozen then unfrozen)
def build(make_env):                                           # [RESUME] only called when NOT resuming
    encoder = pretrain_ae(use_aug=False, noise_level=0.05, epochs=AE_EPOCHS)
    pk = make_ssl_policy_kwargs(encoder)
    return PPO("MlpPolicy", make_env(), policy_kwargs=pk, **PPO_KW_MLP)

unfreeze_cb = UnfreezeEncoderCB(
    unfreeze_at=N_WARMUP_STEPS[METHOD],
    finetune_lr=FINETUNE_LR[METHOD])

model, cb = train_resumable(                                   # [RESUME]
    METHOD,
    make_env    = lambda: make_vec_env(make_base_env, n_envs=N_ENVS, seed=GLOBAL_SEED),
    build_model = build,
    extra_callbacks = [unfreeze_cb])
model.save(os.path.join(RESULTS_DIR, f"ckpt_{METHOD}"))

final_evaluate(model, METHOD)    # [REVIEW-3] final deterministic evaluation + step accounting
rewards = cb.ep_rews.copy()
print(f"✅  {METHOD}  last-50 avg: {np.mean(rewards[-50:]):.1f}")

video_log = record_video(model, METHOD)
save_method_results(METHOD, rewards, cb.ep_steps, video_log)
mark_complete(METHOD)            # [RESUME] this seed/method is finished -> skipped on the next run

print(f"\n✅  {METHOD} complete. All results in {RESULTS_DIR}/")
