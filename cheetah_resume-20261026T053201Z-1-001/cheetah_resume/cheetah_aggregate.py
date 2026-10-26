"""
cheetah_aggregate.py — Simplified Results Summary
==================================================
Shows per-method:
  - Best reward
  - Last 10% mean
  - Last 50% mean
  - Convergence reward (rolling-20 ≥ 500)
  - ≥700% rate

Run after all training scripts have finished.

[REVIEW] Added (everything above is unchanged):
  - results are read per seed from  cheetah_vscoderesult/seed_<S>/  (seeds 42, 123, 2024)
  - the original table is printed for every seed that exists
  - ACROSS-SEED table: mean ± std (sample std, ddof=1) of the DETERMINISTIC evaluation, n = number of seeds
  - MATCHED-STEPS table: deterministic evaluation at the same total env steps (pretraining frames included)
  - a seed counts only if its final evaluation file exists (unfinished runs are listed, never averaged in)
"""

import os, json, csv
import numpy as np

# ── Config (must match dmc_common.py) ────────────────────────────────
RESULTS_BASE = "cheetah_vscoderesult"
SEEDS        = [42, 123, 2024]          # [REVIEW] same 3 seeds for every method
SOLVE_THRESH = 700
ALL_METHODS  = ["PPO", "AUG+RL", "AUTO+RL", "AUTO+RL+AUG", "CURL+RL", "AUG+CURL+RL"]


# ── Load per-method data ──────────────────────────────────────────
def load_method(method, seed):
    RESULTS_DIR = os.path.join(RESULTS_BASE, f"seed_{seed}")   # [REVIEW] per-seed folder
    safe = method.replace("+", "_")      # save_method_results() always writes this name (e.g. CURL_RL)

    r_path = os.path.join(RESULTS_DIR, f"rewards_{safe}.json")
    s_path = os.path.join(RESULTS_DIR, f"ep_steps_{safe}.json")

    if not os.path.exists(r_path):
        print(f"  Missing: {r_path}")
        return None, None

    with open(r_path) as f:
        rewards = json.load(f)
    with open(s_path) as f:
        ep_steps = json.load(f)

    return rewards, ep_steps


# ── Stats per method ─────────────────────────────────────────────
def compute_stats(rewards):
    rews = np.array(rewards, dtype=float)
    n    = len(rews)

    best_reward = float(np.max(rews))

    n10  = max(1, int(n * 0.10))
    last10_mean = float(np.mean(rews[-n10:]))

    n50  = max(1, min(50, n))
    last50_mean = float(np.mean(rews[-n50:]))

    high_rate = float(np.sum(rews >= SOLVE_THRESH)) / n * 100

    # Convergence: first episode where rolling-20 mean >= 500
    conv_reward = None
    conv_ep     = None
    if n >= 20:
        rm20 = np.convolve(rews, np.ones(20) / 20, "valid")
        idx  = np.where(rm20 >= 500)[0]
        if len(idx):
            conv_ep     = int(idx[0]) + 20
            conv_reward = float(rews[conv_ep - 1])

    return dict(
        best_reward  = best_reward,
        total_mean   = float(np.mean(rews)),
        last10_mean  = last10_mean,
        last50_mean  = last50_mean,
        high_rate    = high_rate,
        conv_ep      = conv_ep,
        conv_reward  = conv_reward,
        total_eps    = n,
    )


# ── [REVIEW] helpers for evaluation files ────────────────────────
def load_eval(method, seed):
    p = os.path.join(RESULTS_BASE, f"seed_{seed}", f"eval_{method.replace('+', '_')}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def is_complete(method, seed):
    """Finished = final evaluation record exists (written only after training ended)."""
    ev = load_eval(method, seed)
    return bool(ev) and any(e.get("final") for e in ev)


def mean_std(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def fmt(v):
    m, s = mean_std(v)
    return f"{m:8.1f} ± {s:6.1f}" if len(v) > 1 else f"{m:8.1f} ±    n/a"


# ── Per-seed original table (unchanged columns) ──────────────────
def print_table(all_stats, available, title):
    W = 92
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(f"  {'Method':<16} {'Best':>8}  {'Last10%':>8}  {'Last50':>8}  {'TotalMean':>10}  {'≥700%':>7}  {'ConvRew':>8}  {'ConvEp':>7}  {'TotalEps':>9}")
    print(f"  {'-'*90}")
    for m in ALL_METHODS:
        s = all_stats.get(m)
        if s is None:
            print(f"  {m:<16}  (no data)")
            continue
        conv_r  = f"{s['conv_reward']:+.1f}" if s["conv_reward"] is not None else "  N/A"
        conv_ep = f"{s['conv_ep']}"           if s["conv_ep"]     is not None else "  N/A"
        print(f"  {m:<16} {s['best_reward']:>+8.1f}  {s['last10_mean']:>+8.1f}"
              f"  {s['last50_mean']:>+8.1f}  {s['total_mean']:>+10.1f}  {s['high_rate']:>6.1f}%"
              f"  {conv_r:>8}  {conv_ep:>7}  {s['total_eps']:>9}")
    print(f"{'='*W}")


per_seed_stats = {}          # (method, seed) -> training-episode stats
any_data = False
for seed in SEEDS:
    stats, avail = {}, []
    for m in ALL_METHODS:
        rews, steps = load_method(m, seed)
        if rews is not None:
            stats[m] = compute_stats(rews)
            per_seed_stats[(m, seed)] = stats[m]
            avail.append(m)
    if avail:
        any_data = True
        print_table(stats, avail, f"Cheetah-Run | seed {seed} | TRAINING-episode statistics (diagnostic only)")
if not any_data:
    raise RuntimeError("No result files found. Run training scripts first.")

# ── [REVIEW] ACROSS-SEED deterministic evaluation (final) ────────
print(f"\n{'='*92}\n  DETERMINISTIC EVALUATION at the common budget — mean ± std ACROSS SEEDS\n{'='*92}")
print(f"  {'Method':<16} {'n_seeds':>7}  {'seeds':<16} {'final eval return':>22}  {'last-50 TRAIN (diag.)':>24}")
rows = []
for m in ALL_METHODS:
    done = [sd for sd in SEEDS if is_complete(m, sd)]
    unfinished = [sd for sd in SEEDS if sd not in done and (m, sd) in per_seed_stats]
    if not done:
        print(f"  {m:<16} {0:>7}  (no finished run)" + (f"   unfinished seeds: {unfinished}" if unfinished else ""))
        continue
    fin = [[e for e in load_eval(m, sd) if e.get("final")][0]["mean_return"] for sd in done]
    tr  = [per_seed_stats[(m, sd)]["last50_mean"] for sd in done if (m, sd) in per_seed_stats]
    print(f"  {m:<16} {len(done):>7}  {str(done):<16} {fmt(fin):>22}  {fmt(tr) if tr else 'n/a':>24}"
          + (f"   unfinished (not used): {unfinished}" if unfinished else ""))
    mm, ss = mean_std(fin)
    rows.append(dict(method=m, n_seeds=len(done), seeds=" ".join(map(str, done)), final_eval_mean=mm, final_eval_std=ss))
print("  std = sample std over seeds (ddof=1). No significance test is performed; with 3 seeds treat differences with caution.")

# ── [REVIEW] MATCHED-STEPS table (total steps = pretraining frames + RL steps) ──
targets = sorted({e["nominal_total_steps"] for m in ALL_METHODS for sd in SEEDS for e in (load_eval(m, sd) or [])})
if targets:
    print(f"\n{'='*92}\n  DETERMINISTIC EVALUATION at MATCHED total environment steps (pretraining frames included)\n{'='*92}")
    print(f"  {'Method':<16}" + "".join(f"{t/1e6:>24.2f}M" for t in targets))
    for m in ALL_METHODS:
        cells = []
        for t in targets:
            v = []
            for sd in SEEDS:
                ev = load_eval(m, sd) or []
                v += [e["mean_return"] for e in ev if e["nominal_total_steps"] == t]
            cells.append(f"{fmt(v) + f' (n={len(v)})':>25}" if v else f"{'—':>25}")
        print(f"  {m:<16}" + "".join(cells))
        for t in targets:
            pass
    csv_path = os.path.join(RESULTS_BASE, "across_seed_matched_steps.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "nominal_total_steps", "n_seeds", "mean_eval_return", "std_over_seeds"])
        for m in ALL_METHODS:
            for t in targets:
                v = [e["mean_return"] for sd in SEEDS for e in (load_eval(m, sd) or []) if e["nominal_total_steps"] == t]
                if v:
                    mm, ss = mean_std(v)
                    w.writerow([m, t, len(v), mm, ss])
    print(f"\n  saved {csv_path}")
print()
