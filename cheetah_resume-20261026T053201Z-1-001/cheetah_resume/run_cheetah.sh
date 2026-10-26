#!/usr/bin/env bash
# Linux version of run_cheetah.ps1
# [REVIEW-2] Same 3 seeds for EVERY method; two jobs at a time.
# [RESUME]   Safe to stop (Ctrl+C, crash, power cut) and start again:
#              - a finished (seed, method) is skipped      (done_<METHOD>.json exists)
#              - an unfinished one continues from its last auto-save (resume_<METHOD>.zip)
#              - a job that crashes is restarted automatically, up to MAX_ATTEMPTS times
#
# Usage:   chmod +x run_cheetah.sh && ./run_cheetah.sh
# Long run (keeps going after you close the terminal):   nohup ./run_cheetah.sh > run_cheetah.log 2>&1 &
# Another python:   PYTHON=python ./run_cheetah.sh

cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python3}"
SEEDS=(42 123 2024)
MAX_ATTEMPTS=3

PIDS=()
cleanup() {                       # Ctrl+C / kill -> stop the training jobs too
    echo; echo "Stopping jobs... run ./run_cheetah.sh again to continue from the last auto-save."
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

is_done() { [[ -f "cheetah_vscoderesult/seed_$1/done_$2.json" ]]; }

# run_batch SEED LABEL "script.py:NAME" "script.py:NAME"
run_batch() {
    local seed="$1" label="$2"; shift 2
    local jobs=("$@") attempt job script name pending
    echo "=== seed $seed : $label ==="
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
        pending=()
        for job in "${jobs[@]}"; do
            is_done "$seed" "${job#*:}" || pending+=("$job")
        done
        if [[ ${#pending[@]} -eq 0 ]]; then
            (( attempt == 1 )) && echo "    already finished - skipping"
            return 0
        fi
        (( attempt > 1 )) && echo "    restarting unfinished job(s) (attempt $attempt of $MAX_ATTEMPTS) - they continue from the last auto-save"
        PIDS=()
        for job in "${pending[@]}"; do
            "$PYTHON" "${job%%:*}" --seed "$seed" &
            PIDS+=($!)
        done
        for p in "${PIDS[@]}"; do wait "$p"; done     # ignore exit codes: done_*.json decides
        PIDS=()
    done
    local left=()
    for job in "${jobs[@]}"; do is_done "$seed" "${job#*:}" || left+=("${job#*:}"); done
    if [[ ${#left[@]} -gt 0 ]]; then
        echo "WARNING: seed $seed : still unfinished after $MAX_ATTEMPTS attempts: ${left[*]}. Run ./run_cheetah.sh again to continue."
    fi
}

for s in "${SEEDS[@]}"; do
    run_batch "$s" "batch 1 (PPO, AUG+RL)"           "train_cr_ppo.py:PPO"           "train_cr_aug_rl.py:AUG_RL"
    run_batch "$s" "batch 2 (AUTO+RL methods)"       "train_cr_ssl_rl.py:AUTO_RL"    "train_cr_aug_ssl_rl.py:AUTO_RL_AUG"
    run_batch "$s" "batch 3 (CURL+RL methods)"       "train_cr_curl_rl.py:CURL_RL"   "train_cr_aug_curl_rl.py:AUG_CURL_RL"
done

echo "All training done. Aggregating results..."
"$PYTHON" cheetah_aggregate.py
