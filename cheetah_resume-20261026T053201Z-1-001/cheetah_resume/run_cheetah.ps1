# [REVIEW-2] Same 3 seeds for EVERY method; two jobs at a time (same batches as before).
# [RESUME]   Safe to stop (Ctrl+C, crash, power cut) and start again:
#              - a finished (seed, method) is skipped      (done_<METHOD>.json exists)
#              - an unfinished one continues from its last auto-save (resume_<METHOD>.zip)
#              - a job that crashes is restarted automatically, up to $maxAttempts times
$seeds       = 42, 123, 2024
$maxAttempts = 3

function Test-Done($seed, $name) {
    Test-Path "cheetah_vscoderesult\seed_$seed\done_$name.json"
}

function Run-Batch($seed, $label, $jobs) {
    Write-Host "=== seed $seed : $label ==="
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $pending = @($jobs | Where-Object { -not (Test-Done $seed $_.Name) })
        if ($pending.Count -eq 0) {
            if ($attempt -eq 1) { Write-Host "    already finished - skipping" }
            return
        }
        if ($attempt -gt 1) { Write-Host "    restarting unfinished job(s) (attempt $attempt of $maxAttempts) - they continue from the last auto-save" }
        $procs = @()
        foreach ($j in $pending) {
            $procs += Start-Process python -ArgumentList "$($j.Script) --seed $seed" -NoNewWindow -PassThru
        }
        Wait-Process -Id ($procs | ForEach-Object { $_.Id })
    }
    $left = @($jobs | Where-Object { -not (Test-Done $seed $_.Name) })
    if ($left.Count -gt 0) {
        Write-Warning "seed $seed : still unfinished after $maxAttempts attempts: $(($left | ForEach-Object { $_.Name }) -join ', '). Run run_cheetah.ps1 again to continue."
    }
}

foreach ($s in $seeds) {
    Run-Batch $s "batch 1 (PPO, AUG+RL)" @(
        @{ Script = "train_cr_ppo.py";     Name = "PPO" },
        @{ Script = "train_cr_aug_rl.py";  Name = "AUG_RL" })

    Run-Batch $s "batch 2 (AUTO+RL methods)" @(
        @{ Script = "train_cr_ssl_rl.py";     Name = "AUTO_RL" },
        @{ Script = "train_cr_aug_ssl_rl.py"; Name = "AUTO_RL_AUG" })

    Run-Batch $s "batch 3 (CURL+RL methods)" @(
        @{ Script = "train_cr_curl_rl.py";     Name = "CURL_RL" },
        @{ Script = "train_cr_aug_curl_rl.py"; Name = "AUG_CURL_RL" })
}

Write-Host "All training done. Aggregating results..."
python cheetah_aggregate.py
