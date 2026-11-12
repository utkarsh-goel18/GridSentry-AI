$ErrorActionPreference = "Stop"

# Run each experiment from a fresh, strategy-specific preprocessing cache.
# Checkpoints and metrics are kept separate; the production best_model.pt is untouched.
$experiments = @(
    @{ Name = "random"; Split = "random" },
    @{ Name = "distance"; Split = "distance_holdout" },
    @{ Name = "resistance"; Split = "resistance_holdout" },
    @{ Name = "angle"; Split = "angle_holdout" }
)

foreach ($experiment in $experiments) {
    $name = $experiment.Name
    $strategy = $experiment.Split
    $checkpoint = "artifacts/best_model_{0}_audit.pt" -f $name
    $metrics = "artifacts/evaluation_metrics_{0}_audit.json" -f $name
    $config = "artifacts/config_{0}_audit.json" -f $name

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Training experiment: $name ($strategy)"
    Write-Host "============================================================"

    python -m ml.train --model 1d_cnn --epochs 20 --split-strategy $strategy --checkpoint $checkpoint --config-out $config
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $strategy (exit code $LASTEXITCODE)."
    }

    python -m ml.evaluate --checkpoint $checkpoint --output $metrics --split-strategy $strategy
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $strategy (exit code $LASTEXITCODE)."
    }
}

Write-Host ""
Write-Host "All experiments completed. Review artifacts/evaluation_metrics_*_audit.json."
