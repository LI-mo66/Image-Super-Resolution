param(
    [ValidateSet('scratch', 'finetune')]
    [string]$Mode = 'scratch',
    [int]$MaxTrainBatches = 20,
    [int]$Epochs = 1,
    [int]$ValidationEnd = 810,
    [string]$Python = 'E:\anaconda\envs\dl\python.exe',
    [string]$DataRoot = 'E:\fuxian_LFMN_jianghe\datasets',
    [string]$SaveName = ''
)

$ErrorActionPreference = 'Stop'
if ($ValidationEnd -lt 801 -or $ValidationEnd -gt 900) {
    throw 'ValidationEnd must be between 801 and 900.'
}
if ($MaxTrainBatches -lt 0) { throw 'MaxTrainBatches cannot be negative.' }
if ($Epochs -lt 1) { throw 'Epochs must be at least 1.' }

$projectRoot = Split-Path -Parent $PSScriptRoot
$lfmnRoot = Join-Path $projectRoot 'LFMN'
$checkpoint = Join-Path $lfmnRoot 'model\scale4_model_939.pt'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable not found: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot 'DIV2K\DIV2K_train_HR\0001.png'))) { throw "DIV2K training data not found under: $DataRoot" }
if ($Mode -eq 'finetune' -and -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Checkpoint not found: $checkpoint" }

if ([string]::IsNullOrWhiteSpace($SaveName)) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $limitTag = if ($MaxTrainBatches -eq 0) { 'full' } else { "${MaxTrainBatches}b" }
    $SaveName = "sanity/x4_${Mode}_${limitTag}_$stamp"
}

$arguments = @(
    'main.py',
    '--dir_data', $DataRoot,
    '--model', 'LFMN',
    '--data_train', 'DIV2K',
    '--data_test', 'DIV2K',
    '--data_range', "1-800/801-$ValidationEnd",
    '--scale', '4',
    '--patch_size', '256',
    '--batch_size', '4',
    '--n_threads', '4',
    '--ext', 'img',
    '--epochs', $Epochs,
    '--test_every', '1000',
    '--lr', '2e-4',
    '--decay', '200-400-600-800',
    '--gamma', '0.5',
    '--loss', '1*L1',
    '--max_train_batches', $MaxTrainBatches,
    '--print_every', '10',
    '--save', $SaveName
)
if ($Mode -eq 'finetune') { $arguments += @('--pre_train', $checkpoint) }

Write-Host "Mode: $Mode; max batches/epoch: $MaxTrainBatches; validation: 0801-$ValidationEnd"
Write-Host "Output: experiment/all_runs/$SaveName"
Push-Location $lfmnRoot
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Training run failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
