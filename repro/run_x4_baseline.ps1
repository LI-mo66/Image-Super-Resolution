param(
    [string]$Python = 'E:\anaconda\envs\dl\python.exe',
    [string]$DataRoot = 'E:\fuxian_LFMN_jianghe\datasets',
    [string]$DataTest = 'Set5',
    [switch]$SelfEnsemble
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$lfmnRoot = Join-Path $projectRoot 'LFMN'
$checkpoint = Join-Path $lfmnRoot 'model\scale4_model_939.pt'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable not found: $Python" }
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Checkpoint not found: $checkpoint" }

$runName = if ($SelfEnsemble) { 'baseline/x4_x8' } else { 'baseline/x4_single' }
$arguments = @('main.py', '--dir_data', $DataRoot, '--model', 'LFMN', '--data_test', $DataTest, '--scale', '4', '--pre_train', $checkpoint, '--test_only', '--save', $runName)
if ($SelfEnsemble) { $arguments += '--self_ensemble' }

Push-Location $lfmnRoot
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Baseline evaluation failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
