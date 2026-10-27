param(
    [ValidateSet("smoke", "pilot", "test", "verify", "train")]
    [string]$Task = "smoke",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot "..\Graph-Seq-MADAC\.conda-env\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found. Pass -Python with an environment containing the documented dependencies."
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Push-Location $PSScriptRoot
try {
    switch ($Task) {
        "smoke" {
            & $Python -B train.py --name "smoke_graph_$stamp" --smoke
            if ($LASTEXITCODE -ne 0) { throw "Graph smoke failed" }
            & $Python -B train.py --name "smoke_stats_$stamp" --mode stats --smoke
        }
        "pilot" {
            & $Python -B train.py --name "pilot_graph_$stamp" --smoke --budget-ratio 10 --total-steps 400 --graph-hidden-dim 64
            if ($LASTEXITCODE -ne 0) { throw "Graph pilot failed" }
            & $Python -B train.py --name "pilot_stats_$stamp" --mode stats --smoke --budget-ratio 10 --total-steps 400 --graph-hidden-dim 64
        }
        "test" { & $Python -B -m pytest "--basetemp=runs/test_tmp_$stamp" }
        "verify" { & $Python -B verify_originals.py }
        "train" { & $Python -B train.py --name "graph_$stamp" }
    }
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
