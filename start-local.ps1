[CmdletBinding()]
param(
    [string]$BrokerUrl = "http://127.0.0.1:9000/api/v1/db",
    [int]$Port = 8765,
    [string]$QuantSecureEnv = "G:\projects\quant-platform\project\quant-platform\local-state\security\server.env"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Local DeepStock runtime is missing. Run setup-local.ps1 first."
}

$env:BNHMM_WEB_HOST = "127.0.0.1"
$env:BNHMM_WEB_HEALTH_HOST = "127.0.0.1"
$env:BNHMM_WEB_PORT = [string]$Port
$env:BNHMM_DEFAULT_DEVICE = "cuda:0"
$env:BNHMM_MARKET_SOURCE = $BrokerUrl
$env:BNHMM_WEB_JOB_ROOT = Join-Path $root "local-state\runs"
$env:BNHMM_SECTOR_DATASET_ROOT = Join-Path $root "local-state\sector-datasets"
$env:BNHMM_VISUAL_DATASET_ROOT = Join-Path $root "local-state\visual-datasets"
$env:BNHMM_BOUNDARY_CACHE_ROOT = Join-Path $root "local-state\boundary-cache"
$env:BNHMM_PROJECT_ROOT = $root
$env:BNHMM_QUANT_SERVER = "https://quant.hapgetaquant.com"

if (Test-Path -LiteralPath $QuantSecureEnv -PathType Leaf) {
    $brokerTokenLine = Get-Content -LiteralPath $QuantSecureEnv | Where-Object { $_ -match '^\s*QUANT_DB_BROKER_TOKEN\s*=' } | Select-Object -First 1
    if ($brokerTokenLine) {
        $brokerToken = ($brokerTokenLine -split '=', 2)[1].Trim()
        if (($brokerToken.StartsWith('"') -and $brokerToken.EndsWith('"')) -or ($brokerToken.StartsWith("'") -and $brokerToken.EndsWith("'"))) {
            $brokerToken = $brokerToken.Substring(1, $brokerToken.Length - 2)
        }
        $env:BNHMM_MARKET_BROKER_TOKEN = $brokerToken
    }
}

Push-Location $root
try {
    Write-Host "DeepStock Lab: http://127.0.0.1:$Port"
    Write-Host "Market source: $BrokerUrl | training device: cuda:0 | artifacts: $env:BNHMM_WEB_JOB_ROOT"
    & $python -m bnhmm.web.app
    if ($LASTEXITCODE -ne 0) { throw "DeepStock exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
