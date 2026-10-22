[CmdletBinding()]
param(
    [string]$Python312 = "G:\projects\quant-platform\runtime\python\python.exe",
    [string]$Proxy = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$wheel = Join-Path $root "local-state\downloads\torch-2.11.0+cu128-cp312-cp312-win_amd64.whl"
$torchSha256 = "7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c"

if (-not (Test-Path -LiteralPath $Python312 -PathType Leaf)) {
    throw "Python 3.12 was not found at $Python312. Pass -Python312 with a Python 3.12 executable."
}
$version = & $Python312 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($version -ne "3.12") { throw "DeepStock requires Python 3.12; found $version." }

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $Python312 -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the DeepStock virtual environment." }
}

$proxyArgs = @()
if ($Proxy) { $proxyArgs = @("--proxy", $Proxy) }

& $venvPython -m pip install @proxyArgs --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
if (Test-Path -LiteralPath $wheel -PathType Leaf) {
    $actualSha256 = (Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $torchSha256) { throw "Downloaded CUDA PyTorch wheel checksum does not match the official digest." }
    & $venvPython -m pip install $wheel
} else {
    & $venvPython -m pip install @proxyArgs --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7"
}
if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed." }
& $venvPython -m pip install @proxyArgs -e "$root[training,web,visualization]"
if ($LASTEXITCODE -ne 0) { throw "DeepStock training dependencies failed to install." }

& $venvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "PyTorch cannot use the local CUDA GPU." }
