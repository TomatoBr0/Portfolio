param([int]$WaitForProcessId = 0)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$statusPath = Join-Path $PSScriptRoot 'SETUP_STATUS.md'
function Invoke-CheckedPython {
    param([string[]]$PythonArguments)
    & "$PSScriptRoot\.venv\Scripts\python.exe" @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($PythonArguments -join ' ')" }
}
try {
    '# Setup in progress' | Set-Content -LiteralPath $statusPath
    if ($WaitForProcessId -gt 0) {
        'Waiting for the existing PyTorch download to finish and populate the pip cache.' | Add-Content -LiteralPath $statusPath
        Wait-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    }
    Invoke-CheckedPython -PythonArguments @('-m','pip','install','--upgrade','pip')
    Invoke-CheckedPython -PythonArguments @('-m','pip','install','torch==2.14.1+cu126','--index-url','https://download.pytorch.org/whl/cu126')
    Invoke-CheckedPython -PythonArguments @('-m','pip','install','-r','requirements.txt')
    Invoke-CheckedPython -PythonArguments @('scripts/check_setup.py')
    if ($LASTEXITCODE -ne 0) { throw 'Setup check failed' }
    & "$PSScriptRoot\.venv\Scripts\python.exe" -m pip freeze | Set-Content -LiteralPath "$PSScriptRoot\requirements-lock.txt"
    Invoke-CheckedPython -PythonArguments @('-m','pip','check')
    '# Setup verified' | Set-Content -LiteralPath $statusPath
    "Completed: $(Get-Date -Format o). Offline model generation passed. See setup.log for GPU availability and package details." | Add-Content -LiteralPath $statusPath
} catch {
    '# Setup needs attention' | Set-Content -LiteralPath $statusPath
    $_.Exception.Message | Add-Content -LiteralPath $statusPath
    throw
}
