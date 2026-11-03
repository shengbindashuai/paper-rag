param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [Parameter(Mandatory=$true)][string]$WorkspaceRoot,
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$entryPath = Join-Path $PSScriptRoot 'background.py'
Set-Location -LiteralPath $WorkspaceRoot
$serviceArgs = '-u "{0}" --root "{1}" --port {2}' -f $entryPath, $WorkspaceRoot, $Port
$serviceProcess = Start-Process -FilePath $PythonPath -ArgumentList $serviceArgs -WorkingDirectory $WorkspaceRoot -WindowStyle Hidden -PassThru -Wait
exit $serviceProcess.ExitCode
