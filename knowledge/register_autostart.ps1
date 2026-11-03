param(
    [string]$PythonPath = (Get-Command python -ErrorAction Stop).Source,
    [string]$WorkspaceRoot = (Join-Path $PSScriptRoot '..'),
    [int]$Port = 8765,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$taskName = 'LocalKnowledgeMCP'
$workspaceRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
$pythonAbsolute = (Resolve-Path -LiteralPath $PythonPath).Path
$entryPath = Join-Path $PSScriptRoot 'background.py'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
# 直接跟踪实际服务进程，避免停止 PowerShell 包装器后遗留 Start-Process 子进程。
$pythonWindowless = Join-Path (Split-Path -Parent $pythonAbsolute) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonWindowless)) { throw 'pythonw.exe is required for the hidden scheduled service.' }
$arguments = '-u "{0}" --root "{1}" --port {2}' -f $entryPath,$workspaceRoot,$Port
$action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument $arguments -WorkingDirectory $workspaceRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$trigger.Delay = 'PT20S'
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -RestartCount 20 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Local knowledge MCP for Codex; managed by RAG_code/knowledge/register_autostart.ps1'
$xmlPath = Join-Path $PSScriptRoot 'autostart-task.xml'
$task | Export-ScheduledTask | Set-Content -LiteralPath $xmlPath -Encoding Unicode
if ($Preview) {
    Write-Output "Prepared task XML: $xmlPath"
    exit 0
}
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.Description -notlike '*RAG_code/knowledge/register_autostart.ps1*') {
    throw "An unrelated task already uses the name $taskName; refusing to replace it."
}
if ($existing -and $existing.State -eq 'Running') {
    throw 'The existing service is running. Stop it explicitly before updating its registration.'
}
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output "Service endpoint: http://127.0.0.1:$Port/mcp"
