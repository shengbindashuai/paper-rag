# 在你自己的 PowerShell 窗口运行；输入内容不会回显或写入命令历史。
$ErrorActionPreference = 'Stop'
$protectedKey = Read-Host 'DeepSeek API key' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($protectedKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer).Trim()
    if ([string]::IsNullOrWhiteSpace($plainKey)) { throw 'API key cannot be empty.' }
    [Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', $plainKey, 'User')
    $env:DEEPSEEK_API_KEY = $plainKey
    Write-Output 'Saved DEEPSEEK_API_KEY to your Windows user environment.'
    Write-Output 'Restart the LocalKnowledgeMCP task when no ingestion job is running.'
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $plainKey = $null
    $protectedKey.Dispose()
}
