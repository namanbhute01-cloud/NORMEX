# Run the NORMEX dashboard at logon / startup without a console window.
#
#   powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
#   powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1 -AtStartup
#
# The task restarts the server up to 3 times and runs even if the machine was
# asleep at the scheduled time.  Uninstall with:
#   Unregister-ScheduledTask -TaskName "NORMEX dashboard" -Confirm:$false

param(
    [string]$TaskName = "NORMEX dashboard",
    [switch]$AtStartup
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $launcher)) {
    throw "Create the virtual environment first (see README_QUICKSTART.md): $launcher is missing"
}

$action = New-ScheduledTaskAction `
    -Execute $launcher `
    -Argument "launch.py dashboard --no-open" `
    -WorkingDirectory $root

$trigger = if ($AtStartup) {
    New-ScheduledTaskTrigger -AtStartup
} else {
    New-ScheduledTaskTrigger -AtLogOn
}

$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "NORMEX CPU inspection dashboard on http://127.0.0.1:8765" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Installed and started '$TaskName'. Dashboard: http://127.0.0.1:8765"
