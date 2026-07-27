# Thin wrapper so PowerShell users can double-click / tab-complete.
# All the logic lives in local.py -- one implementation, testable on any platform.
#   .\run-local.ps1 -Demo
#   .\run-local.ps1
python "$PSScriptRoot\local.py" @args
