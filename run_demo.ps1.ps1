@'
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "       CAMPUSMAIL DEMONSTRATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path ".\outputs\auto_categorized_emails.csv")) {
    Write-Host "Categorized data not found. Running classifier..."
    python .\src\auto_categorize.py
    if ($LASTEXITCODE -ne 0) { throw "Classifier failed." }
}

python -u .\demo.py
if ($LASTEXITCODE -ne 0) { throw "Demo failed." }
'@ | Set-Content -Path .\run_demo.ps1 -Encoding utf8