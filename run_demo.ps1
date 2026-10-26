$ErrorActionPreference = "Stop"

Write-Host "`n=== CAMPUSMAIL: CATEGORIZATION ===" -ForegroundColor Cyan
python .\src\auto_categorize.py
if ($LASTEXITCODE -ne 0) { throw "Categorization failed." }

Write-Host "`n=== CAMPUSMAIL: VALIDATION ===" -ForegroundColor Cyan
python .\check_accuracy.py
if ($LASTEXITCODE -ne 0) { throw "Validation failed." }

Write-Host "`n=== CAMPUSMAIL: INTERACTIVE DEMO ===" -ForegroundColor Cyan
python .\demo.py
if ($LASTEXITCODE -ne 0) { throw "Demo failed." }
