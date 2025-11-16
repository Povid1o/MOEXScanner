# PowerShell скрипт для запуска Jupyter Lab
# Обход Execution Policy через прямое обращение к Python
Write-Host "Запуск Jupyter Lab для MOEX Scanner ML Workspace..." -ForegroundColor Green
Write-Host ""

Write-Host "Использование Python из виртуального окружения..." -ForegroundColor Yellow
$pythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (Test-Path $pythonPath) {
    Write-Host "Запуск Jupyter Lab..." -ForegroundColor Yellow
    & $pythonPath -m jupyter lab --no-browser --port=8888
} else {
    Write-Host "Ошибка: Python не найден в виртуальном окружении!" -ForegroundColor Red
    Write-Host "Путь: $pythonPath" -ForegroundColor Red
    pause
}
