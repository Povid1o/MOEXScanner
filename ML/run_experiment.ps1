# 🚀 PowerShell скрипт для запуска эксперимента
# Использование: .\run_experiment.ps1 [PRESET_NAME]

param(
    [string]$Preset = "MORE_TRAIN"
)

# Определяем директорию скрипта
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Проверяем наличие виртуального окружения
$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "❌ ОШИБКА: Виртуальное окружение не найдено!" -ForegroundColor Red
    Write-Host "   Проверьте путь: $PythonExe" -ForegroundColor Yellow
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🧪 ЗАПУСК ЭКСПЕРИМЕНТА" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "📌 Пресет: $Preset" -ForegroundColor Green
Write-Host "📅 Время: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "📍 Директория: $ScriptDir" -ForegroundColor Green
Write-Host "🐍 Python: $PythonExe" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Проверяем версию Python
$PythonVersion = & $PythonExe --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ОШИБКА: Python не запускается!" -ForegroundColor Red
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

Write-Host "✅ Python найден: $PythonVersion" -ForegroundColor Green
Write-Host ""

# Запускаем эксперимент
Write-Host "🚀 Запуск эксперимента..." -ForegroundColor Yellow
Write-Host ""

& $PythonExe run_experiment.py --preset $Preset --skip-features

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
if ($ExitCode -eq 0) {
    Write-Host "✅ ЭКСПЕРИМЕНТ ЗАВЕРШЁН" -ForegroundColor Green
} else {
    Write-Host "❌ ЭКСПЕРИМЕНТ ЗАВЕРШИЛСЯ С ОШИБКОЙ" -ForegroundColor Red
}
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Для сравнения результатов запустите:" -ForegroundColor Yellow
Write-Host "   python compare_models.py" -ForegroundColor White
Write-Host ""

Read-Host "Нажмите Enter для выхода"

