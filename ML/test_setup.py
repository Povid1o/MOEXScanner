"""
Скрипт для проверки правильности настройки проекта
Запустите: .\\venv\\Scripts\\python.exe test_setup.py
"""
import sys
import os
from pathlib import Path

# Установка кодировки для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

print("=" * 50)
print("ПРОВЕРКА НАСТРОЙКИ ПРОЕКТА")
print("=" * 50)

# 1. Проверка Python пути
python_exe = sys.executable
print(f"\n1. Python интерпретатор: {python_exe}")
if 'venv' in python_exe or 'ML' in python_exe:
    print("   [OK] Используется venv")
else:
    print("   [WARNING] Используется системный Python!")
    print("   Решение: Выберите интерпретатор ML\\venv\\Scripts\\python.exe в Cursor")

# 2. Проверка библиотек
print("\n2. Проверка библиотек:")
libraries = {
    'pandas': 'pd',
    'numpy': 'np',
    'seaborn': 'sns',
    'matplotlib': 'plt',
    'sklearn': None,
    'scipy': None
}

all_ok = True
for lib_name, alias in libraries.items():
    try:
        if alias:
            module = __import__(lib_name, fromlist=[alias])
            version = getattr(module, '__version__', 'unknown')
            print(f"   [OK] {lib_name} {version}")
        else:
            module = __import__(lib_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   [OK] {lib_name} {version}")
    except ImportError as e:
        print(f"   [ERROR] {lib_name} - НЕ НАЙДЕНА!")
        print(f"      Ошибка: {e}")
        all_ok = False

# 3. Проверка путей проекта
print("\n3. Проверка структуры проекта:")
try:
    from config import BASE_DIR, DATA_DIR, MOEX_DATA_DIR
    print(f"   [OK] BASE_DIR: {BASE_DIR}")
    print(f"   [OK] DATA_DIR: {DATA_DIR}")
    print(f"   [OK] MOEX_DATA_DIR: {MOEX_DATA_DIR}")
    
    # Проверка существования директорий
    if DATA_DIR.exists():
        print(f"   [OK] Папка данных существует")
    else:
        print(f"   [WARNING] Папка данных не найдена: {DATA_DIR}")
        
except ImportError as e:
    print(f"   [ERROR] Ошибка импорта config: {e}")
    all_ok = False

# 4. Проверка модулей features
print("\n4. Проверка модулей features:")
try:
    # Проверяем, что модуль существует
    import features.Loaders.load_prices
    print("   [OK] load_prices модуль найден")
except (ImportError, EOFError) as e:
    print(f"   [WARNING] load_prices: {type(e).__name__} - {e}")

try:
    from features.Loaders.price_cleaner import PriceCleaner
    print("   [OK] PriceCleaner импортирован")
except (ImportError, EOFError) as e:
    print(f"   [WARNING] PriceCleaner: {type(e).__name__} - {e}")

print("\n" + "=" * 50)
if all_ok:
    print("[SUCCESS] ВСЁ НАСТРОЕНО ПРАВИЛЬНО!")
    print("Можете начинать работу с проектом.")
else:
    print("[WARNING] ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
    print("См. руководство ML/GUIDE.md для решения проблем.")
print("=" * 50)

