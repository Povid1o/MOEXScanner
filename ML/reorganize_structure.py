"""
🔧 Скрипт для реорганизации структуры ML/

Перемещает файлы в логичные папки и обновляет пути в скриптах.
"""

import shutil
from pathlib import Path
import re

ML_ROOT = Path(__file__).parent

# Создаём новые директории
NEW_DIRS = {
    'scripts': ML_ROOT / 'scripts',
    'notebooks': ML_ROOT / 'notebooks',
    'docs': ML_ROOT / 'docs',
    'tools': ML_ROOT / 'tools',
}

# Файлы для перемещения
FILES_TO_MOVE = {
    'scripts': [
        'run_full_pipeline.py',
        'run_experiment.py',
        'run_experiment.bat',
        'run_experiment.ps1',
        'validate_model.py',
        'compare_models.py',
        'analyze_feature_correlation.py',
        'example_usage.py',
        'test_setup.py',
    ],
    'notebooks': [
        '01_data_loading.ipynb',
        'plots.ipynb',
    ],
    'docs': [
        'README.md',
        'GUIDE.md',
        'QUICKSTART.md',
        'EXPERIMENTS_GUIDE.md',
        'PROJECT_STRUCTURE.md',
        'ARCHITECTURE_AUDIT.md',
        'DEVELOPMENT_ROADMAP.md',
        'NOTEBOOKS_VS_PRODUCTION_AUDIT.md',
    ],
    'tools': [
        'start_jupyter.bat',
        'start_jupyter.ps1',
    ],
}

# config.py перемещаем в config/
CONFIG_FILE = ML_ROOT / 'config.py'
CONFIG_DIR = ML_ROOT / 'config'


def create_directories():
    """Создаёт новые директории."""
    print("📁 Создание директорий...")
    for name, path in NEW_DIRS.items():
        path.mkdir(exist_ok=True)
        print(f"   ✅ {name}/")
    print()


def move_files():
    """Перемещает файлы в новые директории."""
    print("📦 Перемещение файлов...")
    
    moved_count = 0
    skipped_count = 0
    
    for target_dir, files in FILES_TO_MOVE.items():
        dest = NEW_DIRS[target_dir]
        for filename in files:
            src = ML_ROOT / filename
            if src.exists():
                dest_file = dest / filename
                if dest_file.exists():
                    print(f"   ⚠️ {filename} уже существует в {target_dir}/, пропускаем")
                    skipped_count += 1
                else:
                    shutil.move(str(src), str(dest_file))
                    print(f"   ✅ {filename} → {target_dir}/")
                    moved_count += 1
            else:
                print(f"   ⚠️ {filename} не найден, пропускаем")
                skipped_count += 1
    
    # Перемещаем config.py в config/
    if CONFIG_FILE.exists():
        dest_config = CONFIG_DIR / 'config.py'
        if not dest_config.exists():
            shutil.move(str(CONFIG_FILE), str(dest_config))
            print(f"   ✅ config.py → config/")
            moved_count += 1
        else:
            print(f"   ⚠️ config.py уже существует в config/, пропускаем")
            skipped_count += 1
    
    print(f"\n📊 Перемещено: {moved_count}, пропущено: {skipped_count}\n")
    return moved_count


def update_paths_in_scripts():
    """Обновляет пути в скриптах после перемещения."""
    print("🔧 Обновление путей в скриптах...")
    
    scripts_dir = NEW_DIRS['scripts']
    
    # Файлы, которые нужно обновить
    files_to_update = [
        'run_full_pipeline.py',
        'run_experiment.py',
        'validate_model.py',
        'compare_models.py',
    ]
    
    updated_count = 0
    
    for filename in files_to_update:
        filepath = scripts_dir / filename
        if not filepath.exists():
            # Проверяем в корне (если ещё не перемещён)
            filepath = ML_ROOT / filename
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Обновляем пути к ML_ROOT
            # Было: ML_ROOT = Path(__file__).parent
            # Стало: ML_ROOT = Path(__file__).parent.parent
            
            # Для скриптов в scripts/ нужно подняться на уровень выше
            if 'scripts' in str(filepath):
                # Заменяем Path(__file__).parent на Path(__file__).parent.parent
                content = re.sub(
                    r'ML_ROOT = Path\(__file__\)\.parent',
                    'ML_ROOT = Path(__file__).parent.parent',
                    content
                )
                
                # Обновляем sys.path.insert для config/
                content = re.sub(
                    r"sys\.path\.insert\(0, str\(ML_ROOT / 'config'\)\)",
                    "sys.path.insert(0, str(ML_ROOT / 'config'))",
                    content
                )
                
                # Обновляем sys.path.insert для 03_models/
                content = re.sub(
                    r"sys\.path\.insert\(0, str\(ML_ROOT / '03_models'\)\)",
                    "sys.path.insert(0, str(ML_ROOT / '03_models'))",
                    content
                )
            
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"   ✅ Обновлён: {filename}")
                updated_count += 1
            else:
                print(f"   ⏭️ Без изменений: {filename}")
    
    print(f"\n📊 Обновлено файлов: {updated_count}\n")


def create_readme_in_scripts():
    """Создаёт README в scripts/ с описанием скриптов."""
    readme_content = """# 📜 Scripts Directory

Исполняемые скрипты для работы с моделью.

## 🚀 Основные скрипты

### run_full_pipeline.py
Полный pipeline: Feature Engineering → Training → Inference
```bash
python scripts/run_full_pipeline.py [--skip-features] [--skip-training]
```

### run_experiment.py
Запуск эксперимента с разными конфигурациями
```bash
python scripts/run_experiment.py --preset MORE_TRAIN --skip-features
```

### validate_model.py
Валидация модели на тестовых данных
```bash
python scripts/validate_model.py
```

### compare_models.py
Сравнение результатов разных моделей
```bash
python scripts/compare_models.py
```

### analyze_feature_correlation.py
Анализ корреляций между признаками
```bash
python scripts/analyze_feature_correlation.py
```

## 🛠️ Вспомогательные

- **example_usage.py** - Примеры использования API
- **test_setup.py** - Проверка окружения

## 📝 Батники

- **run_experiment.bat** - Windows batch для экспериментов
- **run_experiment.ps1** - PowerShell скрипт для экспериментов
"""
    
    readme_path = NEW_DIRS['scripts'] / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("   ✅ Создан README.md в scripts/")


def main():
    """Главная функция."""
    print("=" * 70)
    print("🔧 РЕОРГАНИЗАЦИЯ СТРУКТУРЫ ML/")
    print("=" * 70)
    print()
    
    # 1. Создаём директории
    create_directories()
    
    # 2. Перемещаем файлы
    moved = move_files()
    
    if moved == 0:
        print("⚠️ Все файлы уже на месте. Пропускаем обновление путей.")
        return
    
    # 3. Обновляем пути в скриптах
    update_paths_in_scripts()
    
    # 4. Создаём README
    create_readme_in_scripts()
    
    print("=" * 70)
    print("✅ РЕОРГАНИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print()
    print("📁 Новая структура:")
    print("   scripts/     - Все исполняемые скрипты")
    print("   notebooks/   - Jupyter ноутбуки")
    print("   docs/        - Документация")
    print("   tools/       - Вспомогательные скрипты")
    print()
    print("💡 Теперь запускайте скрипты из scripts/:")
    print("   python scripts/run_experiment.py --preset MORE_TRAIN")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

