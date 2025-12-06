"""
🚀 Скрипт для запуска эксперимента с новой конфигурацией

Использование:
    python run_experiment.py [--preset PRESET_NAME] [--skip-features]

Примеры:
    python run_experiment.py --preset MORE_TRAIN
    python run_experiment.py --preset REGULARIZED --skip-features
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Проверка зависимостей перед импортом
try:
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
except ImportError as e:
    print("❌ ОШИБКА: Не установлены необходимые библиотеки!")
    print(f"   {e}")
    print("\n💡 Решение:")
    print("   1. Активируйте виртуальное окружение:")
    print("      venv\\Scripts\\activate")
    print("   2. Установите зависимости:")
    print("      pip install -r requirements.txt")
    sys.exit(1)

ML_ROOT = Path(__file__).parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))

# Импортируем функции из пайплайна
try:
    from run_full_pipeline import run_model_training, run_inference
except ImportError as e:
    print(f"❌ ОШИБКА импорта: {e}")
    print(f"   Убедитесь, что вы находитесь в директории ML/")
    sys.exit(1)


def print_experiment_header(preset_name: str):
    """Печатает заголовок эксперимента."""
    print("\n" + "=" * 70)
    print("🧪 ЭКСПЕРИМЕНТ С МОДЕЛЬЮ")
    print("=" * 70)
    print(f"📌 Пресет: {preset_name}")
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Запуск эксперимента с обучением модели",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python run_experiment.py --preset MORE_TRAIN
  python run_experiment.py --preset REGULARIZED --skip-features
  python run_experiment.py --preset NO_TICKER --ticker SBER
        """
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        default="MORE_TRAIN",
        choices=["BASELINE", "MORE_TRAIN", "REGULARIZED", "NO_TICKER"],
        help="Пресет конфигурации (default: MORE_TRAIN)"
    )
    
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Пропустить этап Feature Engineering"
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        default="SBER",
        help="Тикер для инференса (default: SBER)"
    )
    
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Не использовать ансамбль (только LightGBM)"
    )
    
    args = parser.parse_args()
    
    # Обновляем активный пресет в конфигурации
    config_file = ML_ROOT / "config" / "training_config.py"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Заменяем ACTIVE_PRESET
        import re
        content = re.sub(
            r"ACTIVE_PRESET = ['\"][^'\"]+['\"]",
            f"ACTIVE_PRESET = '{args.preset}'",
            content
        )
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Установлен пресет: {args.preset}")
    else:
        print(f"⚠️ Файл конфигурации не найден: {config_file}")
        return
    
    # Печатаем заголовок
    print_experiment_header(args.preset)
    
    # Запускаем пайплайн
    success = True
    
    # Этап 1: Feature Engineering (если нужно)
    if not args.skip_features:
        print("\n⏭️ Feature Engineering пропущен (используйте --skip-features для пропуска)")
        print("   Предполагаем, что features уже готовы")
    
    # Этап 2: Model Training
    print("\n" + "=" * 70)
    print("🚀 ЭТАП 1: ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)
    
    if not run_model_training():
        print("❌ Обучение завершилось с ошибками")
        success = False
        return
    
    # Этап 3: Inference
    if success:
        print("\n" + "=" * 70)
        print("🔮 ЭТАП 2: ИНФЕРЕНС")
        print("=" * 70)
        
        run_inference(
            ticker=args.ticker,
            use_ensemble=not args.no_ensemble
        )
    
    # Итоги
    print("\n" + "=" * 70)
    print(f"✅ ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 70)
    print(f"📌 Пресет: {args.preset}")
    print(f"📁 Модели: {ML_ROOT / 'data' / 'models'}")
    print(f"📁 Отчёты: {ML_ROOT / 'reports'}")
    print("\n💡 Для сравнения запустите: python compare_models.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

