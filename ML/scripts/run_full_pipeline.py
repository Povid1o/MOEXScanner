"""
🚀 Полный Production Pipeline для MOEX Volatility Scanner

Запуск:
    python scripts/run_full_pipeline.py [--preset PRESET] [--skip-features] [--skip-training] [--ticker SBER]

Этапы:
    1. Feature Engineering (D1 + H1 признаки)
    2. Model Training (LightGBM Quantile) - использует config/training_config.py
    3. Inference + Результаты

Параметры обучения настраиваются в: config/training_config.py

Автор: ML Pipeline v2.0
"""

import argparse
import sys
import time
import re
from pathlib import Path
from datetime import datetime

# Добавляем пути для импорта
ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))
sys.path.insert(0, str(ML_ROOT / "config"))


def print_header(title: str):
    """Печатает заголовок секции."""
    print("\n" + "=" * 70)
    print(f"🔷 {title}")
    print("=" * 70 + "\n")


def print_step(step: int, total: int, description: str):
    """Печатает номер шага."""
    print(f"\n[{step}/{total}] {description}")
    print("-" * 50)


def run_feature_engineering(include_intraday: bool = True):
    """
    Этап 1: Feature Engineering Pipeline.
    
    Загружает D1 данные, считает все признаки (включая H1 intraday),
    сохраняет в processed_ml/.
    """
    print_header("ЭТАП 1: FEATURE ENGINEERING")
    
    from features.feature_builder import process_all_tickers
    
    start_time = time.time()
    
    processed, errors = process_all_tickers(
        data_dir=ML_ROOT / "data" / "processed",
        output_ml_dir=ML_ROOT / "data" / "processed_ml",
        output_backtest_dir=ML_ROOT / "data" / "backtest",
        include_intraday=include_intraday
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Время выполнения: {elapsed:.1f} сек")
    print(f"✅ Обработано тикеров: {processed}")
    
    if errors:
        print(f"⚠️ Ошибки: {errors}")
    
    return processed > 0


def run_model_training():
    """
    Этап 2: Model Training Pipeline.
    
    Обучает глобальную квантильную модель LightGBM.
    """
    print_header("ЭТАП 2: MODEL TRAINING")
    
    # Импортируем функции из train_global_model.py
    from train_global_model import (
        Config, 
        load_all_ticker_data, 
        create_target_variable,
        time_series_split,
        prepare_lgbm_data,
        train_quantile_models,
        save_models,
        plot_feature_importance,
        generate_validation_report
    )
    import gc
    
    start_time = time.time()
    
    try:
        # 1. Загрузка данных
        global_df = load_all_ticker_data(Config.DATA_DIR)
        
        # 2. Создание целевой переменной
        global_df = create_target_variable(global_df, horizon=Config.TARGET_HORIZON)
        
        # 3. Временной split
        train_df, test_df = time_series_split(global_df, Config.TRAIN_CUTOFF_DATE)
        
        del global_df
        gc.collect()
        
        # 4. Подготовка данных для LightGBM
        data = prepare_lgbm_data(
            train_df, test_df, 
            Config.TARGET_COL,
            Config.CATEGORICAL_FEATURES
        )
        
        del train_df, test_df
        gc.collect()
        
        # 5. Обучение моделей
        models = train_quantile_models(
            data,
            Config.QUANTILES,
            Config.LGBM_PARAMS,
            Config.NUM_BOOST_ROUND,
            Config.EARLY_STOPPING_ROUNDS
        )
        
        # 6. Сохранение
        save_models(models, Config.OUTPUT_MODEL_DIR)
        
        # 7. Feature Importance
        plot_feature_importance(models, Config.REPORTS_DIR, top_n=30)
        
        # 8. Валидационный отчёт
        generate_validation_report(models, data, Config.REPORTS_DIR)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Время обучения: {elapsed:.1f} сек")
        print(f"✅ Модели сохранены в: {Config.OUTPUT_MODEL_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference(ticker: str = "SBER", use_ensemble: bool = True):
    """
    Этап 3: Inference и демонстрация результатов.
    
    Делает прогноз на последних данных указанного тикера.
    """
    print_header("ЭТАП 3: INFERENCE & RESULTS")
    
    import pandas as pd
    import numpy as np
    from inference import GlobalQuantileModel, ENSEMBLE_AVAILABLE
    
    start_time = time.time()
    
    # Загрузка модели
    print(f"📦 Режим: {'Ансамбль (LightGBM + GARCH)' if use_ensemble and ENSEMBLE_AVAILABLE else 'LightGBM'}")
    
    model = GlobalQuantileModel(
        use_ensemble=use_ensemble and ENSEMBLE_AVAILABLE,
        ensemble_weights={'lgbm': 0.7, 'garch': 0.3}
    )
    
    try:
        model.load_models()
    except FileNotFoundError as e:
        print(f"❌ Модели не найдены: {e}")
        print("   Запустите сначала этап обучения (--skip-features)")
        return False
    
    # Загрузка данных для прогноза
    data_file = ML_ROOT / "data" / "processed_ml" / f"{ticker}_ml_features.parquet"
    
    if not data_file.exists():
        print(f"❌ Файл не найден: {data_file}")
        # Пробуем найти любой файл
        available = list((ML_ROOT / "data" / "processed_ml").glob("*_ml_features.parquet"))
        if available:
            data_file = available[0]
            ticker = data_file.stem.replace("_ml_features", "")
            print(f"   Используем: {ticker}")
        else:
            print("   Нет доступных файлов с признаками")
            return False
    
    df = pd.read_parquet(data_file)
    print(f"\n📊 Данные: {ticker}, {len(df)} записей")
    
    # Прогноз на последних 30 днях
    n_predict = min(30, len(df))
    df_predict = df.tail(n_predict).copy()
    
    # Выполняем прогноз
    if use_ensemble and ENSEMBLE_AVAILABLE:
        predictions = model.predict_ensemble(df_predict, return_components=True)
    else:
        predictions = model.predict(df_predict, return_interval=True)
    
    # Добавляем дату для читаемости
    if 'date' in df_predict.columns:
        predictions['date'] = df_predict['date'].values
    
    elapsed = time.time() - start_time
    
    # === РЕЗУЛЬТАТЫ ===
    print_header("📈 РЕЗУЛЬТАТЫ ПРОГНОЗА")
    
    print(f"Тикер: {ticker}")
    print(f"Период: последние {n_predict} дней")
    print(f"Время инференса: {elapsed:.2f} сек\n")
    
    # Форматируем вывод
    display_cols = ['date'] if 'date' in predictions.columns else []
    display_cols += ['pred_q16', 'pred_q50', 'pred_q84', 'interval_width']
    
    if 'lgbm_q50' in predictions.columns:
        display_cols += ['lgbm_q50', 'garch_forecast']
    
    print("Последние 10 прогнозов:")
    print("-" * 80)
    
    display_df = predictions[display_cols].tail(10).copy()
    
    # Округляем для читаемости
    for col in display_df.columns:
        if col != 'date' and display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].round(4)
    
    print(display_df.to_string(index=False))
    
    # Статистика
    print("\n" + "-" * 80)
    print("📊 Статистика прогнозов:")
    print(f"   Медиана (q50): {predictions['pred_q50'].mean():.4f} (mean), {predictions['pred_q50'].std():.4f} (std)")
    print(f"   Интервал:      [{predictions['pred_q16'].mean():.4f}, {predictions['pred_q84'].mean():.4f}] (mean)")
    print(f"   Ширина интервала (mean): {predictions['interval_width'].mean():.4f}")
    
    # Feature Importance (топ-5)
    print("\n📋 Топ-5 признаков по важности:")
    try:
        importance = model.get_feature_importance(top_n=5)
        for _, row in importance.iterrows():
            print(f"   • {row['feature']}: {row['importance']:.1f}")
    except Exception:
        print("   (недоступно)")
    
    # Сохранение результатов
    output_file = ML_ROOT / "data" / "models" / f"{ticker}_predictions.csv"
    predictions.to_csv(output_file, index=False)
    print(f"\n💾 Прогнозы сохранены: {output_file}")
    
    return True


def set_training_preset(preset_name: str):
    """
    Устанавливает активный пресет в config/training_config.py.
    
    Args:
        preset_name: Имя пресета (BASELINE, MORE_TRAIN, REGULARIZED, NO_TICKER)
    """
    config_file = ML_ROOT / "config" / "training_config.py"
    
    if not config_file.exists():
        print(f"⚠️ Файл конфигурации не найден: {config_file}")
        print("   Используется дефолтная конфигурация")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Заменяем ACTIVE_PRESET
        original_content = content
        content = re.sub(
            r"ACTIVE_PRESET = ['\"][^'\"]+['\"]",
            f"ACTIVE_PRESET = '{preset_name}'",
            content
        )
        
        if content != original_content:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Установлен пресет: {preset_name}")
            return True
        else:
            print(f"⚠️ Пресет {preset_name} уже установлен или не найден в конфиге")
            return False
    except Exception as e:
        print(f"⚠️ Ошибка при установке пресета: {e}")
        return False


def main():
    """Главная функция."""
    
    parser = argparse.ArgumentParser(
        description="MOEX Volatility Scanner - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python scripts/run_full_pipeline.py                    # Полный pipeline
  python scripts/run_full_pipeline.py --preset MORE_TRAIN # С пресетом конфигурации
  python scripts/run_full_pipeline.py --skip-features   # Только обучение и инференс
  python scripts/run_full_pipeline.py --skip-training   # Только features и инференс
  python scripts/run_full_pipeline.py --ticker GAZP      # Прогноз для GAZP
  python scripts/run_full_pipeline.py --no-intraday      # Без H1 признаков

Пресеты конфигурации (в config/training_config.py):
  - BASELINE: Текущая модель (60/40 split)
  - MORE_TRAIN: Больше train данных (70/30 split)
  - REGULARIZED: Сильная регуляризация
  - NO_TICKER: Без ticker_id признака
        """
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["BASELINE", "MORE_TRAIN", "REGULARIZED", "NO_TICKER"],
        help="Пресет конфигурации из config/training_config.py (default: текущий)"
    )
    
    parser.add_argument(
        "--skip-features", 
        action="store_true",
        help="Пропустить этап Feature Engineering"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true", 
        help="Пропустить этап обучения модели"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SBER",
        help="Тикер для инференса (default: SBER)"
    )
    parser.add_argument(
        "--no-intraday",
        action="store_true",
        help="Не использовать внутридневные (H1) признаки"
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Не использовать ансамбль (только LightGBM)"
    )
    
    args = parser.parse_args()
    
    # Устанавливаем пресет если указан
    if args.preset:
        set_training_preset(args.preset)
    
    # Заголовок
    print("\n" + "🚀" * 35)
    print("   MOEX VOLATILITY SCANNER - FULL PIPELINE")
    print("   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if args.preset:
        print(f"   📌 Пресет: {args.preset}")
    print("🚀" * 35)
    
    total_start = time.time()
    success = True
    
    # Этап 1: Feature Engineering
    if not args.skip_features:
        if not run_feature_engineering(include_intraday=not args.no_intraday):
            print("⚠️ Feature Engineering завершился с ошибками")
            success = False
    else:
        print("\n⏭️ Feature Engineering пропущен (--skip-features)")
    
    # Этап 2: Model Training
    if not args.skip_training and success:
        if not run_model_training():
            print("⚠️ Model Training завершился с ошибками")
            success = False
    else:
        if args.skip_training:
            print("\n⏭️ Model Training пропущен (--skip-training)")
    
    # Этап 3: Inference
    if success:
        run_inference(
            ticker=args.ticker,
            use_ensemble=not args.no_ensemble
        )
    
    # Итоги
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print(f"🏁 PIPELINE ЗАВЕРШЁН")
    print(f"   Общее время: {total_elapsed:.1f} сек ({total_elapsed/60:.1f} мин)")
    print(f"   Статус: {'✅ Успешно' if success else '⚠️ С ошибками'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

