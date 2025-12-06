"""
📊 Скрипт для сравнения результатов двух моделей

Сравнивает метрики из validation_report.csv и выводит различия.

Использование:
    python compare_models.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Проверка зависимостей перед импортом
try:
    import pandas as pd
except ImportError as e:
    print("❌ ОШИБКА: Не установлен pandas!")
    print(f"   {e}")
    print("\n💡 Решение:")
    print("   1. Убедитесь, что виртуальное окружение активировано:")
    print("      venv\\Scripts\\activate")
    print("   2. Установите зависимости:")
    print("      pip install pandas")
    print("   или")
    print("      pip install -r requirements.txt")
    sys.exit(1)

ML_ROOT = Path(__file__).parent.parent
REPORTS_DIR = ML_ROOT / "reports"


def load_validation_reports():
    """Загружает все validation_report.csv из reports/."""
    reports = []
    
    # Ищем все validation_report.csv
    for report_file in REPORTS_DIR.glob("validation_report*.csv"):
        df = pd.read_csv(report_file)
        df['source_file'] = report_file.name
        reports.append(df)
    
    if len(reports) < 2:
        print("⚠️ Нужно минимум 2 отчёта для сравнения")
        print(f"   Найдено: {len(reports)}")
        print(f"   Ищите в: {REPORTS_DIR}")
        return None
    
    return pd.concat(reports, ignore_index=True)


def compare_models():
    """Сравнивает модели и выводит отчёт."""
    
    print("=" * 70)
    print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Загружаем отчёты
    reports_df = load_validation_reports()
    if reports_df is None:
        return
    
    print(f"📋 Найдено отчётов: {len(reports_df)}\n")
    
    # Ключевые метрики для сравнения
    metrics = [
        'coverage_68',
        'interval_width',
        'mae_median',
        'quantile_loss_16',
        'quantile_loss_50',
        'quantile_loss_84',
        'train_samples',
        'test_samples',
    ]
    
    # Выводим сравнение
    print("=" * 70)
    print("📈 СРАВНЕНИЕ МЕТРИК")
    print("=" * 70)
    
    for metric in metrics:
        if metric not in reports_df.columns:
            continue
        
        print(f"\n📊 {metric.upper()}:")
        print("-" * 70)
        
        for idx, row in reports_df.iterrows():
            value = row[metric]
            cutoff = row.get('cutoff_date', 'N/A')
            source = row.get('source_file', 'N/A')
            
            # Форматируем значение
            if isinstance(value, float):
                if metric in ['coverage_68']:
                    print(f"   {source:30} | {cutoff:12} | {value:.1%}")
                elif metric in ['interval_width', 'mae_median']:
                    print(f"   {source:30} | {cutoff:12} | {value:.4f}")
                else:
                    print(f"   {source:30} | {cutoff:12} | {value:.6f}")
            else:
                print(f"   {source:30} | {cutoff:12} | {value}")
    
    # Находим лучшую модель по каждой метрике
    print("\n" + "=" * 70)
    print("🏆 ЛУЧШИЕ МОДЕЛИ ПО МЕТРИКАМ")
    print("=" * 70)
    
    # Coverage (чем ближе к 0.68, тем лучше)
    if 'coverage_68' in reports_df.columns:
        reports_df['coverage_error'] = (reports_df['coverage_68'] - 0.68).abs()
        best_coverage = reports_df.loc[reports_df['coverage_error'].idxmin()]
        print(f"\n✅ Лучшая калибровка (ближе к 68%):")
        print(f"   {best_coverage.get('source_file', 'N/A')}")
        print(f"   Coverage: {best_coverage['coverage_68']:.1%}")
        print(f"   Cutoff: {best_coverage.get('cutoff_date', 'N/A')}")
    
    # MAE (меньше = лучше)
    if 'mae_median' in reports_df.columns:
        best_mae = reports_df.loc[reports_df['mae_median'].idxmin()]
        print(f"\n✅ Лучшая точность (меньше MAE):")
        print(f"   {best_mae.get('source_file', 'N/A')}")
        print(f"   MAE: {best_mae['mae_median']:.4f}")
        print(f"   Cutoff: {best_mae.get('cutoff_date', 'N/A')}")
    
    # Interval width (уже зависит от задачи - узкий = увереннее, но может не покрывать)
    if 'interval_width' in reports_df.columns:
        print(f"\n📏 Ширина интервала:")
        for idx, row in reports_df.iterrows():
            print(f"   {row.get('source_file', 'N/A'):30} | {row['interval_width']:.4f}")
    
    # Train/Test split
    print("\n" + "=" * 70)
    print("📊 TRAIN/TEST SPLIT")
    print("=" * 70)
    
    for idx, row in reports_df.iterrows():
        train_n = row.get('train_samples', 0)
        test_n = row.get('test_samples', 0)
        total = train_n + test_n
        train_pct = (train_n / total * 100) if total > 0 else 0
        
        print(f"\n   {row.get('source_file', 'N/A')}:")
        print(f"      Train: {train_n:,} ({train_pct:.1f}%)")
        print(f"      Test:  {test_n:,} ({100-train_pct:.1f}%)")
        print(f"      Cutoff: {row.get('cutoff_date', 'N/A')}")
    
    # Рекомендации
    print("\n" + "=" * 70)
    print("💡 РЕКОМЕНДАЦИИ")
    print("=" * 70)
    
    if len(reports_df) >= 2:
        # Сравниваем coverage
        coverage_values = reports_df['coverage_68'].values
        if all(abs(c - 0.68) < 0.05 for c in coverage_values):
            print("\n✅ Все модели хорошо откалиброваны (coverage близко к 68%)")
        else:
            best_idx = reports_df['coverage_error'].idxmin()
            print(f"\n📌 Рекомендуется модель с лучшей калибровкой:")
            print(f"   {reports_df.loc[best_idx, 'source_file']}")
        
        # Сравниваем MAE
        mae_values = reports_df['mae_median'].values
        if max(mae_values) / min(mae_values) > 1.1:
            best_mae_idx = reports_df['mae_median'].idxmin()
            print(f"\n📌 Модель с лучшей точностью (MAE):")
            print(f"   {reports_df.loc[best_mae_idx, 'source_file']}")
    
    print("\n" + "=" * 70)
    print("✅ СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    compare_models()

