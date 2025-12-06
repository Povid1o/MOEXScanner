"""
🔬 Скрипт валидации модели волатильности

Проверяет:
1. Калибровку квантилей (Coverage)
2. Качество прогнозов (Accuracy)
3. Стабильность по тикерам (Consistency)
4. Визуализация прогнозов vs фактических значений
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))

from inference import GlobalQuantileModel


def calculate_coverage(df: pd.DataFrame, q_low: float = 0.16, q_high: float = 0.84) -> dict:
    """
    Проверяет калибровку квантилей.
    
    Теоретически: 68% фактических значений должны попасть в [q16, q84]
    """
    in_interval = (df['actual'] >= df['pred_q16']) & (df['actual'] <= df['pred_q84'])
    coverage = in_interval.mean()
    
    # Проверка каждого квантиля отдельно
    below_q16 = (df['actual'] < df['pred_q16']).mean()
    above_q84 = (df['actual'] > df['pred_q84']).mean()
    
    return {
        'coverage_68': coverage,
        'expected_coverage': q_high - q_low,
        'below_q16': below_q16,  # должно быть ~16%
        'above_q84': above_q84,  # должно быть ~16%
        'calibration_error': abs(coverage - (q_high - q_low))
    }


def calculate_accuracy_metrics(df: pd.DataFrame) -> dict:
    """Метрики точности прогнозов."""
    errors = df['actual'] - df['pred_q50']
    
    return {
        'mae': np.abs(errors).mean(),
        'rmse': np.sqrt((errors ** 2).mean()),
        'mape': (np.abs(errors) / df['actual'].replace(0, np.nan)).mean(),
        'bias': errors.mean(),  # систематическое смещение
        'correlation': df['actual'].corr(df['pred_q50'])
    }


def load_and_validate(ticker: str = None) -> pd.DataFrame:
    """Загружает данные и делает прогнозы для валидации."""
    
    model = GlobalQuantileModel()
    model.load_models()
    
    # Загрузка данных
    data_dir = ML_ROOT / "data" / "processed_ml"
    
    if ticker:
        files = [data_dir / f"{ticker}_ml_features.parquet"]
    else:
        files = list(data_dir.glob("*_ml_features.parquet"))
    
    all_results = []
    
    for file in files:
        if not file.exists():
            continue
            
        ticker_name = file.stem.replace('_ml_features', '')
        df = pd.read_parquet(file)
        
        # Пропускаем IMOEX (индекс)
        if ticker_name == 'IMOEX':
            continue
        
        # Получаем прогнозы
        predictions = model.predict(df, return_interval=True)
        
        # Создаём таргет - фактическая реализованная волатильность на следующий день
        # target = volatility_{t+1} (forward looking)
        if 'rv_5d' in df.columns:
            df['actual'] = df['rv_5d'].shift(-5)  # 5-дневная реализованная волатильность
        elif 'parkinson_vol_10d' in df.columns:
            df['actual'] = df['parkinson_vol_10d'].shift(-10)
        else:
            df['actual'] = df['ewma_vol_20d'].shift(-1) if 'ewma_vol_20d' in df.columns else np.nan
        
        # Объединяем
        result = pd.concat([
            df[['date', 'actual']].reset_index(drop=True),
            predictions.reset_index(drop=True)
        ], axis=1)
        
        result['ticker'] = ticker_name
        result = result.dropna(subset=['actual', 'pred_q50'])
        
        all_results.append(result)
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def run_validation():
    """Запускает полную валидацию модели."""
    
    print("=" * 70)
    print("🔬 ВАЛИДАЦИЯ МОДЕЛИ ВОЛАТИЛЬНОСТИ")
    print("=" * 70)
    
    # 1. Загрузка и прогнозы
    print("\n📊 Загрузка данных и генерация прогнозов...")
    df = load_and_validate()
    
    if df.empty:
        print("❌ Нет данных для валидации!")
        return
    
    print(f"   Загружено: {len(df):,} записей по {df['ticker'].nunique()} тикерам")
    
    # 2. Общая калибровка
    print("\n" + "=" * 70)
    print("📈 ПРОВЕРКА КАЛИБРОВКИ КВАНТИЛЕЙ")
    print("=" * 70)
    
    coverage = calculate_coverage(df)
    
    print(f"\n   Покрытие интервала [q16, q84]:")
    print(f"      Ожидаемое: {coverage['expected_coverage']:.1%}")
    print(f"      Фактическое: {coverage['coverage_68']:.1%}")
    print(f"      Ошибка калибровки: {coverage['calibration_error']:.1%}")
    
    print(f"\n   Распределение хвостов:")
    print(f"      Ниже q16: {coverage['below_q16']:.1%} (ожидается ~16%)")
    print(f"      Выше q84: {coverage['above_q84']:.1%} (ожидается ~16%)")
    
    # Оценка калибровки
    if coverage['calibration_error'] < 0.05:
        print("\n   ✅ Калибровка ОТЛИЧНАЯ (ошибка <5%)")
    elif coverage['calibration_error'] < 0.10:
        print("\n   ✓ Калибровка ХОРОШАЯ (ошибка <10%)")
    else:
        print("\n   ⚠️ Калибровка ТРЕБУЕТ УЛУЧШЕНИЯ (ошибка >10%)")
    
    # 3. Метрики точности
    print("\n" + "=" * 70)
    print("📏 МЕТРИКИ ТОЧНОСТИ")
    print("=" * 70)
    
    accuracy = calculate_accuracy_metrics(df)
    
    print(f"\n   MAE (Mean Absolute Error): {accuracy['mae']:.4f}")
    print(f"   RMSE: {accuracy['rmse']:.4f}")
    print(f"   MAPE: {accuracy['mape']:.1%}")
    print(f"   Bias (смещение): {accuracy['bias']:.4f}")
    print(f"   Корреляция: {accuracy['correlation']:.3f}")
    
    # Интерпретация
    if accuracy['correlation'] > 0.7:
        print("\n   ✅ Корреляция ВЫСОКАЯ - модель хорошо улавливает тренды")
    elif accuracy['correlation'] > 0.5:
        print("\n   ✓ Корреляция УМЕРЕННАЯ - есть потенциал улучшения")
    else:
        print("\n   ⚠️ Корреляция НИЗКАЯ - модель требует доработки")
    
    if abs(accuracy['bias']) > 0.02:
        direction = "ЗАВЫШАЕТ" if accuracy['bias'] > 0 else "ЗАНИЖАЕТ"
        print(f"   ⚠️ Модель систематически {direction} прогнозы на {abs(accuracy['bias']):.2%}")
    
    # 4. Анализ по тикерам
    print("\n" + "=" * 70)
    print("📊 АНАЛИЗ ПО ТИКЕРАМ")
    print("=" * 70)
    
    ticker_stats = []
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        cov = calculate_coverage(ticker_df)
        acc = calculate_accuracy_metrics(ticker_df)
        
        ticker_stats.append({
            'ticker': ticker,
            'n_samples': len(ticker_df),
            'coverage': cov['coverage_68'],
            'mae': acc['mae'],
            'correlation': acc['correlation']
        })
    
    stats_df = pd.DataFrame(ticker_stats).sort_values('correlation', ascending=False)
    
    print(f"\n   {'Тикер':<8} {'N':<8} {'Coverage':<10} {'MAE':<10} {'Corr':<10}")
    print("   " + "-" * 46)
    
    for _, row in stats_df.head(15).iterrows():
        cov_status = "✓" if abs(row['coverage'] - 0.68) < 0.1 else "⚠"
        print(f"   {row['ticker']:<8} {row['n_samples']:<8} {row['coverage']:.1%} {cov_status:<3} {row['mae']:.4f}    {row['correlation']:.3f}")
    
    # 5. Худшие/лучшие тикеры
    print("\n" + "=" * 70)
    print("🏆 ЛУЧШИЕ/ХУДШИЕ ТИКЕРЫ")
    print("=" * 70)
    
    print("\n   Топ-5 по корреляции:")
    for _, row in stats_df.head(5).iterrows():
        print(f"      {row['ticker']}: r={row['correlation']:.3f}")
    
    print("\n   Худшие 5 по корреляции:")
    for _, row in stats_df.tail(5).iterrows():
        print(f"      {row['ticker']}: r={row['correlation']:.3f}")
    
    # 6. Проверка на look-ahead bias
    print("\n" + "=" * 70)
    print("🔍 ПРОВЕРКА НА LOOK-AHEAD BIAS")
    print("=" * 70)
    
    # Если корреляция слишком высокая - возможен data leakage
    if accuracy['correlation'] > 0.95:
        print("\n   ⚠️ ВНИМАНИЕ: Корреляция подозрительно высокая!")
        print("      Возможен look-ahead bias или data leakage")
    else:
        print("\n   ✅ Корреляция в нормальном диапазоне")
    
    # Проверка по времени (последние данные должны быть хуже)
    df_sorted = df.sort_values('date')
    n = len(df_sorted)
    
    first_half = df_sorted.head(n // 2)
    second_half = df_sorted.tail(n // 2)
    
    corr_first = first_half['actual'].corr(first_half['pred_q50'])
    corr_second = second_half['actual'].corr(second_half['pred_q50'])
    
    print(f"\n   Корреляция (первая половина): {corr_first:.3f}")
    print(f"   Корреляция (вторая половина): {corr_second:.3f}")
    
    if corr_first > corr_second + 0.15:
        print("   ⚠️ Модель деградирует на новых данных")
    else:
        print("   ✅ Модель стабильна во времени")
    
    # 7. Итоговая оценка
    print("\n" + "=" * 70)
    print("📋 ИТОГОВАЯ ОЦЕНКА")
    print("=" * 70)
    
    score = 0
    max_score = 5
    
    if coverage['calibration_error'] < 0.05:
        score += 1
        print("   [✓] Отличная калибровка квантилей")
    else:
        print("   [ ] Калибровка требует улучшения")
    
    if accuracy['correlation'] > 0.5:
        score += 1
        print("   [✓] Приемлемая корреляция прогнозов")
    else:
        print("   [ ] Низкая корреляция прогнозов")
    
    if abs(accuracy['bias']) < 0.02:
        score += 1
        print("   [✓] Нет систематического смещения")
    else:
        print("   [ ] Есть систематическое смещение")
    
    if accuracy['correlation'] < 0.95:
        score += 1
        print("   [✓] Нет признаков data leakage")
    else:
        print("   [ ] Подозрение на data leakage")
    
    if abs(corr_first - corr_second) < 0.15:
        score += 1
        print("   [✓] Модель стабильна во времени")
    else:
        print("   [ ] Модель деградирует на новых данных")
    
    print(f"\n   ИТОГО: {score}/{max_score}")
    
    if score >= 4:
        print("   🎉 Модель ГОТОВА к production использованию")
    elif score >= 3:
        print("   ✓ Модель ПРИГОДНА с оговорками")
    else:
        print("   ⚠️ Модель ТРЕБУЕТ ДОРАБОТКИ")
    
    # Сохраняем отчет
    output_file = ML_ROOT / "reports" / "validation_detailed.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"\n   💾 Детальный отчет сохранен: {output_file}")
    
    return {
        'coverage': coverage,
        'accuracy': accuracy,
        'ticker_stats': stats_df
    }


if __name__ == "__main__":
    import io
    import sys
    
    # Перенаправляем вывод в файл
    output_file = ML_ROOT / "reports" / "validation_console.txt"
    
    # Сохраняем stdout
    original_stdout = sys.stdout
    
    # Записываем в файл и одновременно в консоль
    with open(output_file, 'w', encoding='utf-8') as f:
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = Tee(original_stdout, f)
        
        try:
            run_validation()
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
    
    print(f"\nРезультаты записаны в: {output_file}")

