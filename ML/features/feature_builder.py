"""
Главный оркестратор Feature Engineering Pipeline для Global ML Model.

Объединяет все модули признаков и обеспечивает:
1. Расчет ВСЕХ нормализованных признаков
2. Разделение на ML и Backtest выходные наборы
3. Добавление метаданных тикера (ticker_id, sector_id)
4. Валидация и очистка данных

КРИТИЧНО: ML выход НЕ содержит абсолютных значений цен/объема!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import sys

# Добавляем путь для импорта модулей
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# Импорт модулей признаков
from features.volume_features import build_volume_features, VOLUME_FEATURE_COLUMNS
from features.trend_features import build_trend_features, TREND_FEATURE_COLUMNS
from features.calendar_features import build_calendar_features, CALENDAR_FEATURE_COLUMNS

# Импорт конфигурации
try:
    from config import get_ticker_metadata, encode_metadata_features
except ImportError:
    # Fallback если config не найден
    def get_ticker_metadata(ticker): return None
    def encode_metadata_features(ticker): return {}


# === ЗАПРЕЩЕННЫЕ СТОЛБЦЫ ДЛЯ ML ВЫХОДА ===
FORBIDDEN_ML_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'value',
    'sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50',  # Абсолютные MA
    'vp_poc', 'vp_va_high', 'vp_va_low',  # Абсолютные уровни VP
    'volume_ma', 'begin', 'end'
]


def validate_ml_output(df: pd.DataFrame, ticker: str) -> bool:
    """
    Валидация ML выхода: проверяет отсутствие запрещенных столбцов.
    
    Args:
        df: DataFrame с ML признаками
        ticker: Тикер для логирования
        
    Returns:
        True если валидация пройдена
        
    Raises:
        ValueError если найдены запрещенные столбцы
    """
    found_forbidden = []
    for col in df.columns:
        if col.lower() in [f.lower() for f in FORBIDDEN_ML_COLUMNS]:
            found_forbidden.append(col)
                
    if found_forbidden:
        raise ValueError(
            f"[{ticker}] ОШИБКА ВАЛИДАЦИИ: ML выход содержит запрещенные столбцы: {found_forbidden}\n"
            f"ML данные не должны содержать абсолютные значения цен/объема!"
        )
    
    return True


def handle_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка бесконечных значений: замена на NaN.
    
    Args:
        df: DataFrame с потенциальными inf значениями
        
    Returns:
        DataFrame без inf значений
    """
    return df.replace([np.inf, -np.inf], np.nan)


def build_all_features(
    df: pd.DataFrame,
    ticker: str,
    include_volatility: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Главная функция: строит ВСЕ признаки и разделяет на ML/Backtest выходы.
    
    Args:
        df: DataFrame с OHLCV и log_return
        ticker: Тикер акции
        include_volatility: Включать ли признаки волатильности из исходного df
        
    Returns:
        Tuple[ml_features, backtest_data]:
        - ml_features: ТОЛЬКО нормализованные признаки для ML
        - backtest_data: Сырые OHLCV для бэктеста
    """
    # Копируем для безопасности
    df = df.copy()
    
    # Проверяем наличие необходимых столбцов
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {missing}")
    
    print(f"  📊 Расчет признаков для {ticker}...")
    
    # === 1. ОБЪЕМНЫЕ ПРИЗНАКИ (нормализованные) ===
    print(f"    • Объемные признаки...")
    volume_features = build_volume_features(df)
    
    # === 2. ТРЕНДОВЫЕ ПРИЗНАКИ (нормализованные) ===
    print(f"    • Трендовые признаки...")
    trend_features = build_trend_features(df)
    
    # === 3. КАЛЕНДАРНЫЕ ПРИЗНАКИ + GAP ===
    print(f"    • Календарные признаки и Gap...")
    calendar_features = build_calendar_features(df)
    
    # === 4. СОБИРАЕМ ML FEATURES ===
    ml_features = pd.DataFrame(index=df.index)
    
    # Дата (для join и идентификации)
    if 'date' in df.columns:
        ml_features['date'] = df['date']
    
    # Log return (основной признак)
    if 'log_return' in df.columns:
        ml_features['log_return'] = df['log_return']
    
    # Признаки волатильности (если уже есть в данных)
    if include_volatility:
        vol_cols = [col for col in df.columns if 'vol_' in col.lower() or 'volatility' in col.lower()]
        for col in vol_cols:
            if col not in FORBIDDEN_ML_COLUMNS:
                ml_features[col] = df[col]
    
    # Добавляем все нормализованные признаки
    ml_features = pd.concat([
        ml_features,
        volume_features,
        trend_features,
        calendar_features
    ], axis=1)
    
    # === 5. ДОБАВЛЯЕМ МЕТАДАННЫЕ ТИКЕРА ===
    ml_features['ticker_id'] = ticker
    
    # Sector ID из конфигурации
    meta = get_ticker_metadata(ticker)
    ml_features['sector_id'] = meta.get('sector', 'Unknown') if meta else 'Unknown'
    
    # Дополнительные метаданные (закодированные)
    meta_features = encode_metadata_features(ticker)
    for key, value in meta_features.items():
        ml_features[key] = value
    
    # === 6. ОЧИСТКА ML FEATURES ===
    ml_features = handle_infinities(ml_features)
    
    # === 7. ВАЛИДАЦИЯ ===
    validate_ml_output(ml_features, ticker)
    
    # === 8. BACKTEST DATA (сырые цены) ===
    backtest_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    if 'date' not in df.columns and df.index.name == 'date':
        df = df.reset_index()
    
    backtest_cols_present = [col for col in backtest_columns if col in df.columns]
    backtest_data = df[backtest_cols_present].copy()
    
    print(f"    ✅ Готово: {len(ml_features.columns)} ML признаков, {len(backtest_data.columns)} Backtest столбцов")
    
    return ml_features, backtest_data


def process_single_ticker(
    ticker: str,
    data_dir: Path,
    output_ml_dir: Path,
    output_backtest_dir: Path,
    input_suffix: str = "_ohlcv_returns.parquet"
) -> bool:
    """
    Обрабатывает один тикер: загружает, считает признаки, сохраняет.
    
    Args:
        ticker: Тикер акции
        data_dir: Директория с исходными данными
        output_ml_dir: Директория для ML выхода
        output_backtest_dir: Директория для Backtest выхода
        input_suffix: Суффикс входных файлов
        
    Returns:
        True если успешно
    """
    try:
        # Загрузка
        input_path = data_dir / f"{ticker}{input_suffix}"
        df = pd.read_parquet(input_path)
        
        # Расчет признаков
        ml_features, backtest_data = build_all_features(df, ticker)
        
        # Сохранение ML features
        ml_path = output_ml_dir / f"{ticker}_ml_features.parquet"
        ml_features.to_parquet(ml_path, index=False)
        
        # Сохранение Backtest data
        backtest_path = output_backtest_dir / f"{ticker}_price_data.parquet"
        backtest_data.to_parquet(backtest_path, index=False)
        
        print(f"  💾 Сохранено: {ml_path.name}, {backtest_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка для {ticker}: {e}")
        return False


def process_all_tickers(
    data_dir: Path,
    output_ml_dir: Path,
    output_backtest_dir: Path,
    tickers: Optional[List[str]] = None
) -> Tuple[int, List[str]]:
    """
    Batch обработка всех тикеров.
    
    Args:
        data_dir: Директория с исходными данными
        output_ml_dir: Директория для ML выхода
        output_backtest_dir: Директория для Backtest выхода
        tickers: Список тикеров (если None - обрабатываем все файлы)
        
    Returns:
        Tuple[успешно_обработано, список_ошибок]
    """
    # Создаем выходные директории
    output_ml_dir.mkdir(parents=True, exist_ok=True)
    output_backtest_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем список тикеров
    if tickers is None:
        available_files = list(data_dir.glob("*_ohlcv_returns.parquet"))
        tickers = [f.stem.replace('_ohlcv_returns', '') for f in available_files]
    
    print(f"📋 Обработка {len(tickers)} тикеров...")
    print(f"   Тикеры: {tickers}\n")
    
    processed = 0
    errors = []
    
    for ticker in tickers:
        print(f"🔄 {ticker}...")
        success = process_single_ticker(
            ticker, data_dir, output_ml_dir, output_backtest_dir
        )
        if success:
            processed += 1
        else:
            errors.append(ticker)
    
    print(f"\n{'='*50}")
    print(f"✅ Обработано успешно: {processed}/{len(tickers)}")
    if errors:
        print(f"❌ Ошибки: {errors}")
    
    return processed, errors


def get_ml_feature_columns() -> List[str]:
    """
    Возвращает список ВСЕХ ML признаков (для документации и валидации).
    """
    return (
        ['date', 'log_return'] +
        VOLUME_FEATURE_COLUMNS +
        TREND_FEATURE_COLUMNS +
        CALENDAR_FEATURE_COLUMNS +
        ['ticker_id', 'sector_id', 'sector_encoded', 'liquidity_rank', 'is_blue_chip', 'lot_size_log']
    )


# === ЭКСПОРТ ===
__all__ = [
    'build_all_features',
    'process_single_ticker', 
    'process_all_tickers',
    'get_ml_feature_columns',
    'validate_ml_output',
    'FORBIDDEN_ML_COLUMNS'
]


if __name__ == "__main__":
    # Тестовый запуск
    from pathlib import Path
    
    ML_ROOT = Path(__file__).parent.parent
    DATA_DIR = ML_ROOT / "data" / "processed"
    OUTPUT_ML_DIR = ML_ROOT / "data" / "processed_ml"
    OUTPUT_BACKTEST_DIR = ML_ROOT / "data" / "backtest"
    
    print("🚀 Feature Engineering Pipeline")
    print(f"   Источник: {DATA_DIR}")
    print(f"   ML выход: {OUTPUT_ML_DIR}")
    print(f"   Backtest выход: {OUTPUT_BACKTEST_DIR}\n")
    
    process_all_tickers(DATA_DIR, OUTPUT_ML_DIR, OUTPUT_BACKTEST_DIR)

