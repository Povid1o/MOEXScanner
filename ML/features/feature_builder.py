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
from features.volatility_features import build_volatility_features, VOLATILITY_FEATURE_COLUMNS
from features.market_features import build_market_features, load_index_data, MARKET_FEATURE_COLUMNS
from features.intraday_features import build_intraday_features, INTRADAY_FEATURE_COLUMNS
from features.Loaders.load_hourly import load_hourly_data

# Импорт конфигурации
try:
    from config import get_ticker_metadata, encode_metadata_features
except ImportError:
    # Fallback если config не найден
    def get_ticker_metadata(ticker: str) -> Optional[Dict]:
        return None
    def encode_metadata_features(ticker: str) -> Dict:
        return {}


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
    include_volatility: bool = True,
    index_df: Optional[pd.DataFrame] = None,
    hourly_data_dir: Optional[Path] = None,
    include_intraday: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Главная функция: строит ВСЕ признаки и разделяет на ML/Backtest выходы.
    
    Args:
        df: DataFrame с OHLCV и log_return
        ticker: Тикер акции
        include_volatility: Включать ли признаки волатильности из исходного df
        index_df: DataFrame с данными индекса IMOEX (опционально, для market features)
        hourly_data_dir: Директория с часовыми данными MOEX_DATA (для intraday features)
        include_intraday: Включать ли внутридневные признаки (требует часовых данных)
        
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
    
    # === 4. ПРИЗНАКИ ВОЛАТИЛЬНОСТИ ===
    print(f"    • Признаки волатильности...")
    volatility_features = build_volatility_features(df)
    
    # === 5. РЫНОЧНЫЕ ПРИЗНАКИ (Beta, Correlation с IMOEX) ===
    market_features = None
    if index_df is not None and ticker != 'IMOEX':
        print(f"    • Рыночные признаки (Beta, Correlation)...")
        try:
            market_features = build_market_features(df, index_df)
        except Exception as e:
            print(f"    ⚠️ Рыночные признаки недоступны: {e}")
    
    # === 6. ВНУТРИДНЕВНЫЕ ПРИЗНАКИ (из часовых данных) ===
    intraday_features = None
    if include_intraday and ticker != 'IMOEX':
        print(f"    • Внутридневные признаки (IVR, OPM, VDS, POCS)...")
        try:
            # Определяем директорию с часовыми данными
            if hourly_data_dir is None:
                hourly_data_dir = Path(__file__).parent.parent / "data" / "MOEX_DATA"
            
            # Загружаем часовые данные
            hourly_df = load_hourly_data(ticker, data_dir=hourly_data_dir)
            
            # Строим внутридневные признаки
            intraday_features = build_intraday_features(hourly_df)
            
            # Подготовка к мёрджу: убеждаемся что индекс - datetime
            if intraday_features.index.dtype == 'object':
                intraday_features.index = pd.to_datetime(intraday_features.index)
            
            print(f"    ✅ Внутридневные признаки: {len(intraday_features.columns)} колонок, {len(intraday_features)} дней")
            
        except FileNotFoundError as e:
            print(f"    ⚠️ Часовые данные не найдены для {ticker}: {e}")
            print(f"       Внутридневные признаки будут пропущены")
        except Exception as e:
            print(f"    ⚠️ Внутридневные признаки недоступны: {e}")
    
    # === 7. СОБИРАЕМ ML FEATURES ===
    ml_features = pd.DataFrame(index=df.index)
    
    # Дата (для join и идентификации)
    if 'date' in df.columns:
        ml_features['date'] = df['date']
    
    # Log return (основной признак)
    if 'log_return' in df.columns:
        ml_features['log_return'] = df['log_return']
    
    # Добавляем все нормализованные признаки
    features_to_concat = [
        ml_features,
        volume_features,
        trend_features,
        calendar_features,
        volatility_features
    ]
    
    # Добавляем market_features если они есть
    if market_features is not None:
        features_to_concat.append(market_features)
    
    ml_features = pd.concat(features_to_concat, axis=1)
    
    # === 8. МЁРДЖ ВНУТРИДНЕВНЫХ ПРИЗНАКОВ ===
    if intraday_features is not None and len(intraday_features) > 0:
        print(f"    • Мёрдж внутридневных признаков...")
        
        # Подготавливаем ключ для join
        if 'date' in ml_features.columns:
            # Создаём временный индекс для мёрджа
            ml_features_temp = ml_features.copy()
            # Приводим к normalized datetime (без времени)
            ml_features_temp['_merge_date'] = pd.to_datetime(ml_features['date']).dt.normalize()
            
            intraday_temp = intraday_features.copy()
            # Используем Series с dt.floor для приведения к дате (убираем время)
            intraday_dates = pd.Series(intraday_features.index)
            intraday_temp['_merge_date'] = intraday_dates.dt.floor('D').values
            intraday_temp = intraday_temp.reset_index(drop=True)
            
            # Мёрджим по дате
            n_before = len(ml_features)
            ml_features = ml_features_temp.merge(
                intraday_temp, 
                on='_merge_date', 
                how='left'
            )
            
            # Удаляем служебную колонку
            ml_features = ml_features.drop(columns=['_merge_date'])
            
            # Статистика по мёрджу
            n_matched = ml_features[INTRADAY_FEATURE_COLUMNS[0]].notna().sum()
            n_missing = n_before - n_matched
            
            if n_missing > 0:
                print(f"    ⚠️ {n_missing} дней без внутридневных данных (NaN заполнение)")
            
            print(f"    ✅ Внутридневные признаки добавлены: {n_matched}/{n_before} дней")
    
    # === 9. ДОБАВЛЯЕМ МЕТАДАННЫЕ ТИКЕРА ===
    ml_features['ticker_id'] = ticker
    
    # Sector ID из конфигурации
    meta = get_ticker_metadata(ticker)
    ml_features['sector_id'] = meta.get('sector', 'Unknown') if meta else 'Unknown'
    
    # Дополнительные метаданные (закодированные)
    meta_features = encode_metadata_features(ticker)
    for key, value in meta_features.items():
        ml_features[key] = value
    
    # === 10. ОЧИСТКА ML FEATURES ===
    ml_features = handle_infinities(ml_features)
    
    # Явное приведение к DataFrame после всех операций
    if not isinstance(ml_features, pd.DataFrame):
        ml_features = pd.DataFrame(ml_features)
    
    # === 11. ВАЛИДАЦИЯ ===
    validate_ml_output(ml_features, ticker)
    
    # === 12. BACKTEST DATA (сырые цены) ===
    backtest_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    if 'date' not in df.columns and df.index.name == 'date':
        df = df.reset_index()
    
    backtest_cols_present = [col for col in backtest_columns if col in df.columns]
    backtest_data = pd.DataFrame(df[backtest_cols_present].copy())
    
    print(f"    ✅ Готово: {len(ml_features.columns)} ML признаков, {len(backtest_data.columns)} Backtest столбцов")
    
    return ml_features, backtest_data


def process_single_ticker(
    ticker: str,
    data_dir: Path,
    output_ml_dir: Path,
    output_backtest_dir: Path,
    input_suffix: str = "_ohlcv_returns.parquet",
    index_df: Optional[pd.DataFrame] = None,
    hourly_data_dir: Optional[Path] = None,
    include_intraday: bool = True
) -> bool:
    """
    Обрабатывает один тикер: загружает, считает признаки, сохраняет.
    
    Args:
        ticker: Тикер акции
        data_dir: Директория с исходными данными
        output_ml_dir: Директория для ML выхода
        output_backtest_dir: Директория для Backtest выхода
        input_suffix: Суффикс входных файлов
        index_df: DataFrame с данными индекса IMOEX (для market features)
        hourly_data_dir: Директория с часовыми данными MOEX_DATA (для intraday features)
        include_intraday: Включать ли внутридневные признаки
        
    Returns:
        True если успешно
    """
    try:
        # Загрузка
        input_path = data_dir / f"{ticker}{input_suffix}"
        df = pd.read_parquet(input_path)
        
        # Расчет признаков
        ml_features, backtest_data = build_all_features(
            df, ticker, 
            index_df=index_df,
            hourly_data_dir=hourly_data_dir,
            include_intraday=include_intraday
        )
        
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
    tickers: Optional[List[str]] = None,
    include_intraday: bool = True
) -> Tuple[int, List[str]]:
    """
    Batch обработка всех тикеров.
    
    Args:
        data_dir: Директория с исходными данными
        output_ml_dir: Директория для ML выхода
        output_backtest_dir: Директория для Backtest выхода
        tickers: Список тикеров (если None - обрабатываем все файлы)
        include_intraday: Включать ли внутридневные признаки (из H1 данных)
        
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
    print(f"   Тикеры: {tickers}")
    print(f"   Внутридневные признаки: {'✅ включены' if include_intraday else '❌ отключены'}\n")
    
    # Загружаем индекс IMOEX для market features
    index_df = None
    try:
        index_df = load_index_data(data_dir)
        print(f"📈 Индекс IMOEX загружен: {len(index_df)} записей\n")
    except FileNotFoundError:
        print("⚠️ Индекс IMOEX не найден, market features будут пропущены\n")
    
    # Определяем директорию с часовыми данными
    hourly_data_dir = data_dir.parent / "MOEX_DATA" if include_intraday else None
    
    processed = 0
    errors = []
    
    for ticker in tickers:
        print(f"🔄 {ticker}...")
        success = process_single_ticker(
            ticker, data_dir, output_ml_dir, output_backtest_dir, 
            index_df=index_df,
            hourly_data_dir=hourly_data_dir,
            include_intraday=include_intraday
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
    
    Включает:
    - Базовые: date, log_return
    - Объемные признаки (VOLUME_FEATURE_COLUMNS)
    - Трендовые признаки (TREND_FEATURE_COLUMNS)
    - Календарные признаки (CALENDAR_FEATURE_COLUMNS)
    - Признаки волатильности (VOLATILITY_FEATURE_COLUMNS)
    - Рыночные признаки (MARKET_FEATURE_COLUMNS)
    - Внутридневные признаки (INTRADAY_FEATURE_COLUMNS) - NEW!
    - Метаданные тикера
    """
    return (
        ['date', 'log_return'] +
        VOLUME_FEATURE_COLUMNS +
        TREND_FEATURE_COLUMNS +
        CALENDAR_FEATURE_COLUMNS +
        VOLATILITY_FEATURE_COLUMNS +
        MARKET_FEATURE_COLUMNS +
        INTRADAY_FEATURE_COLUMNS +
        ['ticker_id', 'sector_id', 'sector_encoded', 'liquidity_rank', 'is_blue_chip', 'lot_size_log']
    )


# === ЭКСПОРТ ===
__all__ = [
    'build_all_features',
    'process_single_ticker', 
    'process_all_tickers',
    'get_ml_feature_columns',
    'validate_ml_output',
    'FORBIDDEN_ML_COLUMNS',
    # Re-export feature columns для удобства
    'VOLUME_FEATURE_COLUMNS',
    'TREND_FEATURE_COLUMNS',
    'CALENDAR_FEATURE_COLUMNS',
    'VOLATILITY_FEATURE_COLUMNS',
    'MARKET_FEATURE_COLUMNS',
    'INTRADAY_FEATURE_COLUMNS'
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

