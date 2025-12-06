"""
Внутридневные признаки (Intraday Features) для Global ML Model.

Модуль агрегирует часовые (H1) данные в дневные признаки волатильности:

- IVR (Intraday Volatility Realized): Реализованная волатильность из часовых returns
- OPM (Opening Momentum): Momentum первого часа торгов (10:00-11:00)
- VDS (Volatility Distribution Skew): Асимметрия распределения часовых returns
- POCS (POC Session Shift): Дрейф объема между утренней и вечерней сессиями

КРИТИЧНО: Все признаки НОРМАЛИЗОВАНЫ (относительные, безразмерные)!
НЕ содержит абсолютных значений цен или объемов.

Автор: ML Pipeline v2.0 (Intraday Features)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === КОНСТАНТЫ ===

# Часы торговых сессий на MOEX
MORNING_SESSION_START = 10  # 10:00 - начало основной сессии
MORNING_SESSION_END = 14    # 14:00 - условная граница утро/вечер
EVENING_SESSION_START = 14  # 14:00
EVENING_SESSION_END = 19    # 19:00 - конец основной сессии (до вечерней)

# Для аннуализации внутридневной волатильности
# MOEX торгуется ~9-14 часов в день (с вечерней сессией)
DEFAULT_TRADING_HOURS = 9   # Основная сессия
TRADING_DAYS_YEAR = 252


def intraday_volatility_realized(
    hourly_df: pd.DataFrame,
    annualize: bool = True,
    trading_hours: int = DEFAULT_TRADING_HOURS
) -> pd.Series:
    """
    IVR (Intraday Volatility Realized): Волатильность из часовых log returns.
    
    Формула: std(log_returns_hourly) * sqrt(hours_per_day * 252)
    
    Преимущество перед дневной RV: более точная оценка "истинной" волатильности,
    т.к. использует внутридневную информацию вместо только close-to-close.
    
    Args:
        hourly_df: DataFrame с часовыми OHLCV (index = datetime)
        annualize: Аннуализировать ли волатильность
        trading_hours: Количество торговых часов в дне
        
    Returns:
        pd.Series с дневным IVR, индекс = date
    """
    df = hourly_df.copy()
    
    # Вычисляем часовые log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Извлекаем дату из индекса
    if 'date' not in df.columns:
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
    
    # Группируем по дням и считаем std
    daily_vol = df.groupby('date')['log_return'].apply(
        lambda x: x.std() if len(x) > 1 else np.nan
    )
    
    # Аннуализация: sqrt(hours_per_day * 252)
    if annualize:
        daily_vol = daily_vol * np.sqrt(trading_hours * TRADING_DAYS_YEAR)
    
    daily_vol.name = 'ivr'
    
    return daily_vol


def opening_momentum(hourly_df: pd.DataFrame) -> pd.Series:
    """
    OPM (Opening Momentum): Return первого часа торгов (10:00-11:00).
    
    Формула: (close_10am - open_10am) / open_10am
    
    Логика: Первый час после открытия часто задает тон всей торговой сессии.
    Сильный OPM может сигнализировать о направленном движении в течение дня.
    
    Args:
        hourly_df: DataFrame с часовыми OHLCV (index = datetime)
        
    Returns:
        pd.Series с дневным OPM, индекс = date
    """
    df = hourly_df.copy()
    
    # Извлекаем час из индекса
    df['hour'] = df.index.hour
    
    # Извлекаем дату
    if 'date' not in df.columns:
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
    
    # Фильтруем первый час основной сессии (10:00)
    first_hour = df[df['hour'] == MORNING_SESSION_START].copy()
    
    if len(first_hour) == 0:
        logger.warning("⚠️ Нет данных для часа 10:00, пробуем час 9:00")
        first_hour = df[df['hour'] == 9].copy()
    
    if len(first_hour) == 0:
        logger.warning("⚠️ Нет данных для расчета OPM")
        return pd.Series(dtype=float, name='opm')
    
    # Вычисляем return первого часа
    first_hour['opm'] = (first_hour['close'] - first_hour['open']) / first_hour['open']
    
    # Группируем по дате (на случай дубликатов)
    opm = first_hour.groupby('date')['opm'].first()
    opm.name = 'opm'
    
    return opm


def volatility_distribution_skew(hourly_df: pd.DataFrame) -> pd.Series:
    """
    VDS (Volatility Distribution Skew): Асимметрия распределения часовых returns.
    
    Формула: skewness(hourly_returns)
    
    Интерпретация:
    - Положительная skew: больше экстремальных положительных returns (upside risk)
    - Отрицательная skew: больше экстремальных отрицательных returns (downside risk)
    - Около 0: симметричное распределение
    
    Args:
        hourly_df: DataFrame с часовыми OHLCV (index = datetime)
        
    Returns:
        pd.Series с дневным VDS (skewness), индекс = date
    """
    df = hourly_df.copy()
    
    # Вычисляем часовые log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Извлекаем дату
    if 'date' not in df.columns:
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
    
    # Группируем по дням и считаем skewness
    # Используем pandas skew (Fisher-Pearson standardized moment)
    vds = df.groupby('date')['log_return'].apply(
        lambda x: x.skew() if len(x) >= 3 else np.nan  # Минимум 3 точки для skew
    )
    
    vds.name = 'vds'
    
    return vds


def volume_session_drift(hourly_df: pd.DataFrame) -> pd.Series:
    """
    POCS (POC Session Shift): Дрейф объема между утренней и вечерней сессиями.
    
    Формула: (Volume_PM - Volume_AM) / Volume_Total
    
    Интерпретация:
    - Положительный: больше активности во второй половине дня (institutional flow)
    - Отрицательный: больше активности утром (retail/news driven)
    - Около 0: равномерное распределение объема
    
    Упрощенная версия POC Shift, которая захватывает суть без сложного
    расчета Volume Profile.
    
    Args:
        hourly_df: DataFrame с часовыми OHLCV (index = datetime)
        
    Returns:
        pd.Series с дневным POCS, индекс = date
    """
    df = hourly_df.copy()
    
    # Извлекаем час
    df['hour'] = df.index.hour
    
    # Извлекаем дату
    if 'date' not in df.columns:
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
    
    # Классифицируем свечи по сессиям
    df['session'] = np.where(
        df['hour'] < EVENING_SESSION_START,
        'AM',  # Утренняя сессия (10:00-13:59)
        'PM'   # Вечерняя сессия (14:00+)
    )
    
    # Группируем по дате и сессии
    session_volume = df.pivot_table(
        values='volume',
        index='date',
        columns='session',
        aggfunc='sum',
        fill_value=0
    )
    
    # Обработка отсутствующих столбцов
    if 'AM' not in session_volume.columns:
        session_volume['AM'] = 0
    if 'PM' not in session_volume.columns:
        session_volume['PM'] = 0
    
    # Вычисляем дрейф
    total_volume = session_volume['AM'] + session_volume['PM']
    
    # Защита от деления на ноль
    pocs = (session_volume['PM'] - session_volume['AM']) / total_volume.replace(0, np.nan)
    pocs.name = 'pocs'
    
    return pocs


def intraday_range_ratio(hourly_df: pd.DataFrame) -> pd.Series:
    """
    IRR (Intraday Range Ratio): Отношение внутридневного диапазона к средней волатильности.
    
    Формула: (max(high) - min(low)) / (sum(high - low) / n)
    
    Интерпретация:
    - > 1: Экстремальные движения (гэпы внутри дня)
    - = 1: Нормальное распределение диапазонов
    - < 1: Свечи перекрывают друг друга (консолидация)
    
    Args:
        hourly_df: DataFrame с часовыми OHLCV (index = datetime)
        
    Returns:
        pd.Series с дневным IRR, индекс = date
    """
    df = hourly_df.copy()
    
    if 'date' not in df.columns:
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
    
    # Считаем дневные агрегаты
    daily_stats = df.groupby('date').agg({
        'high': ['max', lambda x: (x - df.loc[x.index, 'low']).mean()],
        'low': 'min'
    })
    
    # Переименовываем колонки
    daily_stats.columns = ['high_max', 'avg_range', 'low_min']
    
    # Дневной диапазон
    daily_range = daily_stats['high_max'] - daily_stats['low_min']
    
    # Средний часовой диапазон * кол-во свечей
    candles_per_day = df.groupby('date').size()
    expected_range = daily_stats['avg_range'] * candles_per_day
    
    # IRR
    irr = daily_range / expected_range.replace(0, np.nan)
    irr.name = 'irr'
    
    return irr


def hourly_volume_concentration(hourly_df: pd.DataFrame) -> pd.Series:
    """
    HVC (Hourly Volume Concentration): Herfindahl Index объема по часам.
    
    Формула: sum((volume_i / total_volume)^2)
    
    Интерпретация:
    - Высокий HVC: объем сконцентрирован в нескольких часах (всплески)
    - Низкий HVC: объем равномерно распределен по дню
    
    Args:
        hourly_df: DataFrame с часовыми OHLCV (index = datetime)
        
    Returns:
        pd.Series с дневным HVC (Herfindahl Index), индекс = date
    """
    df = hourly_df.copy()
    
    if 'date' not in df.columns:
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
    
    def herfindahl(group):
        total = group['volume'].sum()
        if total == 0:
            return np.nan
        shares = group['volume'] / total
        return (shares ** 2).sum()
    
    # include_groups=False для избежания FutureWarning в pandas 2.x
    hvc = df.groupby('date').apply(herfindahl, include_groups=False)
    hvc.name = 'hvc'
    
    return hvc


def build_intraday_features(
    hourly_df: pd.DataFrame,
    trading_hours: Optional[int] = None
) -> pd.DataFrame:
    """
    Главная функция: строит ВСЕ внутридневные признаки из часовых данных.
    
    Принимает DataFrame с часовыми свечами и возвращает DataFrame
    с дневными признаками (одна строка на день).
    
    Args:
        hourly_df: DataFrame с часовыми OHLCV данными
                   Ожидаемый формат: index=datetime, columns=[open, high, low, close, volume]
        trading_hours: Количество торговых часов в дне (для аннуализации IVR).
                       Если None, автоматически определяется из данных.
                       
    Returns:
        pd.DataFrame с колонками:
            - ivr: Intraday Volatility Realized (аннуализированная)
            - opm: Opening Momentum (return первого часа)
            - vds: Volatility Distribution Skew
            - pocs: POC Session Shift (дрейф объема)
            - irr: Intraday Range Ratio
            - hvc: Hourly Volume Concentration (Herfindahl)
        
        Index = date (дата торгового дня)
        
    Example:
        >>> from features.Loaders.load_hourly import load_hourly_data
        >>> hourly = load_hourly_data('SBER')
        >>> intraday = build_intraday_features(hourly)
        >>> print(intraday.head())
                      ivr       opm       vds      pocs       irr       hvc
        date                                                                 
        2024-04-11  0.152    0.0005   -0.123     0.15     1.02     0.14
    """
    logger.info("🔧 Построение внутридневных признаков...")
    
    # Валидация входных данных
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in hourly_df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    
    # Определяем количество торговых часов
    if trading_hours is None:
        if 'date' not in hourly_df.columns:
            hourly_df = hourly_df.copy()
            hourly_df['date'] = hourly_df.index.date
        
        candles_per_day = hourly_df.groupby('date').size()
        trading_hours = int(candles_per_day.median())
        logger.info(f"   Определено торговых часов в дне: {trading_hours}")
    
    # === Расчет всех признаков ===
    
    features = pd.DataFrame()
    
    # 1. IVR - Intraday Volatility Realized
    logger.info("   • IVR (Intraday Volatility Realized)...")
    features['ivr'] = intraday_volatility_realized(hourly_df, trading_hours=trading_hours)
    
    # 2. OPM - Opening Momentum
    logger.info("   • OPM (Opening Momentum)...")
    features['opm'] = opening_momentum(hourly_df)
    
    # 3. VDS - Volatility Distribution Skew
    logger.info("   • VDS (Volatility Distribution Skew)...")
    features['vds'] = volatility_distribution_skew(hourly_df)
    
    # 4. POCS - POC Session Shift
    logger.info("   • POCS (Volume Session Drift)...")
    features['pocs'] = volume_session_drift(hourly_df)
    
    # 5. IRR - Intraday Range Ratio
    logger.info("   • IRR (Intraday Range Ratio)...")
    features['irr'] = intraday_range_ratio(hourly_df)
    
    # 6. HVC - Hourly Volume Concentration
    logger.info("   • HVC (Hourly Volume Concentration)...")
    features['hvc'] = hourly_volume_concentration(hourly_df)
    
    # Обработка infinity
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Статистика
    logger.info(f"✅ Построено {len(features.columns)} внутридневных признаков")
    logger.info(f"   Период: {features.index.min()} - {features.index.max()}")
    logger.info(f"   Дней: {len(features)}")
    
    # Статистика по NaN
    nan_pct = features.isna().mean() * 100
    for col, pct in nan_pct.items():
        if pct > 0:
            logger.warning(f"   ⚠️ {col}: {pct:.1f}% NaN")
    
    return features


# Список всех генерируемых колонок (для документации и валидации)
INTRADAY_FEATURE_COLUMNS: List[str] = [
    'ivr',   # Intraday Volatility Realized
    'opm',   # Opening Momentum
    'vds',   # Volatility Distribution Skew
    'pocs',  # POC Session Shift (Volume Drift)
    'irr',   # Intraday Range Ratio
    'hvc'    # Hourly Volume Concentration
]


# === ЭКСПОРТ ===
__all__ = [
    'build_intraday_features',
    'intraday_volatility_realized',
    'opening_momentum',
    'volatility_distribution_skew',
    'volume_session_drift',
    'intraday_range_ratio',
    'hourly_volume_concentration',
    'INTRADAY_FEATURE_COLUMNS'
]


if __name__ == "__main__":
    # Тестовый запуск
    print("🧪 Тест модуля внутридневных признаков")
    
    try:
        from Loaders.load_hourly import load_hourly_data
        
        # Загружаем часовые данные
        hourly = load_hourly_data('SBER')
        
        # Строим признаки
        intraday = build_intraday_features(hourly)
        
        print(f"\n📊 Результат:")
        print(intraday.head(10))
        
        print(f"\n📈 Статистика:")
        print(intraday.describe())
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

