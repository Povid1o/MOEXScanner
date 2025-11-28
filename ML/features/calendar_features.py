"""
Календарные признаки и Gap-анализ для Global ML Model.

Признаки:
- day_of_week (0-6) - день недели
- day_of_month (1-31) - день месяца  
- is_month_end (0/1) - последний торговый день месяца
- is_month_start (0/1) - первый торговый день месяца
- overnight_gap - гэп открытия относительно предыдущего закрытия

Все признаки либо категориальные, либо нормализованные (относительные).
"""

import numpy as np
import pandas as pd
from typing import Optional


def day_of_week(dates: pd.Series) -> pd.Series:
    """
    День недели (0 = понедельник, 6 = воскресенье).
    
    Args:
        dates: Серия дат (datetime)
        
    Returns:
        pd.Series: День недели (int 0-6)
    """
    return pd.to_datetime(dates).dt.dayofweek


def day_of_month(dates: pd.Series) -> pd.Series:
    """
    День месяца (1-31).
    
    Args:
        dates: Серия дат (datetime)
        
    Returns:
        pd.Series: День месяца (int 1-31)
    """
    return pd.to_datetime(dates).dt.day


def is_month_end(dates: pd.Series) -> pd.Series:
    """
    Флаг: является ли дата последним торговым днем месяца.
    
    Проверяем, что следующая дата в данных - это другой месяц.
    
    Args:
        dates: Серия дат (datetime)
        
    Returns:
        pd.Series: 1 если последний день месяца, 0 иначе
    """
    dates = pd.to_datetime(dates)
    current_month = dates.dt.month
    next_month = current_month.shift(-1)
    
    # Последний день месяца = следующая торговая дата имеет другой месяц
    is_end = (current_month != next_month).astype(int)
    
    # Последняя запись в данных - тоже конец (нет следующего дня)
    is_end.iloc[-1] = 1 if pd.isna(next_month.iloc[-1]) else is_end.iloc[-1]
    
    return is_end


def is_month_start(dates: pd.Series) -> pd.Series:
    """
    Флаг: является ли дата первым торговым днем месяца.
    
    Проверяем, что предыдущая дата в данных - это другой месяц.
    
    Args:
        dates: Серия дат (datetime)
        
    Returns:
        pd.Series: 1 если первый день месяца, 0 иначе
    """
    dates = pd.to_datetime(dates)
    current_month = dates.dt.month
    prev_month = current_month.shift(1)
    
    # Первый день месяца = предыдущая торговая дата имеет другой месяц
    is_start = (current_month != prev_month).astype(int)
    
    return is_start


def week_of_month(dates: pd.Series) -> pd.Series:
    """
    Номер недели в месяце (1-5).
    
    Args:
        dates: Серия дат (datetime)
        
    Returns:
        pd.Series: Номер недели в месяце (int 1-5)
    """
    dates = pd.to_datetime(dates)
    day = dates.dt.day
    # Простая формула: (день - 1) // 7 + 1
    return ((day - 1) // 7 + 1).astype(int)


def overnight_gap(open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    """
    Ночной гэп: разница между открытием и предыдущим закрытием.
    
    Формула: (open - close.shift(1)) / close.shift(1)
    
    ВАЖНО: Использует shift(1) - смотрим назад, не в будущее!
    
    Интерпретация:
    - 0 = нет гэпа
    - 0.02 = гэп вверх на 2%
    - -0.01 = гэп вниз на 1%
    
    Args:
        open_price: Серия цен открытия
        close_price: Серия цен закрытия
        
    Returns:
        pd.Series: Относительный размер гэпа
    """
    prev_close = close_price.shift(1)
    gap = (open_price - prev_close) / prev_close
    
    return gap.replace([np.inf, -np.inf], np.nan)


def overnight_gap_zscore(gap: pd.Series, window: int = 60) -> pd.Series:
    """
    Z-score ночного гэпа: нормализованный размер гэпа относительно исторического.
    
    Args:
        gap: Серия гэпов (overnight_gap)
        window: Окно для расчета статистик
        
    Returns:
        pd.Series: Z-score гэпа
    """
    rolling_mean = gap.rolling(window=window).mean()
    rolling_std = gap.rolling(window=window).std()
    
    zscore = (gap - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan)


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Основная функция: строит ВСЕ календарные признаки и гэп.
    
    Args:
        df: DataFrame с колонками 'date', 'open', 'close'
        
    Returns:
        pd.DataFrame с календарными признаками
    """
    result = pd.DataFrame(index=df.index)
    
    # Определяем колонку с датой
    date_col = 'date' if 'date' in df.columns else df.index
    if isinstance(date_col, str):
        dates = df[date_col]
    else:
        dates = pd.Series(df.index)
    
    # === КАЛЕНДАРНЫЕ ПРИЗНАКИ ===
    result['day_of_week'] = day_of_week(dates).values
    result['day_of_month'] = day_of_month(dates).values
    result['week_of_month'] = week_of_month(dates).values
    result['is_month_end'] = is_month_end(dates).values
    result['is_month_start'] = is_month_start(dates).values
    
    # === GAP-АНАЛИЗ ===
    result['overnight_gap'] = overnight_gap(df['open'], df['close']).values
    result['overnight_gap_zscore'] = overnight_gap_zscore(
        result['overnight_gap'], window=60
    ).values
    
    return result


# Список всех календарных признаков для ML
CALENDAR_FEATURE_COLUMNS = [
    'day_of_week',
    'day_of_month',
    'week_of_month',
    'is_month_end',
    'is_month_start',
    'overnight_gap',
    'overnight_gap_zscore'
]

