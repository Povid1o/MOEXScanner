"""
Нормализованные объемные признаки для Global ML Model.

ВАЖНО: Этот модуль генерирует ТОЛЬКО нормализованные признаки!
- volume_zscore - z-score объема (безразмерный)
- volume_ratio - отношение к среднему (относительный)
- volume_spike - бинарный флаг всплеска
- vp_position, vp_width_pct - Volume Profile (нормализованные)

НЕ возвращает абсолютные значения объема!
"""

import numpy as np
import pandas as pd
from typing import Tuple


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-score объема: нормализованное отклонение от скользящего среднего.
    
    Формула: (volume - rolling_mean) / rolling_std
    
    Args:
        volume: Серия объемов
        window: Окно для расчета статистик (по умолчанию 20 дней)
        
    Returns:
        pd.Series: Z-score объема (безразмерный, нормализованный)
    """
    rolling_mean = volume.rolling(window=window).mean()
    rolling_std = volume.rolling(window=window).std()
    
    # Защита от деления на ноль
    zscore = (volume - rolling_mean) / rolling_std.replace(0, np.nan)
    
    # Обработка бесконечных значений
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    
    return zscore


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Отношение текущего объема к среднему (нормализованное).
    
    Формула: volume / rolling_mean - 1
    
    Результат интерпретируется как % отклонения от нормы:
    - 0 = на уровне среднего
    - 0.5 = на 50% выше среднего
    - -0.3 = на 30% ниже среднего
    
    Args:
        volume: Серия объемов
        window: Окно для расчета среднего
        
    Returns:
        pd.Series: Относительное отклонение от среднего
    """
    rolling_mean = volume.rolling(window=window).mean()
    ratio = (volume / rolling_mean) - 1
    return ratio.replace([np.inf, -np.inf], np.nan)


def volume_spike(volume: pd.Series, threshold: float = 2.0, window: int = 20) -> pd.Series:
    """
    Индикатор всплеска объема: превышает ли объем порог в стандартных отклонениях.
    
    Args:
        volume: Серия объемов
        threshold: Порог z-score для определения спайка (по умолчанию 2.0)
        window: Окно для расчета z-score
        
    Returns:
        pd.Series: 1 если всплеск, 0 иначе
    """
    zscore = volume_zscore(volume, window=window)
    return (zscore > threshold).astype(int)


def calculate_volume_profile_normalized(
    df: pd.DataFrame, 
    window: int = 20, 
    num_bins: int = 50
) -> pd.DataFrame:
    """
    Расчет Volume Profile с НОРМАЛИЗОВАННЫМИ выходами.
    
    КРИТИЧНО: Возвращаем только относительные метрики, НЕ абсолютные уровни!
    
    Выходные признаки:
    - vp_position: относительная позиция close к POC (нормализованная шириной VA)
    - vp_width_pct: ширина Value Area как % от текущей цены
    - vp_above_va: 1 если цена выше VA High, -1 если ниже VA Low, 0 внутри
    
    Args:
        df: DataFrame с колонками 'open', 'high', 'low', 'close', 'volume'
        window: Окно для расчета профиля
        num_bins: Количество ценовых уровней
        
    Returns:
        pd.DataFrame с нормализованными признаками Volume Profile
    """
    n = len(df)
    
    # Предаллокация массивов для скорости
    vp_position = np.full(n, np.nan)
    vp_width_pct = np.full(n, np.nan)
    vp_above_va = np.full(n, np.nan)
    
    for i in range(window, n):
        window_data = df.iloc[i-window:i]
        current_close = df.iloc[i]['close']
        
        # Границы ценового диапазона
        price_min = window_data['low'].min()
        price_max = window_data['high'].max()
        
        if price_max == price_min:
            continue
            
        bins = np.linspace(price_min, price_max, num_bins)
        
        # Распределение объема по ценовым уровням
        volume_by_price = np.zeros(len(bins) - 1)
        
        for _, row in window_data.iterrows():
            mask = (bins[:-1] >= row['low']) & (bins[1:] <= row['high'])
            count = mask.sum()
            if count > 0:
                volume_by_price[mask] += row['volume'] / count
        
        # POC: уровень с максимальным объемом
        poc_idx = volume_by_price.argmax()
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area: 70% объема вокруг POC
        total_volume = volume_by_price.sum()
        if total_volume == 0:
            continue
            
        target_volume = total_volume * 0.70
        va_volume = volume_by_price[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while va_volume < target_volume and (va_low_idx > 0 or va_high_idx < len(volume_by_price) - 1):
            if va_low_idx > 0:
                va_volume += volume_by_price[va_low_idx - 1]
                va_low_idx -= 1
            if va_high_idx < len(volume_by_price) - 1 and va_volume < target_volume:
                va_volume += volume_by_price[va_high_idx + 1]
                va_high_idx += 1
        
        va_low = bins[va_low_idx]
        va_high = bins[va_high_idx + 1] if va_high_idx < len(bins) - 1 else bins[-1]
        va_width = va_high - va_low
        
        # === НОРМАЛИЗОВАННЫЕ ПРИЗНАКИ (БЕЗ абсолютных цен!) ===
        
        # 1. Позиция цены относительно POC (нормализовано шириной VA)
        if va_width > 0:
            vp_position[i] = (current_close - poc) / va_width
        
        # 2. Ширина VA как процент от текущей цены
        if current_close > 0:
            vp_width_pct[i] = va_width / current_close
        
        # 3. Позиция относительно Value Area (категориальный)
        if current_close > va_high:
            vp_above_va[i] = 1
        elif current_close < va_low:
            vp_above_va[i] = -1
        else:
            vp_above_va[i] = 0
    
    return pd.DataFrame({
        'vp_position': vp_position,
        'vp_width_pct': vp_width_pct,
        'vp_above_va': vp_above_va
    }, index=df.index)


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Основная функция: строит ВСЕ нормализованные объемные признаки.
    
    Args:
        df: DataFrame с колонками 'open', 'high', 'low', 'close', 'volume'
        
    Returns:
        pd.DataFrame только с нормализованными признаками объема
    """
    result = pd.DataFrame(index=df.index)
    
    # Z-score объема для разных окон
    result['volume_zscore_20'] = volume_zscore(df['volume'], window=20)
    result['volume_zscore_60'] = volume_zscore(df['volume'], window=60)
    
    # Отношение к среднему
    result['volume_ratio_20'] = volume_ratio(df['volume'], window=20)
    
    # Флаг всплеска
    result['volume_spike'] = volume_spike(df['volume'], threshold=2.0, window=20)
    
    # Volume Profile (нормализованный)
    vp_df = calculate_volume_profile_normalized(df, window=20, num_bins=50)
    result = pd.concat([result, vp_df], axis=1)
    
    # Обработка infinity и NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result


# Список всех нормализованных признаков для ML
VOLUME_FEATURE_COLUMNS = [
    'volume_zscore_20',
    'volume_zscore_60',
    'volume_ratio_20',
    'volume_spike',
    'vp_position',
    'vp_width_pct',
    'vp_above_va'
]

