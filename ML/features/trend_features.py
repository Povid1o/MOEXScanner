"""
Нормализованные трендовые признаки для Global ML Model.

ВАЖНО: Все ценовые признаки преобразованы в относительные!
- dist_to_sma_X - расстояние до SMA в процентах: (close / SMA) - 1
- dist_to_ema_X - расстояние до EMA в процентах: (close / EMA) - 1
- rsi - RSI (уже нормализован 0-100)
- ma_slope_norm - наклон MA в процентах от цены

НЕ возвращает абсолютные значения MA, цен и т.д.!
"""

import numpy as np
import pandas as pd
from typing import Optional


def sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average (внутренняя функция)."""
    return prices.rolling(window=window).mean()


def ema(prices: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average (внутренняя функция)."""
    return prices.ewm(span=span, adjust=False).mean()


def dist_to_ma(prices: pd.Series, ma_values: pd.Series) -> pd.Series:
    """
    Расстояние цены до скользящей средней в ОТНОСИТЕЛЬНЫХ единицах.
    
    Формула: (close / MA) - 1
    
    Интерпретация:
    - 0 = цена на уровне MA
    - 0.05 = цена на 5% выше MA  
    - -0.03 = цена на 3% ниже MA
    
    Args:
        prices: Серия цен закрытия
        ma_values: Серия значений скользящей средней
        
    Returns:
        pd.Series: Относительное расстояние до MA
    """
    dist = (prices / ma_values) - 1
    return dist.replace([np.inf, -np.inf], np.nan)


def ma_slope_normalized(ma_values: pd.Series, prices: pd.Series, window: int = 5) -> pd.Series:
    """
    Наклон скользящей средней, нормализованный ценой.
    
    Формула: (MA - MA.shift(window)) / (window * price)
    
    Это дает безразмерный показатель скорости изменения MA.
    
    Args:
        ma_values: Серия значений MA
        prices: Серия цен для нормализации
        window: Окно для расчета наклона
        
    Returns:
        pd.Series: Нормализованный наклон MA
    """
    slope = (ma_values - ma_values.shift(window)) / (window * prices)
    return slope.replace([np.inf, -np.inf], np.nan)


def momentum_normalized(prices: pd.Series, window: int = 10) -> pd.Series:
    """
    Momentum в ОТНОСИТЕЛЬНЫХ единицах (log return за период).
    
    Формула: log(close / close.shift(window))
    
    Args:
        prices: Серия цен закрытия
        window: Период для расчета momentum
        
    Returns:
        pd.Series: Log return за период (нормализованный momentum)
    """
    return np.log(prices / prices.shift(window))


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (уже нормализован в диапазоне 0-100).
    
    Args:
        prices: Серия цен закрытия
        window: Период для расчета RSI
        
    Returns:
        pd.Series: RSI (0-100)
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def price_position_ma(prices: pd.Series, ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
    """
    Положение цены относительно двух скользящих средних.
    
    Возвращает категориальный признак:
    - 1 = цена выше обеих MA
    - 0 = цена между MA
    - -1 = цена ниже обеих MA
    
    Args:
        prices: Серия цен закрытия
        ma_short: Короткая MA
        ma_long: Длинная MA
        
    Returns:
        pd.Series: Положение (-1, 0, 1)
    """
    above_short = (prices > ma_short).astype(int)
    above_long = (prices > ma_long).astype(int)
    return above_short + above_long - 1


def trend_signal(ma_short: pd.Series, ma_long: pd.Series, threshold: float = 0.01) -> pd.Series:
    """
    Сигнал тренда на основе пересечения MA (категориальный).
    
    - 1 = uptrend (короткая MA выше длинной больше чем на threshold)
    - -1 = downtrend (короткая MA ниже длинной больше чем на threshold)
    - 0 = sideways (MA близки друг к другу)
    
    Args:
        ma_short: Короткая MA
        ma_long: Длинная MA  
        threshold: Порог для определения тренда (в долях)
        
    Returns:
        pd.Series: Сигнал тренда (-1, 0, 1)
    """
    diff = (ma_short - ma_long) / ma_long
    trend = np.where(diff > threshold, 1, np.where(diff < -threshold, -1, 0))
    return pd.Series(trend, index=ma_short.index)


def trend_strength(ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
    """
    Сила тренда: насколько далеко разошлись MA (нормализовано).
    
    Формула: abs(ma_short - ma_long) / ma_long
    
    Args:
        ma_short: Короткая MA
        ma_long: Длинная MA
        
    Returns:
        pd.Series: Сила тренда (безразмерная)
    """
    strength = np.abs(ma_short - ma_long) / ma_long
    return strength.replace([np.inf, -np.inf], np.nan)


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Основная функция: строит ВСЕ нормализованные трендовые признаки.
    
    КРИТИЧНО: Возвращает ТОЛЬКО относительные признаки!
    
    Args:
        df: DataFrame с колонкой 'close'
        
    Returns:
        pd.DataFrame только с нормализованными признаками тренда
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']
    
    # === Расчет MA (внутренние, не экспортируются) ===
    sma_20 = sma(close, window=20)
    sma_50 = sma(close, window=50)
    sma_200 = sma(close, window=200)
    ema_20 = ema(close, span=20)
    ema_50 = ema(close, span=50)
    
    # === НОРМАЛИЗОВАННЫЕ ПРИЗНАКИ ===
    
    # Расстояние до MA (относительное)
    result['dist_to_sma_20'] = dist_to_ma(close, sma_20)
    result['dist_to_sma_50'] = dist_to_ma(close, sma_50)
    result['dist_to_sma_200'] = dist_to_ma(close, sma_200)
    result['dist_to_ema_20'] = dist_to_ma(close, ema_20)
    result['dist_to_ema_50'] = dist_to_ma(close, ema_50)
    
    # Наклон MA (нормализованный)
    result['sma_20_slope_norm'] = ma_slope_normalized(sma_20, close, window=5)
    result['sma_50_slope_norm'] = ma_slope_normalized(sma_50, close, window=5)
    
    # Momentum (log return за период)
    result['momentum_10'] = momentum_normalized(close, window=10)
    result['momentum_20'] = momentum_normalized(close, window=20)
    
    # RSI (уже нормализован 0-100)
    result['rsi_14'] = rsi(close, window=14)
    
    # Положение цены относительно MA
    result['price_position_ma'] = price_position_ma(close, sma_20, sma_50)
    
    # Сигнал тренда
    result['trend_signal'] = trend_signal(sma_20, sma_50, threshold=0.01)
    result['trend_strength'] = trend_strength(sma_20, sma_50)
    
    # Обработка infinity и NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result


# Список всех нормализованных признаков тренда для ML
TREND_FEATURE_COLUMNS = [
    'dist_to_sma_20',
    'dist_to_sma_50', 
    'dist_to_sma_200',
    'dist_to_ema_20',
    'dist_to_ema_50',
    'sma_20_slope_norm',
    'sma_50_slope_norm',
    'momentum_10',
    'momentum_20',
    'rsi_14',
    'price_position_ma',
    'trend_signal',
    'trend_strength'
]

