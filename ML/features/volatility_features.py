"""
Нормализованные признаки волатильности для Global ML Model.

Модуль содержит различные метрики волатильности:
- Realized Volatility (RV): rolling std на log returns
- EWMA Volatility: экспоненциальное сглаживание
- Parkinson Volatility: на основе high-low (более эффективная оценка)
- Garman-Klass Volatility: использует OHLC (наиболее эффективная оценка)
- Volatility Ratio: отношение краткосрочной к долгосрочной волатильности

Все значения аннуализированы (* sqrt(252)).
"""

import numpy as np
import pandas as pd
from typing import List


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Реализованная волатильность: rolling std на log returns.
    
    Формула: std(returns, window) * sqrt(252)
    
    Args:
        returns: Серия log returns
        window: Окно для rolling (по умолчанию 20 дней)
        
    Returns:
        pd.Series: Аннуализированная волатильность
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def ewma_volatility(returns: pd.Series, span: int = 20) -> pd.Series:
    """
    EWMA волатильность: экспоненциальное сглаживание.
    
    Более чувствительна к недавним изменениям, чем простая RV.
    
    Формула: ewm_std(returns, span) * sqrt(252)
    
    Args:
        returns: Серия log returns
        span: Период для экспоненциального сглаживания
        
    Returns:
        pd.Series: EWMA волатильность (аннуализированная)
    """
    return returns.ewm(span=span).std() * np.sqrt(252)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility: на основе high-low.
    
    Более эффективная оценка, чем close-to-close, т.к. использует
    внутридневной диапазон. Эффективность ~5x выше простой RV.
    
    Формула: sqrt(mean(log(H/L)^2) / (4*ln(2))) * sqrt(252)
    
    Args:
        high: Серия максимумов
        low: Серия минимумов
        window: Окно для rolling
        
    Returns:
        pd.Series: Parkinson волатильность (аннуализированная)
    """
    hl_ratio = np.log(high / low) ** 2
    parkinson = np.sqrt(hl_ratio.rolling(window=window).mean() / (4 * np.log(2)))
    return parkinson * np.sqrt(252)


def garman_klass_volatility(
    open_price: pd.Series, 
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    window: int = 20
) -> pd.Series:
    """
    Garman-Klass volatility: использует OHLC.
    
    Наиболее эффективная оценка волатильности, использующая
    всю информацию из OHLC. Эффективность ~8x выше простой RV.
    
    Формула: sqrt(mean(0.5*log(H/L)^2 - (2*ln(2)-1)*log(C/O)^2)) * sqrt(252)
    
    Args:
        open_price: Серия открытия
        high: Серия максимумов
        low: Серия минимумов
        close: Серия закрытия
        window: Окно для rolling
        
    Returns:
        pd.Series: Garman-Klass волатильность (аннуализированная)
    """
    hl = np.log(high / low) ** 2
    co = np.log(close / open_price) ** 2
    gk = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(gk.rolling(window=window).mean()) * np.sqrt(252)


def volatility_ratio(short_vol: pd.Series, long_vol: pd.Series) -> pd.Series:
    """
    Отношение краткосрочной к долгосрочной волатильности.
    
    Используется для определения режима волатильности:
    - > 1: волатильность растёт (expansion)
    - < 1: волатильность падает (contraction)
    - = 1: стабильный режим
    
    Args:
        short_vol: Краткосрочная волатильность (например, 5d)
        long_vol: Долгосрочная волатильность (например, 20d)
        
    Returns:
        pd.Series: Отношение волатильностей
    """
    ratio = short_vol / long_vol.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def directional_volatility(
    returns: pd.Series,
    window: int = 20
) -> tuple:
    """
    Направленная волатильность: раздельная для роста и падения.
    
    Разделяет волатильность на:
    - up_vol: волатильность положительных returns (upside risk)
    - down_vol: волатильность отрицательных returns (downside risk)
    
    Полезно для:
    - Определения асимметрии рисков
    - Идентификации режимов рынка (risk-on / risk-off)
    - Корректировки стратегий под downside risk
    
    Args:
        returns: Серия log returns
        window: Окно для rolling std
        
    Returns:
        Tuple[pd.Series, pd.Series]: (up_vol, down_vol) - аннуализированные
    """
    # Разделяем returns
    up_returns = returns.where(returns > 0, 0)
    down_returns = returns.where(returns < 0, 0)
    
    # Волатильность для каждого направления
    up_vol = up_returns.rolling(window=window).std() * np.sqrt(252)
    down_vol = down_returns.abs().rolling(window=window).std() * np.sqrt(252)
    
    return up_vol, down_vol


def volatility_asymmetry(up_vol: pd.Series, down_vol: pd.Series) -> pd.Series:
    """
    Асимметрия волатильности: отношение down_vol к up_vol.
    
    Формула: down_vol / up_vol
    
    Интерпретация:
    - > 1: больше downside risk (типично для акций)
    - = 1: симметричная волатильность
    - < 1: больше upside risk (редко)
    
    Args:
        up_vol: Upside volatility
        down_vol: Downside volatility
        
    Returns:
        pd.Series: Коэффициент асимметрии
    """
    asymmetry = down_vol / up_vol.replace(0, np.nan)
    return asymmetry.replace([np.inf, -np.inf], np.nan)


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит все признаки волатильности из DataFrame с OHLCV и log_return.
    
    Включает:
    - Realized Volatility (окна: 5, 10, 20, 30, 60)
    - EWMA Volatility
    - Parkinson Volatility (high-low)
    - Garman-Klass Volatility (OHLC)
    - Volatility Ratios (режим волатильности)
    - Directional Volatility (up_vol, down_vol) - NEW!
    - Volatility Asymmetry - NEW!
    
    Args:
        df: DataFrame с колонками: open, high, low, close, log_return
        
    Returns:
        pd.DataFrame: Признаки волатильности
    """
    features = pd.DataFrame(index=df.index)
    
    # Проверяем наличие необходимых колонок
    required = ['open', 'high', 'low', 'close', 'log_return']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    returns = df['log_return']
    
    # === REALIZED VOLATILITY (разные окна) ===
    # Добавлены окна 30 и 60 для долгосрочной волатильности
    for window in [5, 10, 20, 30, 60]:
        features[f'rv_{window}d'] = realized_volatility(returns, window=window)
    
    # === EWMA VOLATILITY ===
    for span in [10, 20]:
        features[f'ewma_vol_{span}d'] = ewma_volatility(returns, span=span)
    
    # === PARKINSON VOLATILITY ===
    for window in [10, 20]:
        features[f'parkinson_vol_{window}d'] = parkinson_volatility(
            df['high'], df['low'], window=window
        )
    
    # === GARMAN-KLASS VOLATILITY ===
    for window in [10, 20]:
        features[f'gk_vol_{window}d'] = garman_klass_volatility(
            df['open'], df['high'], df['low'], df['close'], window=window
        )
    
    # === VOLATILITY RATIOS ===
    # Краткосрочная vs долгосрочная (режим волатильности)
    features['vol_ratio_5_20'] = volatility_ratio(
        features['rv_5d'], features['rv_20d']
    )
    
    # Parkinson vs Realized (насколько внутридневные движения отличаются)
    features['vol_ratio_park_rv'] = volatility_ratio(
        features['parkinson_vol_20d'], features['rv_20d']
    )
    
    # Долгосрочный ratio (20 vs 60)
    features['vol_ratio_20_60'] = volatility_ratio(
        features['rv_20d'], features['rv_60d']
    )
    
    # === DIRECTIONAL VOLATILITY (upside/downside) ===
    up_vol_20, down_vol_20 = directional_volatility(returns, window=20)
    features['up_vol_20d'] = up_vol_20
    features['down_vol_20d'] = down_vol_20
    
    # Асимметрия волатильности (down/up ratio)
    features['vol_asymmetry_20d'] = volatility_asymmetry(up_vol_20, down_vol_20)
    
    # === VOLATILITY MOMENTUM ===
    # Изменение волатильности за последние N дней
    features['vol_momentum_5d'] = features['rv_20d'].pct_change(5)
    features['vol_momentum_10d'] = features['rv_20d'].pct_change(10)
    
    # Обработка бесконечных значений
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


# Список всех генерируемых колонок (для документации и валидации)
VOLATILITY_FEATURE_COLUMNS: List[str] = [
    # Realized Volatility (расширенные окна)
    'rv_5d', 'rv_10d', 'rv_20d', 'rv_30d', 'rv_60d',
    # EWMA Volatility
    'ewma_vol_10d', 'ewma_vol_20d',
    # Parkinson Volatility
    'parkinson_vol_10d', 'parkinson_vol_20d',
    # Garman-Klass Volatility
    'gk_vol_10d', 'gk_vol_20d',
    # Volatility Ratios
    'vol_ratio_5_20', 'vol_ratio_park_rv', 'vol_ratio_20_60',
    # Directional Volatility (NEW)
    'up_vol_20d', 'down_vol_20d', 'vol_asymmetry_20d',
    # Volatility Momentum
    'vol_momentum_5d', 'vol_momentum_10d'
]


# === ЭКСПОРТ ===
__all__ = [
    'realized_volatility',
    'ewma_volatility', 
    'parkinson_volatility',
    'garman_klass_volatility',
    'volatility_ratio',
    'directional_volatility',
    'volatility_asymmetry',
    'build_volatility_features',
    'VOLATILITY_FEATURE_COLUMNS'
]

