"""
Рыночные признаки для Global ML Model.

Модуль содержит признаки связи акции с индексом IMOEX:
- Beta: коэффициент бета к рынку (систематический риск)
- Correlation: корреляция доходностей с индексом
- Index Volatility: волатильность рынка (общий risk-on/risk-off режим)

Все расчёты устойчивы к пропущенным значениям (NaN).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional


def calculate_beta(
    stock_returns: pd.Series, 
    market_returns: pd.Series, 
    window: int = 60,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Расчет бета коэффициента к рынку (индексу).
    
    Beta = Cov(stock, market) / Var(market)
    
    Beta > 1: акция более волатильна, чем рынок
    Beta < 1: акция менее волатильна, чем рынок
    Beta < 0: обратная корреляция с рынком
    
    Args:
        stock_returns: Доходности акции (log_return)
        market_returns: Доходности индекса
        window: Окно для расчёта (по умолчанию 60 дней)
        min_periods: Минимум непропущенных значений (по умолчанию 70% окна)
        
    Returns:
        pd.Series: Скользящая бета
    """
    if min_periods is None:
        min_periods = int(window * 0.7)
    
    beta_list = []
    
    for i in range(len(stock_returns)):
        if i < window:
            beta_list.append(np.nan)
            continue
        
        stock_window = stock_returns.iloc[i-window:i]
        market_window = market_returns.iloc[i-window:i]
        
        # Убираем NaN для расчёта
        valid_mask = ~(stock_window.isna() | market_window.isna())
        valid_count = valid_mask.sum()
        
        if valid_count < min_periods:
            beta_list.append(np.nan)
            continue
        
        stock_valid = stock_window[valid_mask]
        market_valid = market_window[valid_mask]
        
        covariance = stock_valid.cov(market_valid)
        market_variance = market_valid.var()
        
        beta = covariance / market_variance if market_variance > 0 else np.nan
        beta_list.append(beta)
    
    return pd.Series(beta_list, index=stock_returns.index)


def calculate_correlation(
    stock_returns: pd.Series, 
    market_returns: pd.Series, 
    window: int = 60,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Скользящая корреляция с индексом.
    
    Высокая корреляция = акция движется синхронно с рынком
    Низкая/отрицательная = акция имеет собственную динамику
    
    Args:
        stock_returns: Доходности акции
        market_returns: Доходности индекса
        window: Окно для расчёта
        min_periods: Минимум непропущенных значений
        
    Returns:
        pd.Series: Скользящая корреляция [-1, 1]
    """
    if min_periods is None:
        min_periods = int(window * 0.7)
    
    corr_list = []
    
    for i in range(len(stock_returns)):
        if i < window:
            corr_list.append(np.nan)
            continue
        
        stock_window = stock_returns.iloc[i-window:i]
        market_window = market_returns.iloc[i-window:i]
        
        # Убираем NaN
        valid_mask = ~(stock_window.isna() | market_window.isna())
        valid_count = valid_mask.sum()
        
        if valid_count < min_periods:
            corr_list.append(np.nan)
            continue
        
        corr = stock_window[valid_mask].corr(market_window[valid_mask])
        corr_list.append(corr)
    
    return pd.Series(corr_list, index=stock_returns.index)


def market_volatility(
    market_returns: pd.Series, 
    window: int = 30,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Волатильность рынка (индекса).
    
    Высокая волатильность индекса = risk-off режим, неопределённость
    Низкая волатильность = спокойный рынок, risk-on
    
    Args:
        market_returns: Доходности индекса
        window: Окно для расчёта
        min_periods: Минимум непропущенных значений
        
    Returns:
        pd.Series: Аннуализированная волатильность индекса
    """
    if min_periods is None:
        min_periods = int(window * 0.7)
    
    return market_returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)


def load_index_data(data_dir: Path) -> pd.DataFrame:
    """
    Загружает данные индекса IMOEX.
    
    Args:
        data_dir: Директория с processed данными
        
    Returns:
        DataFrame с индексом, включая нормализованную дату
    """
    index_path = data_dir / 'IMOEX_ohlcv_returns.parquet'
    
    if not index_path.exists():
        raise FileNotFoundError(f"Файл индекса не найден: {index_path}")
    
    index_df = pd.read_parquet(index_path)
    index_df['date_only'] = pd.to_datetime(index_df['date']).dt.normalize()
    
    return index_df


def build_market_features(
    df: pd.DataFrame, 
    index_df: pd.DataFrame,
    windows: List[int] = [30, 60]
) -> pd.DataFrame:
    """
    Строит все рыночные признаки для DataFrame акции.
    
    Args:
        df: DataFrame акции с колонками date, log_return
        index_df: DataFrame индекса с date_only, log_return
        windows: Окна для расчёта [30, 60]
        
    Returns:
        pd.DataFrame: Только рыночные признаки (без исходных данных)
    """
    features = pd.DataFrame(index=df.index)
    
    # Нормализуем дату для join
    df_temp = df.copy()
    df_temp['date_only'] = pd.to_datetime(df_temp['date']).dt.normalize()
    
    # Объединяем по дате
    merged = pd.merge(
        df_temp[['date_only', 'log_return']],
        index_df[['date_only', 'log_return']],
        on='date_only',
        how='left',
        suffixes=('', '_index')
    )
    
    stock_returns = merged['log_return'].reset_index(drop=True)
    market_returns = merged['log_return_index'].reset_index(drop=True)
    
    # Расчёт признаков для разных окон
    for window in windows:
        features[f'beta_{window}d'] = calculate_beta(
            stock_returns, market_returns, window=window
        ).values
        
        features[f'correlation_{window}d'] = calculate_correlation(
            stock_returns, market_returns, window=window
        ).values
        
        features[f'index_vol_{window}d'] = market_volatility(
            market_returns, window=window
        ).values
    
    # Дополнительные производные признаки
    if 'beta_60d' in features.columns and 'beta_30d' in features.columns:
        # Изменение беты (режим рынка меняется)
        features['beta_change'] = features['beta_60d'] - features['beta_30d']
    
    # Обработка бесконечных значений
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


# Список всех генерируемых колонок
MARKET_FEATURE_COLUMNS: List[str] = [
    'beta_30d', 'correlation_30d', 'index_vol_30d',
    'beta_60d', 'correlation_60d', 'index_vol_60d',
    'beta_change'
]


# === ЭКСПОРТ ===
__all__ = [
    'calculate_beta',
    'calculate_correlation',
    'market_volatility',
    'load_index_data',
    'build_market_features',
    'MARKET_FEATURE_COLUMNS'
]

