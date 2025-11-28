"""
Features extraction and processing module.

Модуль для расчета НОРМАЛИЗОВАННЫХ признаков для Global ML Model.

Подмодули:
- volume_features: Z-score объема, Volume Profile (нормализованный)
- trend_features: dist_to_ma, RSI, momentum (нормализованные)
- calendar_features: день недели, overnight_gap
- feature_builder: главный оркестратор pipeline

Принципы нормализации:
- Цены -> относительные расстояния (dist_to_sma = close/SMA - 1)
- Объем -> z-score ((V - MA) / STD)
- MA уровни -> НЕ экспортируются в ML
"""

from .volume_features import (
    volume_zscore,
    volume_ratio,
    volume_spike,
    calculate_volume_profile_normalized,
    build_volume_features,
    VOLUME_FEATURE_COLUMNS
)

from .trend_features import (
    dist_to_ma,
    rsi,
    momentum_normalized,
    trend_signal,
    trend_strength,
    build_trend_features,
    TREND_FEATURE_COLUMNS
)

from .calendar_features import (
    day_of_week,
    day_of_month,
    is_month_end,
    overnight_gap,
    overnight_gap_zscore,
    build_calendar_features,
    CALENDAR_FEATURE_COLUMNS
)

from .feature_builder import (
    build_all_features,
    process_single_ticker,
    process_all_tickers,
    get_ml_feature_columns,
    validate_ml_output,
    FORBIDDEN_ML_COLUMNS
)


__all__ = [
    # Volume features
    'volume_zscore',
    'volume_ratio', 
    'volume_spike',
    'calculate_volume_profile_normalized',
    'build_volume_features',
    'VOLUME_FEATURE_COLUMNS',
    
    # Trend features
    'dist_to_ma',
    'rsi',
    'momentum_normalized',
    'trend_signal',
    'trend_strength',
    'build_trend_features',
    'TREND_FEATURE_COLUMNS',
    
    # Calendar features
    'day_of_week',
    'day_of_month',
    'is_month_end',
    'overnight_gap',
    'overnight_gap_zscore',
    'build_calendar_features',
    'CALENDAR_FEATURE_COLUMNS',
    
    # Feature builder
    'build_all_features',
    'process_single_ticker',
    'process_all_tickers',
    'get_ml_feature_columns',
    'validate_ml_output',
    'FORBIDDEN_ML_COLUMNS'
]
