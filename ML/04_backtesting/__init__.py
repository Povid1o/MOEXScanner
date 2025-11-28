"""
Модуль бэктестинга для проверки торговых стратегий.

Основные компоненты:
- run_backtest_pipeline: Полный pipeline бэктеста
- SignalGenerator: Генерация сигналов из ML прогнозов
- TradingStrategy: Реализация торговой логики
- BacktestEngine: Симуляция сделок
- PerformanceAnalyzer: Расчёт метрик

Использование:
    python 04_backtesting/run_backtest_pipeline.py
    
    или:
    
    from backtesting.run_backtest_pipeline import run_full_backtest_pipeline
    results = run_full_backtest_pipeline()
"""

from .run_backtest_pipeline import (
    BacktestConfig,
    SignalGenerator,
    TradingStrategy,
    BacktestEngine,
    PerformanceAnalyzer,
    ReportGenerator,
    run_full_backtest_pipeline
)

__all__ = [
    'BacktestConfig',
    'SignalGenerator', 
    'TradingStrategy',
    'BacktestEngine',
    'PerformanceAnalyzer',
    'ReportGenerator',
    'run_full_backtest_pipeline'
]

