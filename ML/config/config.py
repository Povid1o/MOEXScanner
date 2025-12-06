"""
Конфигурация для ML workspace
"""

import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MOEX_DATA_DIR = DATA_DIR / 'MOEX_DATA'

# Настройки данных
DEFAULT_TIMEFRAMES = ['1D', '1H']
DEFAULT_TICKERS = [
    'AFKS', 'AFLT', 'ALRS', 'BELU', 'BSPB', 'CHMF', 'FIVE', 
    'GAZP', 'GMKN', 'HYDR', 'IRAO', 'LENT', 'LKOH', 'MAGN', 
    'MGNT', 'MTSS', 'NLMK', 'NVTK', 'OZON', 'PIKK', 'PLZL', 
    'ROSN', 'RTKM', 'SBER', 'SNGS', 'TATN', 'TCSG', 'VKCO', 
    'VTBR', 'YNDX'
]

# Настройки машинного обучения
ML_CONFIG = {
    'train_test_split': 0.8,
    'random_state': 42,
    'cross_validation_folds': 5,
    'feature_selection_threshold': 0.01
}

# Настройки бэктестинга
BACKTEST_CONFIG = {
    'initial_capital': 100000,  # Начальный капитал в рублях
    'commission': 0.0005,       # Комиссия (0.05%)
    'slippage': 0.0001,        # Проскальзывание (0.01%)
    'max_position_size': 0.1   # Максимальный размер позиции (10% от капитала)
}

# Настройки визуализации
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'font_size': 12,
    'save_format': 'png'
}

# Создание директорий если их нет
def ensure_directories():
    """Создает необходимые директории если их нет"""
    directories = [
        DATA_DIR,
        MOEX_DATA_DIR,
        BASE_DIR / 'models',
        BASE_DIR / 'features',
        BASE_DIR / 'backtest',
        BASE_DIR / 'explainability',
        BASE_DIR / 'utils',
        BASE_DIR / 'config'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Инициализация при импорте
ensure_directories()
