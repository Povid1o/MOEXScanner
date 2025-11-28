"""
Модуль конфигурации для ML pipeline
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Путь к файлу метаданных
CONFIG_DIR = Path(__file__).parent
METADATA_FILE = CONFIG_DIR / 'tickers_metadata.json'


def load_tickers_metadata() -> Dict:
    """
    Загружает метаданные тикеров из JSON файла.
    
    Returns:
        Dict: Словарь с метаданными всех тикеров
    """
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Удаляем служебные поля с описанием
    return {k: v for k, v in data.items() if not k.startswith('_')}


def get_ticker_metadata(ticker: str) -> Optional[Dict]:
    """
    Получает метаданные для конкретного тикера.
    
    Args:
        ticker: Тикер (например, 'SBER')
        
    Returns:
        Dict с метаданными или None если тикер не найден
    """
    metadata = load_tickers_metadata()
    return metadata.get(ticker.upper())


def get_tickers_by_sector(sector: str) -> List[str]:
    """
    Возвращает список тикеров в заданном секторе.
    
    Args:
        sector: Название сектора (например, 'Finance', 'OilGas')
        
    Returns:
        List[str]: Список тикеров
    """
    metadata = load_tickers_metadata()
    return [ticker for ticker, data in metadata.items() 
            if data.get('sector') == sector]


def get_blue_chips() -> List[str]:
    """
    Возвращает список голубых фишек.
    
    Returns:
        List[str]: Список тикеров голубых фишек
    """
    metadata = load_tickers_metadata()
    return [ticker for ticker, data in metadata.items() 
            if data.get('is_blue_chip') == 1]


def get_tickers_by_liquidity(top_n: Optional[int] = None) -> List[str]:
    """
    Возвращает тикеры, отсортированные по ликвидности.
    
    Args:
        top_n: Вернуть только top_n самых ликвидных (если None - все)
        
    Returns:
        List[str]: Список тикеров
    """
    metadata = load_tickers_metadata()
    sorted_tickers = sorted(
        metadata.items(), 
        key=lambda x: x[1].get('liquidity_rank', 999)
    )
    tickers = [ticker for ticker, _ in sorted_tickers]
    
    if top_n is not None:
        return tickers[:top_n]
    return tickers


def get_metadata_as_dataframe() -> pd.DataFrame:
    """
    Возвращает метаданные в виде DataFrame для удобного анализа и merge.
    
    Returns:
        pd.DataFrame: DataFrame с метаданными (индекс = ticker)
    """
    metadata = load_tickers_metadata()
    df = pd.DataFrame.from_dict(metadata, orient='index')
    df.index.name = 'ticker'
    return df


def get_sectors_list() -> List[str]:
    """
    Возвращает список всех уникальных секторов.
    
    Returns:
        List[str]: Список секторов
    """
    metadata = load_tickers_metadata()
    return list(set(data.get('sector') for data in metadata.values()))


def encode_metadata_features(ticker: str) -> Dict:
    """
    Создает числовые признаки из метаданных для ML модели.
    
    Args:
        ticker: Тикер
        
    Returns:
        Dict с числовыми признаками для модели
    """
    meta = get_ticker_metadata(ticker)
    if meta is None:
        return {}
    
    # Маппинг секторов в числовые значения
    sector_mapping = {
        'Finance': 1,
        'OilGas': 2,
        'Mining': 3,
        'Metals': 4,
        'Tech': 5,
        'Telecom': 6,
        'Retail': 7,
        'Utilities': 8,
        'Transport': 9,
        'RealEstate': 10,
        'Conglomerate': 11,
        'Consumer': 12
    }
    
    sector = meta.get('sector', '')
    return {
        'sector_encoded': sector_mapping.get(sector, 0),
        'liquidity_rank': meta.get('liquidity_rank', 30),
        'is_blue_chip': meta.get('is_blue_chip', 0),
        'lot_size_log': np.log1p(meta.get('lot_size', 1))
    }


# Экспорт основных функций
__all__ = [
    'load_tickers_metadata',
    'get_ticker_metadata',
    'get_tickers_by_sector',
    'get_blue_chips',
    'get_tickers_by_liquidity',
    'get_metadata_as_dataframe',
    'get_sectors_list',
    'encode_metadata_features'
]

