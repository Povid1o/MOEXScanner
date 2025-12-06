"""
Загрузчик часовых (H1) данных для внутридневных признаков.

Модуль предоставляет функции для:
- Загрузки OHLCV данных из CSV файлов с часовым таймфреймом
- Парсинга и валидации дат
- Подготовки данных для расчета внутридневных признаков

Формат входных данных (CSV из MOEX ISS API):
- Столбцы: open, close, high, low, value, volume, begin, end
- begin/end: datetime в формате 'YYYY-MM-DD HH:MM:SS'

Автор: ML Pipeline v2.0 (Intraday Features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import warnings
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hourly_data(
    ticker: str,
    data_dir: Optional[Union[str, Path]] = None,
    parse_dates: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Загружает часовые OHLCV данные для указанного тикера.
    
    Функция ищет CSV файлы в директории data/MOEX_DATA/{ticker}/1H/
    и загружает первый найденный файл с часовыми данными.
    
    Args:
        ticker: Тикер акции (например, 'SBER', 'GAZP')
        data_dir: Базовая директория с данными MOEX_DATA.
                  Если None, используется путь по умолчанию.
        parse_dates: Парсить ли столбец 'begin' как datetime
        validate: Проводить ли валидацию загруженных данных
        
    Returns:
        pd.DataFrame с колонками:
            - datetime (index): Дата и время начала свечи
            - open, high, low, close: OHLC цены
            - volume: Объем в лотах
            - value: Объем в рублях
            - date: Дата (без времени) для группировки по дням
            
    Raises:
        FileNotFoundError: Если директория или файл не найдены
        ValueError: Если данные не прошли валидацию
        
    Example:
        >>> df = load_hourly_data('SBER')
        >>> print(df.head())
                             open   close    high     low    volume  date
        datetime                                                          
        2024-04-11 10:00:00  306.7  306.85  307.81  306.00  4510230  2024-04-11
    """
    # Определяем базовую директорию
    if data_dir is None:
        # По умолчанию ищем относительно текущего файла
        current_file = Path(__file__)
        data_dir = current_file.parent.parent.parent / "data" / "MOEX_DATA"
    else:
        data_dir = Path(data_dir)
    
    # Путь к часовым данным тикера
    hourly_dir = data_dir / ticker / "1H"
    
    if not hourly_dir.exists():
        raise FileNotFoundError(
            f"Директория с часовыми данными не найдена: {hourly_dir}\n"
            f"Убедитесь, что данные загружены для тикера {ticker}"
        )
    
    # Ищем CSV файлы с часовыми данными
    csv_files = list(hourly_dir.glob("*hourly*.csv"))
    
    if not csv_files:
        # Пробуем найти любой CSV файл
        csv_files = list(hourly_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"CSV файлы не найдены в директории: {hourly_dir}\n"
            f"Ожидается формат: {ticker}_hourly_*.csv"
        )
    
    # Берем самый свежий файл (по имени)
    csv_path = sorted(csv_files)[-1]
    logger.info(f"Загрузка часовых данных: {csv_path.name}")
    
    # Загружаем CSV
    df = pd.read_csv(csv_path)
    
    # Парсим даты
    if parse_dates and 'begin' in df.columns:
        df['datetime'] = pd.to_datetime(df['begin'], errors='coerce')
        
        # Проверяем на ошибки парсинга
        null_dates = df['datetime'].isna().sum()
        if null_dates > 0:
            logger.warning(f"⚠️ {null_dates} строк с некорректными датами удалены")
            df = df.dropna(subset=['datetime'])
        
        # Устанавливаем datetime как индекс
        df = df.set_index('datetime')
        
        # Добавляем колонку с датой (без времени) для группировки
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
    
    # Приводим названия колонок к нижнему регистру
    df.columns = df.columns.str.lower()
    
    # Убираем служебные колонки
    cols_to_drop = ['begin', 'end']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Сортируем по времени
    df = df.sort_index()
    
    # Валидация данных
    if validate:
        _validate_hourly_data(df, ticker)
    
    logger.info(f"✅ Загружено {len(df)} часовых свечей для {ticker}")
    logger.info(f"   Период: {df.index.min()} - {df.index.max()}")
    
    return df


def _validate_hourly_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Валидация загруженных часовых данных.
    
    Проверяет:
    - Наличие обязательных столбцов
    - Отсутствие NaN в ценах
    - Корректность OHLC (low <= open,close <= high)
    - Положительность объемов
    
    Args:
        df: DataFrame с часовыми данными
        ticker: Тикер для логирования
        
    Raises:
        ValueError: Если данные не прошли валидацию
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"[{ticker}] Отсутствуют обязательные столбцы: {missing_cols}"
        )
    
    # Проверка на NaN в ценах
    price_cols = ['open', 'high', 'low', 'close']
    nan_counts = df[price_cols].isna().sum()
    
    if nan_counts.sum() > 0:
        logger.warning(f"[{ticker}] NaN в ценовых данных: {nan_counts.to_dict()}")
    
    # Проверка OHLC constraint: low <= min(open, close) и max(open, close) <= high
    ohlc_violations = (
        (df['low'] > df['open']) | 
        (df['low'] > df['close']) |
        (df['high'] < df['open']) | 
        (df['high'] < df['close'])
    ).sum()
    
    if ohlc_violations > 0:
        logger.warning(f"[{ticker}] ⚠️ {ohlc_violations} свечей с нарушением OHLC constraint")
    
    # Проверка объемов
    negative_volume = (df['volume'] < 0).sum()
    if negative_volume > 0:
        logger.warning(f"[{ticker}] ⚠️ {negative_volume} свечей с отрицательным объемом")


def load_hourly_data_multi(
    tickers: List[str],
    data_dir: Optional[Union[str, Path]] = None
) -> dict:
    """
    Загружает часовые данные для нескольких тикеров.
    
    Args:
        tickers: Список тикеров
        data_dir: Базовая директория с данными
        
    Returns:
        Dict[ticker, DataFrame] с часовыми данными
    """
    result = {}
    
    for ticker in tickers:
        try:
            result[ticker] = load_hourly_data(ticker, data_dir)
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Пропускаем {ticker}: {e}")
        except Exception as e:
            logger.error(f"❌ Ошибка для {ticker}: {e}")
    
    logger.info(f"📊 Загружено часовых данных для {len(result)}/{len(tickers)} тикеров")
    
    return result


def get_trading_hours(df: pd.DataFrame) -> int:
    """
    Определяет количество торговых часов в дне для данного тикера.
    
    На MOEX основная сессия: 10:00-18:50 (около 9 часов)
    С учетом вечерней сессии: до 23:50 (около 14 часов)
    
    Args:
        df: DataFrame с часовыми данными (index = datetime)
        
    Returns:
        Медианное количество свечей в торговом дне
    """
    if 'date' not in df.columns:
        df = df.copy()
        df['date'] = df.index.date
    
    candles_per_day = df.groupby('date').size()
    median_hours = int(candles_per_day.median())
    
    logger.debug(f"Торговых часов в дне: медиана={median_hours}, min={candles_per_day.min()}, max={candles_per_day.max()}")
    
    return median_hours


# === ЭКСПОРТ ===
__all__ = [
    'load_hourly_data',
    'load_hourly_data_multi',
    'get_trading_hours'
]


if __name__ == "__main__":
    # Тестовый запуск
    print("🧪 Тест загрузчика часовых данных")
    
    try:
        df = load_hourly_data('SBER')
        print(f"\n📊 Структура данных:")
        print(df.head())
        print(f"\n📈 Статистика:")
        print(df[['open', 'high', 'low', 'close', 'volume']].describe())
        
        hours = get_trading_hours(df)
        print(f"\n⏰ Торговых часов в дне: {hours}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")

