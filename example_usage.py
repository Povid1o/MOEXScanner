#!/usr/bin/env python3
"""
Пример использования ML workspace для анализа данных MOEX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_moex_data(ticker='SBER', timeframe='1D'):
    """
    Загружает данные MOEX для указанного тикера и таймфрейма
    
    Args:
        ticker (str): Тикер акции (например, 'SBER', 'GAZP')
        timeframe (str): Таймфрейм ('1D' для дневных данных, '1H' для почасовых)
    
    Returns:
        pd.DataFrame: DataFrame с данными
    """
    data_path = Path('data/MOEX_DATA') / ticker / timeframe
    
    # Поиск CSV файла
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"Не найден CSV файл для {ticker} {timeframe}")
    
    csv_file = csv_files[0]
    print(f"Загружаем данные из: {csv_file}")
    
    # Загрузка данных
    df = pd.read_csv(csv_file)
    
    # Преобразование даты в datetime
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
    
    return df

def basic_analysis(df, ticker):
    """
    Базовый анализ данных
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        ticker (str): Тикер акции
    """
    print(f"\n=== Анализ данных для {ticker} ===")
    print(f"Размер данных: {df.shape}")
    print(f"Период: {df.index.min()} - {df.index.max()}")
    print(f"\nОсновная статистика:")
    print(df.describe())
    
    # Проверка на пропущенные значения
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nПропущенные значения:")
        print(missing_values[missing_values > 0])
    else:
        print("\nПропущенных значений нет")

def plot_price_chart(df, ticker, timeframe):
    """
    Построение графика цен
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        ticker (str): Тикер акции
        timeframe (str): Таймфрейм
    """
    plt.figure(figsize=(12, 6))
    
    if 'CLOSE' in df.columns:
        plt.plot(df.index, df['CLOSE'], label='Цена закрытия', linewidth=2)
        plt.title(f'{ticker} - Цена закрытия ({timeframe})', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Колонка 'CLOSE' не найдена в данных")

def calculate_returns(df):
    """
    Расчет доходности
    
    Args:
        df (pd.DataFrame): DataFrame с данными
    
    Returns:
        pd.DataFrame: DataFrame с добавленными колонками доходности
    """
    df_returns = df.copy()
    
    if 'CLOSE' in df.columns:
        # Простая доходность
        df_returns['SIMPLE_RETURN'] = df['CLOSE'].pct_change()
        
        # Логарифмическая доходность
        df_returns['LOG_RETURN'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
        
        # Волатильность (скользящее окно 20 дней)
        df_returns['VOLATILITY'] = df_returns['SIMPLE_RETURN'].rolling(window=20).std()
    
    return df_returns

def main():
    """
    Основная функция для демонстрации работы с данными
    """
    print("=== MOEX Scanner ML Workspace ===")
    
    # Список доступных тикеров
    data_path = Path('data/MOEX_DATA')
    available_tickers = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"Доступные тикеры: {available_tickers}")
    
    # Анализ данных для нескольких тикеров
    tickers_to_analyze = ['SBER', 'GAZP', 'LKOH'][:2]  # Ограничиваем для демонстрации
    
    for ticker in tickers_to_analyze:
        if ticker in available_tickers:
            try:
                # Загружаем дневные данные
                df = load_moex_data(ticker, '1D')
                
                # Базовый анализ
                basic_analysis(df, ticker)
                
                # Расчет доходности
                df_with_returns = calculate_returns(df)
                
                # Построение графика
                plot_price_chart(df, ticker, '1D')
                
                print(f"\nДоходность {ticker}:")
                print(f"Средняя дневная доходность: {df_with_returns['SIMPLE_RETURN'].mean():.4f}")
                print(f"Стандартное отклонение: {df_with_returns['SIMPLE_RETURN'].std():.4f}")
                print(f"Шарп-отношение: {df_with_returns['SIMPLE_RETURN'].mean() / df_with_returns['SIMPLE_RETURN'].std():.4f}")
                
            except Exception as e:
                print(f"Ошибка при анализе {ticker}: {e}")
        else:
            print(f"Данные для тикера {ticker} не найдены")

if __name__ == "__main__":
    main()
