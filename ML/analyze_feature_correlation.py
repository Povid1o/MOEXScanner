"""
📊 Анализ корреляций между всеми признаками модели

Загружает все данные из processed_ml/, объединяет их и строит корреляционную матрицу
для всех числовых признаков, группируя по категориям.

Запуск:
    python analyze_feature_correlation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Добавляем пути для импорта
ML_ROOT = Path(__file__).parent
sys.path.insert(0, str(ML_ROOT))

# Импорт конфигурации
try:
    from config.training_config import EXCLUDE_TICKERS
except ImportError:
    EXCLUDE_TICKERS = []


# === КАТЕГОРИИ ПРИЗНАКОВ ===
FEATURE_CATEGORIES = {
    'volatility': [
        'volatility', 'realized_vol', 'garch', 'atr', 'parkinson', 'garman_klass',
        'rvol', 'rv', 'vol_ratio', 'volatility_ratio'
    ],
    'volume': [
        'volume', 'vol_zscore', 'vol_ratio', 'vp_', 'volume_profile', 
        'volume_spike', 'vp_above_va'
    ],
    'trend': [
        'rsi', 'momentum', 'dist_to_ma', 'dist_to_sma', 'dist_to_ema',
        'trend_signal', 'trend_strength', 'price_position'
    ],
    'calendar': [
        'day_of_week', 'day_of_month', 'is_month', 'overnight_gap',
        'calendar', 'weekday', 'month'
    ],
    'market': [
        'beta', 'correlation', 'index_vol', 'market', 'imoex'
    ],
    'intraday': [
        'intraday', 'hourly', 'h1_', 'range_', 'spread', 'tick_volume',
        'intraday_vol', 'intraday_range'
    ],
    'metadata': [
        'ticker_id', 'sector_id', 'sector_encoded', 'liquidity_rank',
        'is_blue_chip', 'lot_size'
    ]
}


def categorize_feature(feature_name: str) -> str:
    """Определяет категорию признака по имени."""
    feature_lower = feature_name.lower()
    
    for category, keywords in FEATURE_CATEGORIES.items():
        if any(keyword in feature_lower for keyword in keywords):
            return category
    
    return 'other'


def load_all_features(data_dir: Path) -> pd.DataFrame:
    """
    Загружает все файлы *_ml_features.parquet и объединяет в один DataFrame.
    
    Args:
        data_dir: Директория с parquet файлами
        
    Returns:
        pd.DataFrame: Объединённый датасет всех тикеров
    """
    print("=" * 70)
    print("📥 ЗАГРУЗКА ДАННЫХ")
    print("=" * 70)
    
    # Находим все файлы
    files = list(data_dir.glob("*_ml_features.parquet"))
    
    if not files:
        raise FileNotFoundError(f"Не найдены файлы в {data_dir}")
    
    print(f"📁 Найдено файлов: {len(files)}")
    
    # Загружаем все файлы (с фильтрацией исключённых тикеров)
    dfs = []
    excluded_count = 0
    
    for f in files:
        ticker = f.stem.replace('_ml_features', '')
        
        # Пропускаем исключённые тикеры
        if ticker in EXCLUDE_TICKERS:
            print(f"   ⏭️ {ticker}: ИСКЛЮЧЁН")
            excluded_count += 1
            continue
        
        df = pd.read_parquet(f)
        print(f"   • {ticker}: {len(df)} строк, {len(df.columns)} столбцов")
        dfs.append(df)
    
    if excluded_count:
        print(f"\n⚠️ Исключено тикеров: {excluded_count}")
    
    # Объединяем в один DataFrame
    global_df = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Объединённый датасет: {len(global_df):,} строк, {len(global_df.columns)} столбцов")
    
    return global_df


def prepare_features_for_correlation(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Подготавливает данные для корреляционного анализа.
    
    Исключает служебные колонки и группирует признаки по категориям.
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        Tuple[feature_df, feature_categories]:
        - feature_df: DataFrame только с числовыми признаками
        - feature_categories: Словарь {категория: [список признаков]}
    """
    print("\n" + "=" * 70)
    print("🔧 ПОДГОТОВКА ПРИЗНАКОВ")
    print("=" * 70)
    
    # Служебные колонки для исключения
    exclude_cols = [
        'date', 'ticker_id', 'sector_id',  # Метаданные
        'target_vol_1d', 'target_vol_5d', 'target_vol_10d', 'target_vol_20d',  # Целевые переменные
    ]
    
    # Исключаем служебные колонки
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Оставляем только числовые колонки
    numeric_cols = []
    for col in feature_cols:
        if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
            numeric_cols.append(col)
        elif df[col].dtype.name == 'category':
            # Категориальные признаки можно закодировать, но для корреляции лучше исключить
            continue
    
    print(f"📋 Всего признаков: {len(df.columns)}")
    print(f"📋 Числовых признаков: {len(numeric_cols)}")
    print(f"📋 Исключено служебных: {len(df.columns) - len(numeric_cols)}")
    
    # Создаём DataFrame только с числовыми признаками
    feature_df = df[numeric_cols].copy()
    
    # Обработка бесконечных значений
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    
    # Группируем признаки по категориям
    feature_categories = {}
    for col in numeric_cols:
        category = categorize_feature(col)
        if category not in feature_categories:
            feature_categories[category] = []
        feature_categories[category].append(col)
    
    print(f"\n📊 Категории признаков:")
    for category, features in sorted(feature_categories.items()):
        print(f"   • {category}: {len(features)} признаков")
    
    return feature_df, feature_categories


def calculate_correlation_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет корреляционную матрицу.
    
    Args:
        feature_df: DataFrame с признаками
        
    Returns:
        pd.DataFrame: Корреляционная матрица
    """
    print("\n" + "=" * 70)
    print("📊 РАСЧЁТ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ")
    print("=" * 70)
    
    # Вычисляем корреляцию
    corr_matrix = feature_df.corr(method='pearson')
    
    print(f"✅ Размер матрицы: {corr_matrix.shape}")
    print(f"📈 Диапазон корреляций: [{corr_matrix.min().min():.3f}, {corr_matrix.max().max():.3f}]")
    
    # Статистика по корреляциям (исключая диагональ)
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    correlations = corr_matrix.values[mask]
    
    print(f"\n📊 Статистика корреляций (без диагонали):")
    print(f"   • Среднее: {np.mean(correlations):.3f}")
    print(f"   • Медиана: {np.median(correlations):.3f}")
    print(f"   • Std: {np.std(correlations):.3f}")
    print(f"   • Min: {np.min(correlations):.3f}")
    print(f"   • Max: {np.max(correlations):.3f}")
    
    # Сильные корреляции (>0.7 или <-0.7)
    strong_pos = np.sum(correlations > 0.7)
    strong_neg = np.sum(correlations < -0.7)
    print(f"\n🔍 Сильные корреляции (|r| > 0.7):")
    print(f"   • Положительные (>0.7): {strong_pos}")
    print(f"   • Отрицательные (<-0.7): {strong_neg}")
    
    return corr_matrix


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Находит пары признаков с высокой корреляцией.
    
    Args:
        corr_matrix: Корреляционная матрица
        threshold: Порог для высокой корреляции
        
    Returns:
        pd.DataFrame: Таблица с парами признаков и их корреляциями
    """
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('abs_correlation', ascending=False)
        return high_corr_df
    else:
        return pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation', 'abs_correlation'])


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    feature_categories: Dict[str, List[str]],
    output_dir: Path,
    max_features: int = 100
):
    """
    Строит тепловую карту корреляций.
    
    Args:
        corr_matrix: Корреляционная матрица
        feature_categories: Словарь категорий признаков
        output_dir: Директория для сохранения
        max_features: Максимальное количество признаков для визуализации
    """
    print("\n" + "=" * 70)
    print("🎨 ПОСТРОЕНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 70)
    
    # Если признаков слишком много, выбираем топ по вариативности
    if len(corr_matrix) > max_features:
        print(f"⚠️ Признаков слишком много ({len(corr_matrix)}), выбираем топ-{max_features}...")
        
        # Вычисляем вариативность каждого признака (std)
        variances = corr_matrix.std(axis=1)
        top_features = variances.nlargest(max_features).index.tolist()
        
        corr_matrix = corr_matrix.loc[top_features, top_features]
        print(f"✅ Выбрано {len(corr_matrix)} признаков")
    
    # Сортируем признаки по категориям для лучшей визуализации
    category_order = ['volatility', 'volume', 'trend', 'calendar', 'market', 'intraday', 'metadata', 'other']
    sorted_features = []
    used_features = set()
    
    for category in category_order:
        if category in feature_categories:
            for feat in feature_categories[category]:
                if feat in corr_matrix.columns and feat not in used_features:
                    sorted_features.append(feat)
                    used_features.add(feat)
    
    # Добавляем оставшиеся
    for feat in corr_matrix.columns:
        if feat not in used_features:
            sorted_features.append(feat)
    
    corr_matrix_sorted = corr_matrix.loc[sorted_features, sorted_features]
    
    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(20, 18))
    
    # Строим heatmap
    sns.heatmap(
        corr_matrix_sorted,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        fmt='.2f',
        cbar_kws={'label': 'Корреляция Пирсона'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(
        f'Корреляционная матрица признаков модели\n'
        f'Всего признаков: {len(corr_matrix_sorted)}',
        fontsize=16,
        pad=20
    )
    ax.set_xlabel('Признаки', fontsize=12)
    ax.set_ylabel('Признаки', fontsize=12)
    
    # Поворачиваем подписи
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = output_dir / 'feature_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {output_path}")
    
    plt.close()


def plot_correlation_by_category(
    corr_matrix: pd.DataFrame,
    feature_categories: Dict[str, List[str]],
    output_dir: Path
):
    """
    Строит корреляции внутри и между категориями.
    
    Args:
        corr_matrix: Корреляционная матрица
        feature_categories: Словарь категорий признаков
        output_dir: Директория для сохранения
    """
    print("\n📊 Анализ корреляций по категориям...")
    
    category_order = ['volatility', 'volume', 'trend', 'calendar', 'market', 'intraday', 'metadata', 'other']
    
    # Строим матрицу средних корреляций между категориями
    category_corr = pd.DataFrame(
        index=category_order,
        columns=category_order,
        dtype=float
    )
    
    for cat1 in category_order:
        for cat2 in category_order:
            if cat1 not in feature_categories or cat2 not in feature_categories:
                category_corr.loc[cat1, cat2] = np.nan
                continue
            
            features1 = [f for f in feature_categories[cat1] if f in corr_matrix.columns]
            features2 = [f for f in feature_categories[cat2] if f in corr_matrix.columns]
            
            if not features1 or not features2:
                category_corr.loc[cat1, cat2] = np.nan
                continue
            
            # Вычисляем среднюю корреляцию между категориями
            submatrix = corr_matrix.loc[features1, features2]
            
            if cat1 == cat2:
                # Внутри категории: исключаем диагональ
                mask = ~np.eye(len(submatrix), dtype=bool)
                values = submatrix.values[mask]
            else:
                # Между категориями: все значения
                values = submatrix.values.flatten()
            
            category_corr.loc[cat1, cat2] = np.nanmean(values)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        category_corr,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Средняя корреляция'},
        ax=ax,
        linewidths=1,
        linecolor='black'
    )
    
    ax.set_title(
        'Средние корреляции между категориями признаков',
        fontsize=14,
        pad=20
    )
    ax.set_xlabel('Категория', fontsize=12)
    ax.set_ylabel('Категория', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / 'feature_correlation_by_category.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {output_path}")
    
    plt.close()
    
    return category_corr


def main():
    """Главная функция."""
    print("\n" + "🚀" * 35)
    print("   АНАЛИЗ КОРРЕЛЯЦИЙ ПРИЗНАКОВ МОДЕЛИ")
    print("   " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🚀" * 35)
    
    # Пути
    data_dir = ML_ROOT / "data" / "processed_ml"
    output_dir = ML_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Загрузка данных
    df = load_all_features(data_dir)
    
    # 2. Подготовка признаков
    feature_df, feature_categories = prepare_features_for_correlation(df)
    
    # 3. Расчёт корреляционной матрицы
    corr_matrix = calculate_correlation_matrix(feature_df)
    
    # 4. Поиск высоких корреляций
    print("\n" + "=" * 70)
    print("🔍 ПОИСК ВЫСОКИХ КОРРЕЛЯЦИЙ")
    print("=" * 70)
    
    high_corr = find_high_correlations(corr_matrix, threshold=0.7)
    
    if len(high_corr) > 0:
        print(f"\n📋 Найдено пар с |r| >= 0.7: {len(high_corr)}")
        print("\nТоп-20 самых сильных корреляций:")
        print("-" * 80)
        print(high_corr.head(20).to_string(index=False))
        
        # Сохраняем
        high_corr_path = output_dir / 'high_correlations.csv'
        high_corr.to_csv(high_corr_path, index=False)
        print(f"\n💾 Сохранено: {high_corr_path}")
    else:
        print("✅ Высоких корреляций (|r| >= 0.7) не найдено")
    
    # 5. Сохранение полной корреляционной матрицы
    print("\n💾 Сохранение корреляционной матрицы...")
    corr_matrix_path = output_dir / 'feature_correlation_matrix.csv'
    corr_matrix.to_csv(corr_matrix_path)
    print(f"✅ Сохранено: {corr_matrix_path}")
    
    # 6. Визуализации
    plot_correlation_heatmap(corr_matrix, feature_categories, output_dir, max_features=100)
    category_corr = plot_correlation_by_category(corr_matrix, feature_categories, output_dir)
    
    # Сохраняем корреляции по категориям
    category_corr_path = output_dir / 'feature_correlation_by_category.csv'
    category_corr.to_csv(category_corr_path)
    print(f"✅ Сохранено: {category_corr_path}")
    
    # Итоги
    print("\n" + "=" * 70)
    print("🏁 АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 70)
    print(f"\n📁 Результаты сохранены в: {output_dir}")
    print(f"   • feature_correlation_matrix.csv - полная матрица")
    print(f"   • feature_correlation_heatmap.png - тепловая карта")
    print(f"   • feature_correlation_by_category.png - корреляции по категориям")
    print(f"   • feature_correlation_by_category.csv - таблица по категориям")
    if len(high_corr) > 0:
        print(f"   • high_correlations.csv - пары с высокой корреляцией")
    print()


if __name__ == "__main__":
    main()

