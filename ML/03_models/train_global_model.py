"""
Глобальная модель LightGBM для прогнозирования волатильности.

Этот скрипт реализует полный pipeline обучения:
1. Агрегация данных всех тикеров в единый DataFrame
2. Временной (time-series) split на train/test
3. Обучение 3 квантильных моделей (0.16, 0.50, 0.84)
4. Сохранение моделей и feature importance

КРИТИЧНО: Используется строгий временной split без shuffle!

Автор: ML Pipeline v2.0 (Global Model)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
import gc
import json
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Попытка загрузить внешнюю конфигурацию
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
    from training_config import (
        TRAIN_CUTOFF_DATE as EXT_CUTOFF,
        TARGET_HORIZON as EXT_HORIZON,
        TARGET_COL as EXT_TARGET,
        QUANTILES as EXT_QUANTILES,
        LGBM_PARAMS as EXT_LGBM_PARAMS_BASE,
        NUM_BOOST_ROUND as EXT_NUM_ROUNDS,
        EARLY_STOPPING_ROUNDS as EXT_EARLY_STOP,
        CATEGORICAL_FEATURES as EXT_CAT_FEATURES,
        EXCLUDE_TICKERS as EXT_EXCLUDE_TICKERS,
        get_active_config,
        ACTIVE_PRESET,
    )
    USE_EXTERNAL_CONFIG = True
    
    # Применяем параметры из активного пресета
    preset_config = get_active_config()
    EXT_LGBM_PARAMS = EXT_LGBM_PARAMS_BASE.copy()
    
    # Обновляем параметры из пресета
    if 'num_leaves' in preset_config:
        EXT_LGBM_PARAMS['num_leaves'] = preset_config['num_leaves']
    if 'learning_rate' in preset_config:
        EXT_LGBM_PARAMS['learning_rate'] = preset_config['learning_rate']
    if 'lambda_l1' in preset_config:
        EXT_LGBM_PARAMS['lambda_l1'] = preset_config['lambda_l1']
    if 'lambda_l2' in preset_config:
        EXT_LGBM_PARAMS['lambda_l2'] = preset_config['lambda_l2']
    if 'min_child_samples' in preset_config:
        EXT_LGBM_PARAMS['min_child_samples'] = preset_config['min_child_samples']
    
    # Обновляем cutoff из пресета
    if 'train_cutoff' in preset_config:
        EXT_CUTOFF = preset_config['train_cutoff']
    
    print(f"✅ Загружена внешняя конфигурация из config/training_config.py")
    print(f"📌 Активный пресет: {ACTIVE_PRESET}")
    print(f"📅 Train cutoff: {EXT_CUTOFF}")
except ImportError as e:
    USE_EXTERNAL_CONFIG = False
    EXT_LGBM_PARAMS = None
    print(f"⚠️ Используется встроенная конфигурация (ошибка импорта: {e})")


class Config:
    """
    Конфигурация обучения модели.
    
    💡 Для экспериментов редактируйте: config/training_config.py
    """
    
    # Пути
    ML_ROOT = Path(__file__).parent.parent
    DATA_DIR = ML_ROOT / "data" / "processed_ml"
    OUTPUT_MODEL_DIR = ML_ROOT / "data" / "models"
    REPORTS_DIR = ML_ROOT / "reports"
    
    # === ПАРАМЕТРЫ ИЗ ВНЕШНЕЙ КОНФИГУРАЦИИ (или дефолтные) ===
    
    # Временной split
    TRAIN_CUTOFF_DATE = EXT_CUTOFF if USE_EXTERNAL_CONFIG else '2024-01-01'
    
    # Квантили для прогноза (границы 1-sigma и медиана)
    QUANTILES = EXT_QUANTILES if USE_EXTERNAL_CONFIG else [0.16, 0.50, 0.84]
    
    # Целевая переменная (будущая реализованная волатильность)
    TARGET_HORIZON = EXT_HORIZON if USE_EXTERNAL_CONFIG else 5
    TARGET_COL = EXT_TARGET if USE_EXTERNAL_CONFIG else 'target_vol_5d'
    
    # Категориальные признаки
    CATEGORICAL_FEATURES = EXT_CAT_FEATURES if USE_EXTERNAL_CONFIG else [
        'ticker_id', 
        'sector_id',
        'is_month_end',
        'is_month_start',
        'day_of_week',
        'vp_above_va',
        'volume_spike',
        'trend_signal',
        'price_position_ma'
    ]
    
    # Исключаемые тикеры
    EXCLUDE_TICKERS = EXT_EXCLUDE_TICKERS if USE_EXTERNAL_CONFIG else []
    
    # Столбцы для исключения из признаков
    EXCLUDE_COLS = ['date', TARGET_COL, 'ticker_id', 'sector_id']
    
    # Гиперпараметры LightGBM
    LGBM_PARAMS = EXT_LGBM_PARAMS if USE_EXTERNAL_CONFIG else {
        'boosting_type': 'gbdt',
        'objective': 'quantile',
        'metric': 'quantile',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    # Параметры обучения
    NUM_BOOST_ROUND = EXT_NUM_ROUNDS if USE_EXTERNAL_CONFIG else 1000
    EARLY_STOPPING_ROUNDS = EXT_EARLY_STOP if USE_EXTERNAL_CONFIG else 50


# ============================================================================
# 1. ЗАГРУЗКА И АГРЕГАЦИЯ ДАННЫХ
# ============================================================================

def load_all_ticker_data(data_dir: Path) -> pd.DataFrame:
    """
    Загружает все файлы *_ml_features.parquet и объединяет в один DataFrame.
    
    Оптимизации:
    - Downcast float64 -> float32 для экономии памяти
    - Конвертация ticker_id и sector_id в category
    
    Args:
        data_dir: Директория с parquet файлами
        
    Returns:
        pd.DataFrame: Объединённый датасет всех тикеров
    """
    print("=" * 60)
    print("📥 ЭТАП 1: ЗАГРУЗКА И АГРЕГАЦИЯ ДАННЫХ")
    print("=" * 60)
    
    # Находим все файлы
    files = list(data_dir.glob("*_ml_features.parquet"))
    
    if not files:
        raise FileNotFoundError(f"Не найдены файлы в {data_dir}")
    
    print(f"📁 Найдено файлов: {len(files)}")
    
    # Загружаем метаданные тикеров для liquidity_rank
    metadata_path = data_dir.parent.parent / "config" / "tickers_metadata.json"
    ticker_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            ticker_metadata = json.load(f)
        print(f"📋 Загружены метаданные из {metadata_path.name}")
    
    # Загружаем все файлы (с фильтрацией исключённых тикеров)
    dfs = []
    excluded_count = 0
    
    for f in files:
        ticker = f.stem.replace('_ml_features', '')
        
        # Пропускаем исключённые тикеры
        if ticker in Config.EXCLUDE_TICKERS:
            print(f"   ⏭️ {ticker}: ИСКЛЮЧЁН из обучения")
            excluded_count += 1
            continue
        
        df = pd.read_parquet(f)
        
        # Добавляем liquidity_rank из метаданных, если его нет в данных
        if 'liquidity_rank' not in df.columns and ticker in ticker_metadata:
            df['liquidity_rank'] = ticker_metadata[ticker].get('liquidity_rank', 30)
        
        print(f"   • {ticker}: {len(df)} строк, {len(df.columns)} столбцов")
        dfs.append(df)
    
    if excluded_count:
        print(f"\n⚠️ Исключено тикеров: {excluded_count}")
    
    # Объединяем в один DataFrame
    global_df = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Объединённый датасет: {len(global_df):,} строк")
    
    # Память до оптимизации
    mem_before = global_df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"💾 Память до оптимизации: {mem_before:.1f} MB")
    
    # === ОПТИМИЗАЦИЯ ПАМЯТИ ===
    
    # Downcast float64 -> float32
    float_cols = global_df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        global_df[col] = global_df[col].astype('float32')
    
    # Конвертация категориальных признаков
    if 'ticker_id' in global_df.columns:
        global_df['ticker_id'] = global_df['ticker_id'].astype('category')
    
    if 'sector_id' in global_df.columns:
        global_df['sector_id'] = global_df['sector_id'].astype('category')
    
    # Память после оптимизации
    mem_after = global_df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"💾 Память после оптимизации: {mem_after:.1f} MB ({(1 - mem_after/mem_before)*100:.1f}% экономия)")
    
    # Очистка памяти
    del dfs
    gc.collect()
    
    return global_df


def create_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Создаёт целевую переменную: реализованную волатильность на горизонте.
    
    Формула: rolling_std(log_return, horizon).shift(-horizon) * sqrt(252)
    
    КРИТИЧНО: shift(-horizon) смещает на будущее для предсказания!
    
    Args:
        df: DataFrame с log_return
        horizon: Горизонт прогноза в днях
        
    Returns:
        DataFrame с добавленной целевой переменной
    """
    print(f"\n🎯 Создание целевой переменной (горизонт: {horizon} дней)...")
    
    # Группируем по тикеру для корректного расчёта
    df = df.sort_values(['ticker_id', 'date'])
    
    def calc_target(group):
        """Расчёт target для одного тикера."""
        future_vol = (
            group['log_return']
            .rolling(window=horizon)
            .std()
            .shift(-horizon)  # Смещаем на будущее!
            * np.sqrt(252)    # Аннуализация
        )
        return future_vol
    
    df[Config.TARGET_COL] = df.groupby('ticker_id', observed=True).apply(
        lambda x: calc_target(x)
    ).reset_index(level=0, drop=True)
    
    print(f"   ✅ Целевая переменная создана: {Config.TARGET_COL}")
    print(f"   📊 Статистика target: mean={df[Config.TARGET_COL].mean():.4f}, "
          f"std={df[Config.TARGET_COL].std():.4f}")
    
    return df


# ============================================================================
# 2. ВРЕМЕННОЙ TRAIN/TEST SPLIT
# ============================================================================

def time_series_split(
    df: pd.DataFrame, 
    cutoff_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строгий временной split БЕЗ shuffle.
    
    КРИТИЧНО: Нельзя использовать random split - это приведёт к look-ahead bias!
    
    Args:
        df: Полный датасет
        cutoff_date: Дата разделения (формат 'YYYY-MM-DD')
        
    Returns:
        Tuple[train_df, test_df]
    """
    print("\n" + "=" * 60)
    print("✂️ ЭТАП 2: ВРЕМЕННОЙ SPLIT (Time Series)")
    print("=" * 60)
    
    # Конвертируем дату
    df['date'] = pd.to_datetime(df['date'])
    cutoff = pd.Timestamp(cutoff_date)
    
    # Split по дате
    train_df = df[df['date'] < cutoff].copy()
    test_df = df[df['date'] >= cutoff].copy()
    
    print(f"📅 Cutoff дата: {cutoff_date}")
    print(f"📈 Train: {len(train_df):,} строк ({train_df['date'].min().date()} - {train_df['date'].max().date()})")
    print(f"📉 Test:  {len(test_df):,} строк ({test_df['date'].min().date()} - {test_df['date'].max().date()})")
    print(f"📊 Соотношение: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% / "
          f"{len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")
    
    # Проверка на уникальные тикеры
    train_tickers = set(train_df['ticker_id'].unique())
    test_tickers = set(test_df['ticker_id'].unique())
    print(f"🏷️ Тикеров в train: {len(train_tickers)}, в test: {len(test_tickers)}")
    
    if test_tickers - train_tickers:
        print(f"⚠️ Тикеры только в test: {test_tickers - train_tickers}")
    
    return train_df, test_df


# ============================================================================
# 3. SAMPLE WEIGHTING (по ликвидности)
# ============================================================================

def create_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Создаёт веса сэмплов на основе ликвидности.
    
    Формула: weight = 1 / log(liquidity_rank + 1)
    
    Более ликвидные активы (меньший rank) получают больший вес,
    чтобы модель приоритезировала качественные данные.
    
    Args:
        df: DataFrame с колонкой liquidity_rank
        
    Returns:
        np.ndarray с весами
    """
    if 'liquidity_rank' not in df.columns or df['liquidity_rank'].isna().all():
        print("⚠️ liquidity_rank не найден или пуст, используем равные веса")
        return np.ones(len(df))
    
    # Заполняем NaN медианным значением
    liquidity = df['liquidity_rank'].fillna(df['liquidity_rank'].median()).values
    
    # Формула весов: более ликвидные (rank ближе к 1) получают больший вес
    weights = 1.0 / np.log(liquidity + 2)  # +2 чтобы избежать log(1)=0
    
    # Нормализация для стабильности
    weights = weights / weights.mean()
    
    print(f"⚖️ Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    return weights


# ============================================================================
# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ LIGHTGBM
# ============================================================================

def prepare_lgbm_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    categorical_features: List[str]
) -> Dict:
    """
    Подготавливает данные для обучения LightGBM.
    
    Args:
        train_df: Тренировочный датасет
        test_df: Тестовый датасет
        target_col: Название целевой колонки
        categorical_features: Список категориальных признаков
        
    Returns:
        Dict с X_train, y_train, X_test, y_test, feature_names, cat_features
    """
    print("\n" + "=" * 60)
    print("🔧 ЭТАП 3: ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)
    
    # Удаляем строки с NaN в target
    train_df = train_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])
    
    print(f"📊 После удаления NaN: train={len(train_df):,}, test={len(test_df):,}")
    
    # Определяем признаки (исключаем служебные столбцы)
    # Базовые исключения
    exclude_cols = ['date', target_col]
    
    # Если ticker_id не в категориальных признаках, исключаем его из фичей
    # (чтобы модель не использовала его для прогноза)
    if 'ticker_id' not in categorical_features:
        exclude_cols.append('ticker_id')
        print(f"   ⚠️ ticker_id исключён из фичей (эксперимент без запоминания тикеров)")
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Фильтруем категориальные признаки (только те, что есть в данных)
    cat_features = [col for col in categorical_features if col in feature_cols]
    
    print(f"📋 Всего признаков: {len(feature_cols)}")
    print(f"📋 Категориальных: {len(cat_features)}")
    print(f"   {cat_features}")
    
    # Подготовка X и y
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].values
    
    # Конвертируем категориальные в коды для LightGBM
    for col in cat_features:
        if col in X_train.columns:
            # Если не category - конвертируем
            if X_train[col].dtype.name != 'category':
                X_train[col] = X_train[col].astype('category')
            if X_test[col].dtype.name != 'category':
                X_test[col] = X_test[col].astype('category')
    
    # Sample weights для train
    sample_weights = create_sample_weights(train_df)
    
    # Заполняем NaN только в числовых колонках (категориальные оставляем как есть)
    # LightGBM умеет работать с NaN в категориальных признаках
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(0)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_cols,
        'cat_features': cat_features,
        'sample_weights': sample_weights
    }


# ============================================================================
# 5. ОБУЧЕНИЕ КВАНТИЛЬНЫХ МОДЕЛЕЙ
# ============================================================================

def train_quantile_models(
    data: Dict,
    quantiles: List[float],
    params: Dict,
    num_boost_round: int,
    early_stopping_rounds: int
) -> Dict[float, lgb.Booster]:
    """
    Обучает отдельную модель для каждого квантиля.
    
    Args:
        data: Dict с X_train, y_train, X_test, y_test
        quantiles: Список квантилей [0.16, 0.50, 0.84]
        params: Гиперпараметры LightGBM
        num_boost_round: Количество итераций
        early_stopping_rounds: Early stopping
        
    Returns:
        Dict[alpha, lgb.Booster]: Словарь обученных моделей
    """
    print("\n" + "=" * 60)
    print("🚀 ЭТАП 4: ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Создаём LightGBM datasets
    train_data = lgb.Dataset(
        data['X_train'],
        label=data['y_train'],
        weight=data['sample_weights'],
        categorical_feature=data['cat_features'],
        free_raw_data=False
    )
    
    valid_data = lgb.Dataset(
        data['X_test'],
        label=data['y_test'],
        categorical_feature=data['cat_features'],
        reference=train_data,
        free_raw_data=False
    )
    
    models = {}
    
    for alpha in quantiles:
        print(f"\n🎯 Обучение модели для квантиля α={alpha}...")
        
        # Копируем параметры и устанавливаем квантиль
        model_params = params.copy()
        model_params['alpha'] = alpha
        
        # Обучение с early stopping
        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        models[alpha] = model
        
        # Метрики
        train_pred = model.predict(data['X_train'])
        test_pred = model.predict(data['X_test'])
        
        train_loss = quantile_loss(data['y_train'], train_pred, alpha)
        test_loss = quantile_loss(data['y_test'], test_pred, alpha)
        
        print(f"   📈 Train Quantile Loss: {train_loss:.6f}")
        print(f"   📉 Test Quantile Loss:  {test_loss:.6f}")
        print(f"   🌳 Best iteration: {model.best_iteration}")
    
    return models


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Вычисляет Quantile Loss (Pinball Loss).
    
    Формула: mean(max(alpha*(y-pred), (alpha-1)*(y-pred)))
    """
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)
    return np.mean(loss)


# ============================================================================
# 6. СОХРАНЕНИЕ МОДЕЛЕЙ
# ============================================================================

def save_models(
    models: Dict[float, lgb.Booster],
    output_dir: Path
) -> None:
    """
    Сохраняет обученные модели в файлы.
    
    Args:
        models: Словарь {alpha: model}
        output_dir: Директория для сохранения
    """
    print("\n" + "=" * 60)
    print("💾 ЭТАП 5: СОХРАНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for alpha, model in models.items():
        # Формат: global_lgbm_q16.txt, global_lgbm_q50.txt, global_lgbm_q84.txt
        filename = f"global_lgbm_q{int(alpha*100)}.txt"
        path = output_dir / filename
        model.save_model(str(path))
        print(f"   ✅ Сохранено: {path}")


# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(
    models: Dict[float, lgb.Booster],
    output_dir: Path,
    top_n: int = 30
) -> None:
    """
    Строит и сохраняет график важности признаков.
    
    Args:
        models: Словарь обученных моделей
        output_dir: Директория для сохранения
        top_n: Количество топ признаков для отображения
    """
    print("\n" + "=" * 60)
    print("📊 ЭТАП 6: FEATURE IMPORTANCE")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Используем медианную модель (q50) для importance
    model = models[0.50]
    
    # Получаем importance (по gain)
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    
    # Создаём DataFrame и сортируем
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Топ-{top_n} признаков по важности (gain):")
    print(imp_df.head(top_n).to_string(index=False))
    
    # Сохраняем в CSV
    csv_path = output_dir / "feature_importance.csv"
    imp_df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV сохранён: {csv_path}")
    
    # Строим график
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_features = imp_df.head(top_n)
    y_pos = np.arange(len(top_features))
    
    bars = ax.barh(y_pos, top_features['importance'], color='steelblue', edgecolor='navy')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # Топ сверху
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('LightGBM Global Model - Feature Importance', fontsize=14, fontweight='bold')
    
    # Добавляем значения на барах
    for bar, val in zip(bars, top_features['importance']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Сохраняем график
    plot_path = output_dir / "feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 График сохранён: {plot_path}")


# ============================================================================
# 8. ВАЛИДАЦИЯ И ОТЧЁТ
# ============================================================================

def generate_validation_report(
    models: Dict[float, lgb.Booster],
    data: Dict,
    output_dir: Path
) -> None:
    """
    Генерирует отчёт о валидации модели.
    
    Args:
        models: Обученные модели
        data: Данные для валидации
        output_dir: Директория для отчёта
    """
    print("\n" + "=" * 60)
    print("📋 ОТЧЁТ О ВАЛИДАЦИИ")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Прогнозы для каждого квантиля
    predictions = {}
    for alpha, model in models.items():
        predictions[alpha] = model.predict(data['X_test'])
    
    y_true = data['y_test']
    
    # Проверка покрытия интервала [q16, q84]
    lower = predictions[0.16]
    upper = predictions[0.84]
    
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    expected_coverage = 0.84 - 0.16  # 68%
    
    print(f"\n🎯 Покрытие интервала [q16, q84]:")
    print(f"   Ожидаемое: {expected_coverage*100:.1f}%")
    print(f"   Фактическое: {coverage*100:.1f}%")
    
    # Средняя ширина интервала
    interval_width = np.mean(upper - lower)
    print(f"\n📏 Средняя ширина интервала: {interval_width:.4f}")
    
    # Quantile losses
    print(f"\n📊 Quantile Loss на тесте:")
    for alpha in [0.16, 0.50, 0.84]:
        loss = quantile_loss(y_true, predictions[alpha], alpha)
        print(f"   α={alpha}: {loss:.6f}")
    
    # MAE для медианы
    mae = np.mean(np.abs(y_true - predictions[0.50]))
    print(f"\n📐 MAE (медиана q50): {mae:.6f}")
    
    # Сохраняем отчёт
    report = {
        'cutoff_date': Config.TRAIN_CUTOFF_DATE,
        'train_samples': len(data['y_train']),
        'test_samples': len(data['y_test']),
        'coverage_68': coverage,
        'interval_width': interval_width,
        'mae_median': mae,
        'quantile_loss_16': quantile_loss(y_true, predictions[0.16], 0.16),
        'quantile_loss_50': quantile_loss(y_true, predictions[0.50], 0.50),
        'quantile_loss_84': quantile_loss(y_true, predictions[0.84], 0.84),
        'timestamp': datetime.now().isoformat()
    }
    
    report_df = pd.DataFrame([report])
    report_path = output_dir / "validation_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n💾 Отчёт сохранён: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Главная функция обучения."""
    
    print("\n" + "=" * 60)
    print("🚀 GLOBAL LIGHTGBM MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"📅 Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Целевая переменная: {Config.TARGET_COL}")
    print(f"📊 Квантили: {Config.QUANTILES}")
    
    # 1. Загрузка данных
    global_df = load_all_ticker_data(Config.DATA_DIR)
    
    # 2. Создание целевой переменной
    global_df = create_target_variable(global_df, horizon=Config.TARGET_HORIZON)
    
    # 3. Временной split
    train_df, test_df = time_series_split(global_df, Config.TRAIN_CUTOFF_DATE)
    
    # Освобождаем память
    del global_df
    gc.collect()
    
    # 4. Подготовка данных для LightGBM
    data = prepare_lgbm_data(
        train_df, 
        test_df, 
        Config.TARGET_COL,
        Config.CATEGORICAL_FEATURES
    )
    
    # Освобождаем память
    del train_df, test_df
    gc.collect()
    
    # 5. Обучение моделей
    models = train_quantile_models(
        data,
        Config.QUANTILES,
        Config.LGBM_PARAMS,
        Config.NUM_BOOST_ROUND,
        Config.EARLY_STOPPING_ROUNDS
    )
    
    # 6. Сохранение моделей
    save_models(models, Config.OUTPUT_MODEL_DIR)
    
    # 7. Feature Importance
    plot_feature_importance(models, Config.REPORTS_DIR, top_n=30)
    
    # 8. Валидационный отчёт
    generate_validation_report(models, data, Config.REPORTS_DIR)
    
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"\n📁 Модели сохранены в: {Config.OUTPUT_MODEL_DIR}")
    print(f"📁 Отчёты сохранены в: {Config.REPORTS_DIR}")


if __name__ == "__main__":
    main()

