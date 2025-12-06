"""
🎛️ КОНФИГУРАЦИЯ ОБУЧЕНИЯ МОДЕЛИ

Этот файл содержит ВСЕ параметры, которые можно менять для экспериментов.
Измените нужные параметры и запустите: python run_full_pipeline.py --skip-features

Автор: ML Pipeline v2.0
"""

from datetime import datetime, timedelta

# ============================================================================
# 📅 TRAIN/TEST SPLIT
# ============================================================================

# Вариант 1: Фиксированная дата
# TRAIN_CUTOFF_DATE = '2024-01-01'  # ~60/40 split

# Вариант 2: Динамическая дата (последние N дней = test)
# TEST_DAYS = 365  # Последний год = test

# Вариант 3: Процентное соотношение (рекомендуется)
TRAIN_TEST_RATIO = 0.70  # 70% train, 30% test (было 60/40)

# Текущая настройка:
TRAIN_CUTOFF_DATE = '2024-06-01'  # Сдвинули на 6 месяцев вперёд → больше train данных


# ============================================================================
# 🎯 ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ============================================================================

TARGET_HORIZON = 5  # Горизонт прогноза в днях (1, 5, 10, 20)
TARGET_COL = f'target_vol_{TARGET_HORIZON}d'


# ============================================================================
# 🌳 ГИПЕРПАРАМЕТРЫ LIGHTGBM
# ============================================================================

LGBM_PARAMS = {
    # Архитектура
    'boosting_type': 'gbdt',
    'objective': 'quantile',
    'metric': 'quantile',
    
    # Сложность модели
    'num_leaves': 63,          # Больше = сложнее (31, 63, 127)
    'max_depth': -1,           # -1 = без ограничения (или 6, 8, 10)
    'min_child_samples': 20,   # Мин. записей в листе (10, 20, 50, 100)
    
    # Скорость обучения
    'learning_rate': 0.05,     # Меньше = медленнее, но стабильнее (0.01, 0.05, 0.1)
    
    # Регуляризация (ПРОТИВ ПЕРЕОБУЧЕНИЯ)
    'lambda_l1': 0.1,          # L1 reg (0, 0.1, 0.5, 1.0)
    'lambda_l2': 0.1,          # L2 reg (0, 0.1, 0.5, 1.0)
    'feature_fraction': 0.8,   # Доля признаков для каждого дерева (0.6, 0.8, 1.0)
    'bagging_fraction': 0.8,   # Доля строк для каждого дерева (0.6, 0.8, 1.0)
    'bagging_freq': 5,         # Частота bagging (1, 5, 10)
    
    # Прочее
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# Параметры обучения
NUM_BOOST_ROUND = 1000        # Макс. итераций (500, 1000, 2000)
EARLY_STOPPING_ROUNDS = 50    # Остановка если нет улучшения (30, 50, 100)


# ============================================================================
# 📊 КВАНТИЛИ
# ============================================================================

# Стандартный 68% интервал (±1 sigma)
QUANTILES = [0.16, 0.50, 0.84]

# Альтернатива: 80% интервал
# QUANTILES = [0.10, 0.50, 0.90]

# Альтернатива: 95% интервал
# QUANTILES = [0.025, 0.50, 0.975]


# ============================================================================
# 🏷️ КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
# ============================================================================

# БАЗОВЫЙ СПИСОК (с ticker_id)
CATEGORICAL_FEATURES_BASE = [
    'ticker_id',       # ID тикера (может вызывать переобучение!)
    'sector_id',       # ID сектора
    'is_month_end',
    'is_month_start',
    'day_of_week',
    'vp_above_va',
    'volume_spike',
    'trend_signal',
    'price_position_ma'
]

# ЭКСПЕРИМЕНТ: без ticker_id (для лучшей генерализации)
CATEGORICAL_FEATURES_NO_TICKER = [
    'sector_id',       # ID сектора (остаётся - это обобщённая информация)
    'is_month_end',
    'is_month_start',
    'day_of_week',
    'vp_above_va',
    'volume_spike',
    'trend_signal',
    'price_position_ma'
]

# АКТИВНЫЙ СПИСОК (выберите один из вариантов выше)
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_NO_TICKER  # Эксперимент без ticker_id
# CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_BASE  # С ticker_id (базовый вариант)


# ============================================================================
# 🚫 ИСКЛЮЧАЕМЫЕ ТИКЕРЫ (плохо работают)
# ============================================================================

EXCLUDE_TICKERS = [
    'FIVE',   # r=0.077 - очень плохо
    'BELU',   # r=0.241 - плохо
    'YNDX',   # r=0.343 - слабо
    'LENT',   # r=0.427 - слабо
]


# ============================================================================
# 🧪 PRESET-Ы ДЛЯ ЭКСПЕРИМЕНТОВ
# ============================================================================

# Пресет 1: Базовый (текущий)
PRESET_BASELINE = {
    'train_cutoff': '2024-01-01',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_child_samples': 20,
}

# Пресет 2: Больше train данных (70/30)
PRESET_MORE_TRAIN = {
    'train_cutoff': '2024-06-01',  # +6 месяцев train
    'num_leaves': 63,
    'learning_rate': 0.05,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_child_samples': 20,
}

# Пресет 3: Сильная регуляризация (против переобучения)
PRESET_REGULARIZED = {
    'train_cutoff': '2024-06-01',
    'num_leaves': 31,             # Меньше
    'learning_rate': 0.03,        # Медленнее
    'lambda_l1': 0.5,             # Сильнее
    'lambda_l2': 1.0,             # Сильнее
    'min_child_samples': 50,      # Больше
}

# Пресет 4: Без ticker_id (чистая генерализация)
PRESET_NO_TICKER = {
    'train_cutoff': '2024-06-01',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'lambda_l1': 0.3,
    'lambda_l2': 0.3,
    'min_child_samples': 30,
    'exclude_features': ['ticker_id'],
}


# ============================================================================
# 📝 АКТИВНЫЙ ПРЕСЕТ (измените здесь!)
# ============================================================================

ACTIVE_PRESET = 'MORE_TRAIN'  # Выберите: BASELINE, MORE_TRAIN, REGULARIZED, NO_TICKER


def get_active_config():
    """Возвращает активную конфигурацию."""
    presets = {
        'BASELINE': PRESET_BASELINE,
        'MORE_TRAIN': PRESET_MORE_TRAIN,
        'REGULARIZED': PRESET_REGULARIZED,
        'NO_TICKER': PRESET_NO_TICKER,
    }
    return presets.get(ACTIVE_PRESET, PRESET_BASELINE)


# ============================================================================
# 🔍 ВЫВОД ТЕКУЩЕЙ КОНФИГУРАЦИИ
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎛️ ТЕКУЩАЯ КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
    print("=" * 60)
    
    config = get_active_config()
    print(f"\n📌 Активный пресет: {ACTIVE_PRESET}")
    print(f"\n📅 Train/Test Split:")
    print(f"   Cutoff дата: {config.get('train_cutoff', TRAIN_CUTOFF_DATE)}")
    
    print(f"\n🌳 LightGBM параметры:")
    for key in ['num_leaves', 'learning_rate', 'lambda_l1', 'lambda_l2', 'min_child_samples']:
        value = config.get(key, LGBM_PARAMS.get(key))
        print(f"   {key}: {value}")
    
    print(f"\n🎯 Target: {TARGET_COL}")
    print(f"📊 Quantiles: {QUANTILES}")
    
    if EXCLUDE_TICKERS:
        print(f"\n🚫 Исключённые тикеры: {EXCLUDE_TICKERS}")
    
    print("\n" + "=" * 60)

