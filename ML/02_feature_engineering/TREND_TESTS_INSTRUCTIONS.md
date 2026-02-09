# Инструкция: Тесты для трендовых признаков (Trend Features)

## 📋 Созданные тесты

Файл `trend_features_tests.py` содержит **четыре комплексных теста** для валидации трендовых признаков.

---

## ✅ Тест 1: Стационарность (Stationarity Check)

### Цель
Проверить, "плывет" ли распределение `dist_to_sma_200` во времени.

### Математическая основа
Признаки вида `close/SMA - 1` должны быть **стационарными**:
- Среднее ≈ 0 (цена колеблется вокруг скользящей средней)
- Дисперсия стабильна во времени
- Нет долгосрочного тренда (дрейфа)

### Проблема нестационарности
Если признак **нестационарен**:
- Модель, обученная на 2020-2021, не будет работать на 2023-2024
- Значения признака могут уходить далеко от 0 (например, постоянно \u003e 0.2)
- LightGBM переобучится на исторические паттерны, которые не повторятся

### Что делает тест

1. **Временной ряд**: Визуализация `dist_to_sma_200` с rolling mean (60d)
2. **Распределение до/после кризиса 2022**: Сравнение гистограмм
3. **Q-Q Plot**: Проверка нормальности распределения
4. **Статистические тесты**:
   - **ADF (Augmented Dickey-Fuller)**: Тест на стационарность
   - **T-test**: Сравнение средних до/после кризиса
   - **Levene test**: Сравнение дисперсий до/после

### Интерпретация результатов

```
📊 Результаты теста на стационарность:
=============================================================================
Тикер  Mean до  Mean после  Std до  Std после  ADF p-value  T-test p  Стационарен?
SBER    0.012      0.015     0.082     0.095       0.001       0.45        Да
GAZP   -0.025     -0.018     0.105     0.122       0.032       0.23        Да
```

| ADF p-value | Интерпретация | Действие |
|-------------|---------------|----------|
| **< 0.05** | Ряд стационарен ✅ | Признак пригоден для использования |
| **> 0.05** | Ряд нестационарен ❌ | Требуется коррекция (см. ниже) |

| T-test p | Интерпретация | Значение |
|----------|---------------|----------|
| **> 0.05** | Средние до/после равны ✅ | Признак стабилен во времени |
| **< 0.05** | Средние изменились ⚠️ | Режим сменился (возможен structural break) |

### Решения для нестационарных признаков

#### 1. Увеличить окно MA
```python
# Вместо sma_200 использовать sma_300 или sma_400
result['dist_to_sma_300'] = (close / sma(close, window=300)) - 1
```

**Почему это помогает:**
- Более длинное окно → более стабильная база для нормализации
- Меньше "дрейфует" при смене режима рынка

#### 2. Differencing (первая разность)
```python
# Вместо уровня использовать изменение признака
result['dist_to_sma_200_diff'] = result['dist_to_sma_200'].diff()
```

**Почему это помогает:**
- Убирает долгосрочный тренд
- Делает ряд стационарным по построению

#### 3. Использовать percentile-based нормализацию
```python
# Вместо абсолютного расстояния — ранг в историческом распределении
from scipy.stats import percentileofscore
result['dist_to_sma_200_pct'] = df['dist_to_sma_200'].rolling(window=252).apply(
    lambda x: percentileofscore(x, x.iloc[-1])
)
```

---

## ✅ Тест 2: «Страх падения» vs «Эйфория роста»

### Цель
Проверить **асимметрию волатильности** на uptrend vs downtrend.

### Гипотеза
```
Vol(Downtrend) >> Vol(Uptrend)
```

**Почему:**
- Падения резкие (panic selling, margin calls)
- Рост медленный (накопление, FOMO)
- "Markets take the stairs up and the elevator down"

### Что делает тест

1. **Распределение волатильности**: Histogram для uptrend/downtrend/sideways
2. **Boxplot**: Сравнение медиан и квартилей
3. **Статистические тесты**:
   - **T-test**: Параметрический тест (предполагает нормальность)
   - **Mann-Whitney U**: Непараметрический (устойчив к outliers)
4. **Ratio Down/Up**: Во сколько раз волатильность на даунтренде выше

### Интерпретация результатов

```
📊 Результаты теста «Страх падения» vs «Эйфория роста»:
=============================================================================
Тикер  Vol Uptrend  Vol Downtrend  Vol Sideways  Ratio Down/Up  T-test p  Асимметричен?
SBER       0.28         0.42           0.31          1.50        0.001         Да
GAZP       0.33         0.58           0.38          1.76        0.000         Да
```

| Ratio Down/Up | Интерпретация | Критичность |
|---------------|---------------|-------------|
| **> 1.5** | Сильная асимметрия ⚠️ | Высокая |
| **1.2-1.5** | Умеренная асимметрия | Средняя |
| **< 1.2** | Слабая асимметрия | Низкая |

### Проблема для модели

Если `trend_signal` и `rsi_14` не подсвечивают эту асимметрию:
- Модель **занижает прогноз** волатильности при обвалах
- Backtest показывает хорошие результаты, но **на продакшне fails**

### Решения

#### 1. Добавить кросс-фичу
```python
# В trend_features.py
result['trend_vol_interaction'] = result['trend_signal'] * df['realized_vol_5d']
```

**Эффект:**
- LightGBM явно видит: "на даунтренде волатильность выше"
- Feature importance этой фичи будет высоким

#### 2. Weighted RSI
```python
# RSI с весами для падений
def weighted_rsi(prices, window=14, down_weight=1.5):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta * down_weight).where(delta < 0, 0.0)  # Усиливаем падения
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
```

#### 3. Добавить флаг "extreme downtrend"
```python
# Флаг для сильного падения
result['extreme_downtrend'] = (
    (result['trend_signal'] == -1) & 
    (result['momentum_10'] < -0.05)
).astype(int)
```

---

## ✅ Тест 3: Мультиколлинеарность MA Distances

### Цель
Проверить, дублируют ли `dist_to_sma_20` и `dist_to_ema_20` друг друга.

### Проблема мультиколлинеарности

Если корреляция между признаками **r > 0.95**:
- Оба несут одинаковую информацию
- Feature importance размывается между ними
- Модель становится менее интерпретируемой
- Увеличивается риск overfitting

### Что делает тест

1. **Correlation heatmap**: Корреляционная матрица всех MA distances
2. **High correlation pairs**: Список пар с r > 0.9
3. **Barplot**: Визуализация высоких корреляций

### Интерпретация результатов

```
⚠️  Высокая корреляция (|r| > 0.9) между признаками:
=============================================================================
Feature 1         Feature 2         Correlation
dist_to_sma_20    dist_to_ema_20         0.97
dist_to_sma_50    dist_to_ema_50         0.96
```

| Correlation (r) | Действие | Приоритет |
|-----------------|----------|-----------|
| **> 0.95** | ❌ Удалить один признак | Высокий |
| **0.9-0.95** | ⚠️ Желательно удалить | Средний |
| **< 0.9** | ✅ Оставить оба | - |

### Что удалять: SMA или EMA?

**Рекомендация: Оставить EMA, удалить SMA**

**Почему:**
1. **EMA быстрее реагирует** на изменения цены (больше веса на свежих данных)
2. **SMA лаггирует** (все данные одинаковый вес)
3. Для волатильности важна **скорость** реакции на изменения

### Практическая реализация

```python
# В trend_features.py
TREND_FEATURE_COLUMNS = [
    # Удаляем SMA distances (дублируют EMA)
    # 'dist_to_sma_20',   # Удалено
    # 'dist_to_sma_50',   # Удалено
    'dist_to_sma_200',    # Оставляем (нет EMA_200 в текущей реализации)
    
    # Оставляем EMA distances
    'dist_to_ema_20',
    'dist_to_ema_50',
    
    # ... остальные признаки ...
]
```

**Ожидаемый эффект:**
- ✅ Меньше признаков → быстрее тренировка
- ✅ Более чёткая feature importance
- ✅ Меньше риск overfitting

---

## ✅ Тест 4: Информативность RSI и Momentum

### Цель
Проверить, есть ли **U-образная зависимость** между RSI и волатильностью.

### Гипотеза
```
Vol(RSI < 30) ≈ Vol(RSI > 70) >> Vol(30 < RSI < 70)
```

**Почему:**
- **RSI < 30**: Oversold → паника, резкие падения
- **RSI > 70**: Overbought → эйфория, возможна коррекция
- **RSI 30-70**: Нейтральная зона → спокойный рынок

### Что делает тест

1. **Scatter plot**: RSI vs Realized Volatility
2. **Binned mean**: Условное среднее волатильности по бинам RSI
3. **Статистика по зонам**: Oversold, Neutral, Overbought
4. **Корреляционные тесты**:
   - **Pearson**: Линейная корреляция (должна быть близка к 0)
   - **Spearman**: Монотонная корреляция
   - **U-shape test**: Корреляция `|RSI - 50|` vs `Vol` (должна быть > 0)

### Интерпретация результатов

```
📊 Результаты теста информативности RSI:
=============================================================================
Тикер  Vol Oversold  Vol Neutral  Vol Overbought  Pearson r  U-shape r  Информативен?
SBER       0.42         0.28           0.39         -0.05       0.25          Да
GAZP       0.38         0.31           0.35         -0.02       0.18          Да
```

| U-shape r | U-shape p | Интерпретация | Действие |
|-----------|-----------|---------------|----------|
| **> 0.2** | **< 0.05** | RSI информативен ✅ | Оставить в модели |
| **0.1-0.2** | **< 0.05** | Слабо информативен | Протестировать на backtest |
| **< 0.1** | **> 0.05** | Не информативен ❌ | Удалить или заменить |

### Что делать, если RSI неинформативен

#### 1. Удалить RSI
```python
# В TREND_FEATURE_COLUMNS
# 'rsi_14',  # Удалено (неинформативен для волатильности)
```

#### 2. Заменить на Stochastic RSI
```python
def stochastic_rsi(rsi, window=14):
    """Более чувствительный к экстремумам"""
    rsi_min = rsi.rolling(window=window).min()
    rsi_max = rsi.rolling(window=window).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min)
```

#### 3. Создать "RSI extremes" признак
```python
# Подсвечивает только экстремумы
def rsi_extremes(rsi):
    """Расстояние от нейтральной зоны"""
    return np.maximum(0, 30 - rsi) + np.maximum(0, rsi - 70)

result['rsi_extremes'] = rsi_extremes(result['rsi_14'])
```

**Преимущество:**
- Явно показывает модели: "важны экстремумы, а не середина"
- Линейная зависимость вместо U-образной (проще для LightGBM)

---

## 🚀 Как использовать тесты

### Вариант 1: В Jupyter Notebook

1. Откройте `04_trend_features.ipynb`
2. Создайте новую ячейку **после** расчёта признаков
3. Скопируйте код из `trend_features_tests.py`
4. Запустите ячейку

### Вариант 2: Как отдельный скрипт

```bash
cd /Users/nikitabaslykov/Documents/Python/Trading/MOEXScanner/ML/02_feature_engineering
python trend_features_tests.py
```

---

## 📊 Ожидаемые действия после тестов

### 1. Если признак нестационарен (ADF p > 0.05)

```python
# В trend_features.py
# Увеличить окно или использовать differencing
result['dist_to_sma_300'] = dist_to_ma(close, sma(close, window=300))
# ИЛИ
result['dist_to_sma_200_diff'] = result['dist_to_sma_200'].diff()
```

### 2. Если волатильность асимметрична (Ratio > 1.5)

```python
# Добавить кросс-фичу
result['trend_vol_signal'] = (result['trend_signal'] == -1).astype(int)
# ИЛИ
result['extreme_downtrend'] = (
    (result['trend_signal'] == -1) & 
    (result['momentum_10'] < -0.05)
).astype(int)
```

### 3. Если MA distances коррелируют (r > 0.95)

```python
# Удалить SMA, оставить EMA
TREND_FEATURE_COLUMNS = [
    # 'dist_to_sma_20',   # Удалено
    # 'dist_to_sma_50',   # Удалено
    'dist_to_ema_20',     # Оставлено
    'dist_to_ema_50',     # Оставлено
    'dist_to_sma_200',    # Оставлено (нет EMA альтернативы)
    # ... остальные ...
]
```

### 4. Если RSI неинформативен (U-shape r < 0.1)

```python
# Заменить на RSI extremes
def rsi_extremes(rsi):
    return np.maximum(0, 30 - rsi) + np.maximum(0, rsi - 70)

result['rsi_extremes'] = rsi_extremes(result['rsi_14'])
# И удалить обычный RSI из списка признаков
```

---

## ✅ Checklist

- [ ] Запустить все 4 теста
- [ ] Проверить стационарность (ADF test)
- [ ] Если нестационарен → увеличить окно MA или differencing
- [ ] Проверить асимметрию волатильности
- [ ] Если Ratio > 1.5 → добавить trend_vol_interaction
- [ ] Проверить мультиколлинеарность MA distances
- [ ] Если r > 0.95 → удалить SMA, оставить EMA
- [ ] Проверить информативность RSI
- [ ] Если U-shape r < 0.1 → удалить или заменить на RSI extremes
- [ ] Запустить backtest с обновлёнными признаками

---

---

# 🚀 Продвинутые ML-тесты (Advanced Tests)

Файл: `trend_features_tests_advanced.py`

Эти тесты проверяют **production readiness** трендовых признаков.

---

## ✅ Тест 5: OOS Value Test (Out-of-Sample)

### Цель
Проверить, сохраняют ли трендовые фичи **предсказательную силу на OOS данных**.

### Проблема
**In-sample** корреляция может быть высокой (0.3), но **OOS** — нулевой (0.05).

Это означает **переобучение**: фича полезна на train, но бесполезна на test.

### Что делает тест

1. **Rolling Time Series Split**: 5 фолдов с expanding window
2. **Для каждой фичи**:
   - **IS Correlation**: Корреляция фичи с таргетом на train
   - **OOS Correlation**: Корреляция фичи с таргетом на test
   - **OOS MI** (Mutual Information): Нелинейная связь на test
3. **Correlation Decay**: `IS Corr - OOS Corr`

### Интерпретация

```
📊 OOS Value Test (топ-10 по OOS Correlation):
=============================================================================
Тикер  Feature           IS Corr  OOS Corr  OOS MI   Corr Decay  Stable OOS?
SBER   realized_vol_5d     0.82     0.75     0.12       0.07         Да
GAZP   rsi_14             0.25     0.18     0.08       0.07         Да
LKOH   dist_to_sma_200    0.30     0.08     0.02       0.22         Нет
```

| OOS Corr | OOS MI | Corr Decay | Интерпретация | Действие |
|----------|--------|------------|---------------|----------|
| **> 0.15** | **> 0.05** | **< 0.1** | Стабильная фича ✅ | Оставить |
| **0.05-0.15** | **< 0.05** | **0.1-0.2** | Слабая, но возможно полезна | Проверить в модели |
| **< 0.05** | **< 0.01** | **> 0.2** | Переобучение ❌ | Удалить |

### Почему важен Correlation Decay?

**Decay > 0.2** означает:
- На train фича выглядит отлично (IS Corr = 0.3)
- На test фича бесполезна (OOS Corr = 0.1)
- **Модель переобучается** на эту фичу

### Решение

```python
# Удалить фичи с высоким decay
bad_features = ['dist_to_sma_200', 'dist_to_sma_50']  # Если decay > 0.2

TREND_FEATURE_COLUMNS = [
    f for f in TREND_FEATURE_COLUMNS 
    if f not in bad_features
]
```

---

## ✅ Тест 6: Ablation Test

### Цель
Измерить **вклад** трендовых фич в качество модели.

### Сравнение моделей

1. **Baseline**: Только `volatility_features` (realized_vol, Parkinson, GK)
2. **Full**: `volatility_features` + `trend_features`

### Метрики

- **MAE** (Mean Absolute Error): Основная метрика
- **Pinball Loss**: Quantile regression (α=0.5)
- **Correlation**: Насколько предсказания коррелируют с фактом
- **Coverage**: % предсказаний в пределах ±20% от факта

### Интерпретация

```
📊 Ablation Test Results:
=============================================================================
Тикер  Baseline MAE  Full MAE  MAE Improvement %  Baseline Corr  Full Corr
SBER        0.082      0.074           9.8%             0.61         0.68
GAZP        0.095      0.092           3.2%             0.55         0.57
LKOH        0.088      0.089          -1.1%             0.58         0.56
```

| MAE Improvement | Интерпретация | Действие |
|-----------------|---------------|----------|
| **> 5%** | Трендовые фичи полезны ✅ | Оставить все |
| **2-5%** | Умеренная польза | Оставить, но можно оптимизировать |
| **< 2%** | Не добавляют ценности ❌ | Удалить или упростить |
| **< 0%** (ухудшение) | Вредят модели ❌ | Обязательно удалить |

### Почему важна Coverage?

**Coverage +10%** означает:
- Модель стала точнее в **относительных терминах**
- Меньше экстремальных ошибок (outliers)

**Пример:**
- Baseline: 60% предсказаний в пределах ±20%
- Full: 70% предсказаний в пределах ±20%

→ **На 10% больше точных предсказаний**

---

## ✅ Тест 7: Stability Test (Feature Drift)

### Цель
Проверить **стабильность распределения** фич между train и test.

### Проблема drift

**Сценарий:**
- Train: 2020-2021 (нормальный рынок)
- Test: 2022-2023 (кризис)

**Результат:**
- Распределение фич **изменилось**
- Модель обучена на одном распределении, применяется к другому
- **Провал на продакшне**

### Метрики

#### 1. PSI (Population Stability Index)
```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

**Интерпретация:**
- **PSI < 0.1**: Стабильная фича ✅
- **PSI 0.1-0.2**: Умеренный drift ⚠️
- **PSI > 0.2**: Сильный drift ❌

#### 2. KS (Kolmogorov-Smirnov)
Сравнение двух распределений.

**Интерпретация:**
- **KS p-value > 0.05**: Распределения одинаковые ✅
- **KS p-value < 0.01**: Распределения различны ❌

### Интерпретация

```
📊 Stability Test (топ-10 по PSI):
=============================================================================
Тикер  Feature            PSI    KS Stat  KS p-value  Drift?
SBER   dist_to_sma_200   0.35     0.28      0.001      Да ⚠️
GAZP   rsi_14            0.08     0.12      0.15       Нет ✅
LKOH   momentum_10       0.15     0.18      0.03       Да ⚠️
```

### Решения для drift

#### 1. Удалить нестабильные фичи
```python
# Фичи с PSI > 0.2
unstable_features = ['dist_to_sma_200', 'momentum_20']

TREND_FEATURE_COLUMNS = [
    f for f in TREND_FEATURE_COLUMNS 
    if f not in unstable_features
]
```

#### 2. Ре-нормализация на свежих данных
```python
# Вместо фиксированных параметров
# Пересчитывать на скользящем окне
def adaptive_zscore(values, window=252):
    """Адаптивная нормализация"""
    mean = values.rolling(window).mean()
    std = values.rolling(window).std()
    return (values - mean) / std
```

#### 3. Monitoring в продакшне
```python
# Отслеживать PSI в реальном времени
def monitor_psi(train_dist, prod_values):
    psi = calculate_psi(train_dist, prod_values)
    if psi > 0.2:
        alert("Feature drift detected!")
        retrain_model()
```

---

## ✅ Тест 8: Interaction Check

### Цель
Проверить, добавляют ли **кросс-фичи** (feature interactions) сигнал.

### Тестируемые интеракции

1. **trend_signal × realized_vol**
   - Гипотеза: Волатильность на даунтренде выше, чем на аптренде
   - Кросс-фича подсвечивает эту асимметрию

2. **trend_strength × volume_spike**
   - Гипотеза: Сильный тренд + объёмный всплеск = важный сигнал
   - Обычные ML модели не видят такие комбинации

### Что делает тест

1. **Baseline**: Модель без кросс-фичи
2. **Full**: Модель с кросс-фичей
3. **Сравнение MAE**: Насколько улучшилось качество

### Интерпретация

```
📊 Interaction Check Results:
=============================================================================
Тикер  Interaction          Baseline MAE  With Interaction  MAE Improvement %
SBER   Trend × Vol                0.082          0.076              7.3%
GAZP   Trend × Volume Spike       0.095          0.094              1.1%
```

| MAE Improvement | Интерпретация | Действие |
|-----------------|---------------|----------|
| **> 5%** | Кросс-фича очень полезна ✅ | **ОБЯЗАТЕЛЬНО** добавить |
| **2-5%** | Умеренная польза | Добавить, если не усложняет |
| **< 2%** | Не добавляет ценности ❌ | Не использовать |

### Как добавить кросс-фичи

```python
# В trend_features.py
def build_trend_features(df):
    # ... существующие фичи ...
    
    # Кросс-фичи
    result['trend_vol_interaction'] = (
        result['trend_signal'] * df['realized_vol_5d']
    )
    
    result['trend_volume_interaction'] = (
        result['trend_strength'] * df['volume_spike']
    )
    
    return result

# Обновить список фич
TREND_FEATURE_COLUMNS = [
    # ... существующие ...
    'trend_vol_interaction',
    'trend_volume_interaction'
]
```

### Почему кросс-фичи важны?

LightGBM **не всегда** находит оптимальные комбинации фич:
- Требуется много деревьев
- Может не хватить глубины

**Явная кросс-фича**:
- Сразу показывает модели паттерн
- Feature importance выше
- Быстрее сходится

---

## ✅ Тест 9: Short History Analysis

### Цель
Отдельная статистика для тикеров с **короткой историей**.

### Проблема коротких историй

**Тикеры:** YNDX, FIVE, LENT, OZON, TCSG, VKCO

**Проблемы:**
1. **Мало данных** для long-window фич (SMA_200 требует 200+ дней)
2. **Больше NaN** в начале истории
3. **Другое распределение** (более молодые компании → выше волатильность)

### Что делает тест

1. **Разделение на группы**:
   - Короткие: < 500 дней
   - Длинные: > 1000 дней

2. **Сравнение статистик**:
   - Количество наблюдений
   - Mean/Std для ключевых фич
   - % NaN в long-window фичах

### Интерпретация

```
📊 Короткие истории (< 500 дней):
=============================================================================
Тикер  N obs  dist_to_sma_20_mean  dist_to_sma_200_mean  % NaN SMA_200
YNDX    356         0.012                 NaN                 100%
FIVE    242        -0.008                 NaN                 100%
LENT    990         0.005                0.018                20%

📊 Длинные истории (> 1000 дней):
=============================================================================
Тикер  N obs  dist_to_sma_20_mean  dist_to_sma_200_mean  % NaN SMA_200
SBER   1301         0.003                0.005                 0%
GAZP   1301        -0.001                0.002                 0%
```

### Решения для коротких историй

#### 1. Отдельный feature set
```python
# В feature_builder.py
def build_trend_features(df, short_history=False):
    if short_history:
        # Только короткие окна
        windows = [20, 50]
    else:
        # Полный набор
        windows = [20, 50, 200]
    
    for window in windows:
        result[f'dist_to_sma_{window}'] = dist_to_ma(close, sma(close, window))
```

#### 2. Отдельная модель
```python
# Две модели
model_short = LGBMRegressor()  # Для YNDX, FIVE, LENT
model_long = LGBMRegressor()   # Для SBER, GAZP, LKOH

# Выбор модели при предсказании
if ticker in SHORT_HISTORY_TICKERS:
    prediction = model_short.predict(features)
else:
    prediction = model_long.predict(features)
```

#### 3. Imputation для NaN
```python
# Forward fill с ограничением
df['dist_to_sma_200'] = df['dist_to_sma_200'].fillna(method='ffill', limit=10)

# Или использовать среднее по сектору
sector_mean = df_sector['dist_to_sma_200'].mean()
df['dist_to_sma_200'] = df['dist_to_sma_200'].fillna(sector_mean)
```

---

## 🚀 Как использовать продвинутые тесты

### Запуск

```bash
cd /Users/nikitabaslykov/Documents/Python/Trading/MOEXScanner/ML/02_feature_engineering
python trend_features_tests_advanced.py
```

### Или в notebook

Скопируйте код из `trend_features_tests_advanced.py` в новую ячейку в `04_trend_features.ipynb`.

---

## 📊 Действия после тестов

### 1. OOS Value Test → Удалить нестабильные фичи

```python
# Если OOS Corr < 0.05 и Corr Decay > 0.2
bad_features = ['dist_to_sma_200', 'momentum_20']

TREND_FEATURE_COLUMNS = [f for f in TREND_FEATURE_COLUMNS if f not in bad_features]
```

### 2. Ablation Test → Оценить общую полезность

```python
# Если MAE Improvement < 2%
# Рассмотреть удаление ВСЕХ трендовых фич
# ИЛИ упростить набор (оставить только топ-3)
```

### 3. Stability Test → Monitoring в продакшне

```python
# Добавить в production pipeline
def check_feature_drift(train_dist, prod_values):
    psi = calculate_psi(train_dist, prod_values)
    if psi > 0.2:
        trigger_retraining()
```

### 4. Interaction Check → Добавить кросс-фичи

```python
# В trend_features.py
result['trend_vol_interaction'] = result['trend_signal'] * df['realized_vol_5d']
result['trend_volume_interaction'] = result['trend_strength'] * df['volume_spike']
```

### 5. Short History → Отдельный pipeline

```python
# feature_builder.py
def build_all_features(df, ticker):
    short_history = ticker in ['YNDX', 'FIVE', 'LENT', 'OZON', 'TCSG', 'VKCO']
    
    trend_features = build_trend_features(df, short_history=short_history)
    # ...
```

---

**Дата создания:** 2026-02-09
**Обновлено:** 2026-02-09 (добавлены продвинутые тесты)
