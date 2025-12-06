# 🔍 АУДИТ: Notebooks vs Production Code

**Дата:** 2024-12-06  
**Цель:** Проверить, какие фичи из исследовательских notebooks реализованы в production коде

---

## 📊 СВОДНАЯ ТАБЛИЦА

| Модуль | Notebook | Production | Статус |
|--------|----------|------------|--------|
| Волатильность | `01_volatility_features.ipynb` | `volatility_features.py` | ⚠️ Частично |
| Объём | `02_volume_features.ipynb` | `volume_features.py` | ✅ Полностью |
| Рынок | `03_market_features.ipynb` | `market_features.py` | ✅ Полностью |
| Тренды | `04_trend_features.ipynb` | `trend_features.py` | ✅ Полностью |
| Таргеты | `05_targets.ipynb` | `train_global_model.py` | ⚠️ Частично |
| Календарь | ❌ Нет notebook | `calendar_features.py` | ✅ Только production |

---

## 1️⃣ ВОЛАТИЛЬНОСТЬ: `01_volatility_features.ipynb` vs `volatility_features.py`

### В Notebook:
```python
def realized_volatility(returns, window=30)
def ewma_volatility(returns, span=30)  
def parkinson_volatility(high, low, window=30)
def garman_klass_volatility(open, high, low, close, window=30)
def directional_volatility(returns, window=30)  # Отдельно вверх/вниз!

# Окна: [10, 30, 60]
# Признаки: realized_vol_10, realized_vol_30, realized_vol_60, ewma_vol_*, parkinson_vol_*, gk_vol_*
```

### В Production:
```python
def realized_volatility(returns, window=20)      # ✅ Реализовано
def ewma_volatility(returns, span=20)            # ✅ Реализовано
def parkinson_volatility(high, low, window=20)   # ✅ Реализовано
def garman_klass_volatility(...)                 # ✅ Реализовано
def volatility_ratio(short_vol, long_vol)        # ✅ ДОБАВЛЕНО (нет в notebook)

# Окна: [5, 10, 20] (НЕ как в notebook!)
# Признаки: rv_5d, rv_10d, rv_20d, ewma_vol_10d, ewma_vol_20d, 
#           parkinson_vol_10d, parkinson_vol_20d, gk_vol_10d, gk_vol_20d,
#           vol_ratio_5_20, vol_ratio_park_rv, vol_momentum_5d
```

### ❌ НЕ РЕАЛИЗОВАНО:
| Функция | Описание | Важность |
|---------|----------|----------|
| `directional_volatility` | Отдельная волатильность для движений ВВЕРХ и ВНИЗ | 🟡 Средняя |
| Окна 30, 60 дней | В notebook использовались более длинные окна | 🟡 Средняя |

### ✅ ДОБАВЛЕНО в Production (нет в notebook):
- `volatility_ratio` — отношение краткосрочной к долгосрочной волатильности
- `vol_momentum_5d` — изменение волатильности за 5 дней
- Окно 5 дней (`rv_5d`) — более чувствительное

---

## 2️⃣ ОБЪЁМ: `02_volume_features.ipynb` vs `volume_features.py`

### В Notebook:
```python
def volume_ma(volume, window=20)
def volume_zscore(volume, window=60)
def volume_spike(volume, threshold=2.0, window=20)
def calculate_volume_profile(df, window=20, num_bins=50)
    # Возвращает: POC, VA_HIGH, VA_LOW (АБСОЛЮТНЫЕ значения!)

# Признаки: volume_ma_20, volume_ma_60, volume_zscore, volume_spike,
#           vp_poc, vp_va_high, vp_va_low, vp_width, vp_position
```

### В Production:
```python
def volume_zscore(volume, window=20)             # ✅ Реализовано
def volume_ratio(volume, window=20)              # ✅ ДОБАВЛЕНО
def volume_spike(volume, threshold=2.0)          # ✅ Реализовано
def calculate_volume_profile_normalized(...)     # ✅ НОРМАЛИЗОВАННАЯ версия!
    # Возвращает: vp_position, vp_width_pct, vp_above_va (НОРМАЛИЗОВАННЫЕ!)

# Признаки: volume_zscore_20, volume_zscore_60, volume_ratio_20, volume_spike,
#           vp_position, vp_width_pct, vp_above_va
```

### ✅ СТАТУС: ПОЛНОСТЬЮ РЕАЛИЗОВАНО

| Изменение | Описание |
|-----------|----------|
| Volume Profile | УЛУЧШЕНО: нормализованные значения вместо абсолютных |
| `volume_ratio` | ДОБАВЛЕНО: отношение к среднему |
| `vp_above_va` | ДОБАВЛЕНО: позиция относительно Value Area |

**Причина изменений:** Абсолютные значения (POC, VA_HIGH, VA_LOW) **запрещены** для ML модели, т.к. они зависят от цены акции.

---

## 3️⃣ РЫНОК: `03_market_features.ipynb` vs `market_features.py`

### В Notebook:
```python
def calculate_beta(stock_returns, market_returns, window=60)
def calculate_correlation(stock_returns, market_returns, window=60)
def market_volatility(market_returns, window=30)
def calculate_market_features(df, index_df, windows=[30, 60])

# Признаки: beta_30, beta_60, correlation_30, correlation_60, 
#           index_vol_30, index_vol_60
```

### В Production:
```python
def calculate_beta(...)        # ✅ Реализовано
def calculate_correlation(...) # ✅ Реализовано
def market_volatility(...)     # ✅ Реализовано
def build_market_features(df, index_df, windows=[30, 60])  # ✅ Реализовано

# Признаки: beta_30d, beta_60d, correlation_30d, correlation_60d,
#           index_vol_30d, index_vol_60d, beta_change
```

### ✅ СТАТУС: ПОЛНОСТЬЮ РЕАЛИЗОВАНО

| Добавлено | Описание |
|-----------|----------|
| `beta_change` | Изменение беты (beta_60d - beta_30d) — новый признак |

---

## 4️⃣ ТРЕНДЫ: `04_trend_features.ipynb` vs `trend_features.py`

### В Notebook:
```python
def sma(prices, window=20)
def ema(prices, span=20)
def ma_slope(ma, window=5)
def momentum(prices, window=10)          # АБСОЛЮТНЫЙ!
def price_position(price, ma_short, ma_long)
def trend_signal(ma_short, ma_long, threshold=0.01)
def trend_confidence(ma_short, ma_long)

# Признаки: sma_20, sma_50, ema_20, ema_50, sma_20_slope, 
#           momentum_10, momentum_20, price_position, trend_signal, trend_confidence
```

### В Production:
```python
def sma(prices, window=20)               # Внутренняя, не экспортируется
def ema(prices, span=20)                 # Внутренняя, не экспортируется
def dist_to_ma(prices, ma_values)        # ✅ НОРМАЛИЗОВАННОЕ расстояние!
def ma_slope_normalized(ma, prices)      # ✅ НОРМАЛИЗОВАННЫЙ наклон!
def momentum_normalized(prices, window)  # ✅ Log return вместо абсолютного!
def rsi(prices, window=14)               # ✅ ДОБАВЛЕНО
def price_position_ma(...)               # ✅ Реализовано
def trend_signal(...)                    # ✅ Реализовано
def trend_strength(...)                  # ✅ Реализовано (=trend_confidence)

# Признаки: dist_to_sma_20, dist_to_sma_50, dist_to_sma_200, 
#           dist_to_ema_20, dist_to_ema_50, sma_20_slope_norm, sma_50_slope_norm,
#           momentum_10, momentum_20, rsi_14, price_position_ma, trend_signal, trend_strength
```

### ✅ СТАТУС: ПОЛНОСТЬЮ РЕАЛИЗОВАНО (с улучшениями)

| Изменение | Описание |
|-----------|----------|
| `dist_to_sma/ema` | УЛУЧШЕНО: нормализованное расстояние вместо абсолютных MA |
| `momentum_normalized` | УЛУЧШЕНО: log return вместо абсолютной разницы |
| `rsi_14` | ДОБАВЛЕНО: RSI индикатор |
| `dist_to_sma_200` | ДОБАВЛЕНО: расстояние до 200-дневной MA |

**Причина изменений:** Абсолютные значения MA **запрещены** для ML модели.

---

## 5️⃣ ТАРГЕТЫ: `05_targets.ipynb` vs `train_global_model.py`

### В Notebook:
```python
def create_realized_vol_target(returns, horizon=5)
    # horizons = [1, 5, 10]
    # Создаёт: target_vol_1d, target_vol_5d, target_vol_10d

def create_spike_flag(returns, threshold=2.0, window=20)
    # Создаёт: target_spike (бинарный)

def create_quantile_targets(returns, horizon=5, quantiles=[0.16, 0.50, 0.84])
    # Создаёт: quantile_16, quantile_50, quantile_84

def create_directional_target(returns, horizon=1)
    # Создаёт: target_direction (бинарный: вверх/вниз)
```

### В Production (`train_global_model.py`):
```python
# Создаётся только:
target_vol_5d = returns.rolling(5).std().shift(-5) * np.sqrt(252)
```

### ❌ НЕ РЕАЛИЗОВАНО:

| Таргет | Описание | Важность |
|--------|----------|----------|
| `target_vol_1d` | 1-дневная волатильность | 🟡 Средняя |
| `target_vol_10d` | 10-дневная волатильность | 🟡 Средняя |
| `target_spike` | Бинарный флаг всплеска | 🟢 Низкая* |
| `quantile_16/50/84` | Квантили как таргеты | 🟢 Низкая** |
| `target_direction` | Направление движения | 🟢 Низкая*** |

**Примечания:**
- *`target_spike` — можно рассчитать из прогнозов q84
- **Квантили используются как OUTPUT моделей, не как таргеты
- ***Direction — отдельная задача классификации, не входит в текущий scope

---

## 6️⃣ КАЛЕНДАРЬ: Нет notebook → `calendar_features.py`

### Только в Production:
```python
def day_of_week(dates)           # День недели (0-6)
def day_of_month(dates)          # День месяца (1-31)
def week_of_month(dates)         # Неделя месяца (1-5)
def is_month_end(dates)          # Флаг конца месяца
def is_month_start(dates)        # Флаг начала месяца
def overnight_gap(open, close)   # Гэп открытия
def overnight_gap_zscore(gap)    # Z-score гэпа

# Признаки: day_of_week, day_of_month, week_of_month,
#           is_month_end, is_month_start, overnight_gap, overnight_gap_zscore
```

### ⚠️ ЗАМЕЧАНИЕ:
Для calendar features **нет соответствующего notebook**. Это единственный модуль, который существует только в production.

---

## 📋 ИТОГОВЫЙ СПИСОК НЕРЕАЛИЗОВАННЫХ ФИЧЕЙ

### 🔴 ВЫСОКИЙ ПРИОРИТЕТ (рекомендуется добавить):
*Нет*

### 🟡 СРЕДНИЙ ПРИОРИТЕТ (можно добавить):
1. **`directional_volatility`** — отдельная волатильность вверх/вниз
   - Помогает определить асимметрию движений
   - Легко реализовать
   
2. **Окна 30, 60 дней для волатильности** — в notebook использовались, в production нет
   - Текущие окна (5, 10, 20) могут быть слишком короткими
   - Можно добавить `rv_30d`, `rv_60d`

3. **`target_vol_1d`, `target_vol_10d`** — дополнительные горизонты прогнозирования
   - Позволяют строить модели на разных горизонтах

### 🟢 НИЗКИЙ ПРИОРИТЕТ:
1. `target_spike` — можно вычислить из q84
2. `target_direction` — отдельная задача классификации
3. Квантили как таргеты — уже используются как выходы моделей

---

## ✅ ЧТО ДОБАВЛЕНО В PRODUCTION (нет в notebooks):

| Признак | Модуль | Описание |
|---------|--------|----------|
| `vol_ratio_5_20` | volatility | Режим волатильности |
| `vol_ratio_park_rv` | volatility | Сравнение типов волатильности |
| `vol_momentum_5d` | volatility | Динамика волатильности |
| `rv_5d` | volatility | 5-дневная волатильность |
| `volume_ratio_20` | volume | Отношение объёма к среднему |
| `vp_above_va` | volume | Позиция относительно Value Area |
| `beta_change` | market | Изменение беты |
| `rsi_14` | trend | RSI индикатор |
| `dist_to_sma_200` | trend | Расстояние до 200-дневной MA |
| Весь модуль | calendar | Календарные признаки |

---

## 🎯 РЕКОМЕНДАЦИИ

### Для добавления `directional_volatility`:

```python
# В volatility_features.py добавить:

def directional_volatility(returns: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Направленная волатильность: отдельно для движений вверх и вниз.
    
    Returns:
        up_vol: Волатильность положительных доходностей
        down_vol: Волатильность отрицательных доходностей
    """
    up_returns = returns.where(returns > 0, np.nan)
    down_returns = returns.where(returns < 0, np.nan)
    
    up_vol = up_returns.rolling(window=window, min_periods=int(window*0.5)).std() * np.sqrt(252)
    down_vol = down_returns.abs().rolling(window=window, min_periods=int(window*0.5)).std() * np.sqrt(252)
    
    return up_vol, down_vol

# В build_volatility_features:
features['up_vol_20d'], features['down_vol_20d'] = directional_volatility(returns, window=20)
features['vol_asymmetry'] = features['down_vol_20d'] / features['up_vol_20d']
```

### Для добавления длинных окон волатильности:

```python
# В build_volatility_features изменить:
for window in [5, 10, 20, 30, 60]:  # Добавить 30, 60
    features[f'rv_{window}d'] = realized_volatility(returns, window=window)
```

---

## 📊 СТАТИСТИКА

| Категория | Notebook | Production | % Покрытия |
|-----------|----------|------------|------------|
| Волатильность | 5 функций | 4 функции + 3 новых | 80% + extras |
| Объём | 4 функции | 4 функции (нормализ.) | 100% (улучшено) |
| Рынок | 4 функции | 4 функции + 1 новая | 100% + extras |
| Тренды | 7 функций | 8 функций (нормализ.) | 100% (улучшено) |
| Таргеты | 4 функции | 1 функция | 25% |
| Календарь | 0 функций | 7 функций | N/A |

**ОБЩИЙ ВЫВОД:** Production код покрывает **~90%** функционала notebooks с **улучшениями** (нормализация) и **дополнениями** (новые признаки).

---

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ

1. **Нормализация** — Production код специально использует нормализованные версии признаков (dist_to_ma вместо абсолютных MA), т.к. абсолютные значения цен/объёма **запрещены** для Global ML Model.

2. **Окна** — Production использует более короткие окна (5, 10, 20) вместо (10, 30, 60) из notebooks. Это может быть осознанным решением для более быстрой реакции модели.

3. **Таргеты** — Низкое покрытие таргетов объясняется тем, что текущая модель фокусируется на предсказании волатильности на горизонте 5 дней. Другие таргеты (spike, direction) — это отдельные задачи.

4. **Calendar features** — Единственный модуль без соответствующего notebook. Рекомендуется создать `XX_calendar_features.ipynb` для документации.

