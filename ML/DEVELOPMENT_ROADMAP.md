# 🗺️ ПЛАН РАЗРАБОТКИ ML PIPELINE
## Итоговый Roadmap

**Дата создания:** 2024-12-06  
**Версия:** 1.0  
**Статус:** Активный

---

## 📚 СВЯЗАННЫЕ ДОКУМЕНТЫ

| Документ | Путь | Содержание |
|----------|------|------------|
| **Структура проекта** | [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Полное описание всех модулей, папок и потоков данных |
| **Аудит архитектуры** | [`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md) | Соответствие схеме, критические проблемы |
| **Notebooks vs Production** | [`NOTEBOOKS_VS_PRODUCTION_AUDIT.md`](NOTEBOOKS_VS_PRODUCTION_AUDIT.md) | Сравнение фичей notebook/production |

---

## 📊 ТЕКУЩИЙ СТАТУС

| Категория | Реализовано | Требуется | Примечание |
|-----------|-------------|-----------|------------|
| **Core Features (D1)** | 95% | +5% | `directional_volatility`, длинные окна |
| **Intraday Features (H1)** | 0% | 100% | Полностью отсутствует |
| **Models** | 70% | +30% | Ensemble не интегрирован |
| **Advanced** | 20% | +80% | Adjuster, EWS отсутствуют |
| **Output** | 50% | +50% | API, SHAP интеграция |

**Общее соответствие схеме:** ~65%

---

# 🎯 ПЛАН РАЗРАБОТКИ

## ФАЗА 1: Доработка D1 фичей из Notebooks
**Срок:** 1-2 дня  
**Сложность:** 🟢 Низкая

### Задача 1.1: Добавить `directional_volatility`
**Источник:** `01_volatility_features.ipynb` (строки 24-28)  
**Целевой файл:** `features/volatility_features.py`

**Что делать:**
```python
# Добавить функцию:
def directional_volatility(returns: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Отдельная волатильность для движений ВВЕРХ и ВНИЗ."""
    up_returns = returns.where(returns > 0, np.nan)
    down_returns = returns.where(returns < 0, np.nan)
    
    up_vol = up_returns.rolling(window=window, min_periods=int(window*0.5)).std() * np.sqrt(252)
    down_vol = down_returns.abs().rolling(window=window, min_periods=int(window*0.5)).std() * np.sqrt(252)
    
    return up_vol, down_vol

# В build_volatility_features() добавить:
features['up_vol_20d'], features['down_vol_20d'] = directional_volatility(returns, window=20)
features['vol_asymmetry'] = features['down_vol_20d'] / features['up_vol_20d']
```

**Новые признаки:** `up_vol_20d`, `down_vol_20d`, `vol_asymmetry`

---

### Задача 1.2: Добавить длинные окна волатильности
**Источник:** `01_volatility_features.ipynb` (окна 30, 60)  
**Целевой файл:** `features/volatility_features.py`

**Что делать:**
```python
# В build_volatility_features() изменить окна:
for window in [5, 10, 20, 30, 60]:  # Добавить 30, 60
    features[f'rv_{window}d'] = realized_volatility(returns, window=window)
```

**Новые признаки:** `rv_30d`, `rv_60d`

---

### Задача 1.3: Создать notebook для calendar_features
**Причина:** Единственный модуль без соответствующего notebook  
**Целевой файл:** `02_feature_engineering/XX_calendar_features.ipynb`

**Содержание:**
- Документация всех функций из `calendar_features.py`
- Визуализация сезонности
- Анализ влияния дней недели на волатильность

---

## ФАЗА 2: Разработка Intraday Features (H1)
**Срок:** 3-5 дней  
**Сложность:** 🔴 Высокая  
**Критичность:** 🔴 **ВЫСОКАЯ** — ключевой компонент для улучшения точности

### Задача 2.1: Исследовательский Notebook
**Целевой файл:** `02_feature_engineering/07_intraday_features.ipynb`

**Содержание:**
1. Загрузка H1 данных из `data/MOEX_DATA/{TICKER}/1H/`
2. Исследование и визуализация внутридневных паттернов
3. Реализация 5 ключевых метрик (см. ниже)
4. Агрегация H1 → D1

**5 КЛЮЧЕВЫХ МЕТРИК:**

| Метрика | Формула | Описание |
|---------|---------|----------|
| **IVR** (Intraday Vol Realized) | `std(hourly_returns) * sqrt(252*7)` | Реальная внутридневная волатильность |
| **OPM** (Opening Momentum) | `(close_10:00 - open) / open` | Momentum первого часа торгов |
| **VDS** (Vol Distribution Skew) | `skew(hourly_returns)` | Асимметрия внутридневных движений |
| **PRC** (Price Reversal Count) | `count(sign_changes)` | Количество разворотов за день |
| **POCS** (POC Shift Intraday) | `(POC_last_hour - POC_first_hour) / ATR` | Сдвиг POC в течение дня |

**Пример кода:**
```python
def load_hourly_data(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Загрузка часовых данных."""
    path = data_dir / ticker / '1H' / f'{ticker}_1H.csv'
    return pd.read_csv(path, parse_dates=['begin'])

def aggregate_hourly_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегация часовых данных в дневные фичи."""
    hourly_df['date'] = hourly_df['begin'].dt.date
    hourly_df['hourly_return'] = np.log(hourly_df['close'] / hourly_df['close'].shift(1))
    
    daily = hourly_df.groupby('date').agg({
        'hourly_return': ['std', 'skew', lambda x: (np.sign(x) != np.sign(x.shift())).sum()],
        'volume': 'sum'
    })
    
    daily.columns = ['ivr', 'vds', 'prc', 'total_volume']
    daily['ivr'] = daily['ivr'] * np.sqrt(252 * 7)  # Аннуализация
    
    return daily
```

---

### Задача 2.2: Production модуль
**Целевой файл:** `features/intraday_features.py`

**Структура модуля:**
```python
"""
Внутридневные фичи (H1 → D1) для Global ML Model.

Агрегирует часовые данные в дневные признаки:
- IVR: Внутридневная волатильность
- OPM: Утренний momentum  
- VDS: Асимметрия внутридневных движений
- PRC: Количество разворотов
- POCS: Сдвиг POC
"""

def ivr(hourly_returns: pd.Series) -> float:
    """Intraday Volatility Realized."""
    ...

def opm(open_price: float, close_10am: float) -> float:
    """Opening Momentum (первый час)."""
    ...

def vds(hourly_returns: pd.Series) -> float:
    """Vol Distribution Skew."""
    ...

def prc(hourly_returns: pd.Series) -> int:
    """Price Reversal Count."""
    ...

def pocs(hourly_poc: pd.Series, atr: float) -> float:
    """POC Shift Intraday."""
    ...

def build_intraday_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Строит все внутридневные фичи."""
    ...

INTRADAY_FEATURE_COLUMNS = ['ivr', 'opm', 'vds', 'prc', 'pocs']
```

---

### Задача 2.3: Интеграция в feature_builder.py
**Целевой файл:** `features/feature_builder.py`

**Изменения:**
```python
# Добавить импорт:
from features.intraday_features import build_intraday_features, INTRADAY_FEATURE_COLUMNS

# В build_all_features() добавить:
# === 6. ВНУТРИДНЕВНЫЕ ФИЧИ (H1) ===
if hourly_df is not None:
    print(f"    • Внутридневные фичи (H1)...")
    intraday_features = build_intraday_features(hourly_df)
    features_to_concat.append(intraday_features)
```

---

### Задача 2.4: Pipeline загрузки H1 данных
**Целевой файл:** `features/Loaders/load_hourly.py`

**Функции:**
```python
def load_hourly_data(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Загрузка часовых данных для тикера."""
    ...

def load_all_hourly_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Загрузка часовых данных для всех тикеров."""
    ...
```

---

## ФАЗА 3: Интеграция Ensemble
**Срок:** 2-3 дня  
**Сложность:** 🟡 Средняя

### Задача 3.1: Production модуль ensemble
**Источник:** `03_models/04_ensemble_model.ipynb`  
**Целевой файл:** `models/ensemble.py`

**Структура:**
```python
"""
Ensemble модуль: объединение GARCH + LightGBM.

Методы взвешивания:
- Статическое взвешивание (простое среднее)
- Адаптивное взвешивание (по метрикам на валидации)
"""

class EnsembleModel:
    def __init__(self, garch_weight: float = 0.3, lgbm_weight: float = 0.7):
        self.garch_weight = garch_weight
        self.lgbm_weight = lgbm_weight
    
    def predict(self, garch_pred: np.ndarray, lgbm_pred: np.ndarray) -> np.ndarray:
        """Взвешенное объединение прогнозов."""
        return self.garch_weight * garch_pred + self.lgbm_weight * lgbm_pred
    
    def adaptive_weights(self, val_metrics: Dict) -> Tuple[float, float]:
        """Адаптивное определение весов по метрикам."""
        ...
```

---

### Задача 3.2: Интеграция в inference.py
**Целевой файл:** `03_models/inference.py`

**Изменения:**
```python
from models.ensemble import EnsembleModel

class GlobalQuantileModel:
    def __init__(self):
        ...
        self.ensemble = EnsembleModel()
    
    def predict_ensemble(self, data: pd.DataFrame, garch_forecasts: pd.DataFrame) -> pd.DataFrame:
        """Прогноз с использованием ensemble."""
        lgbm_pred = self.predict(data)
        return self.ensemble.predict(garch_forecasts, lgbm_pred)
```

---

## ФАЗА 4: Intraday Adjuster
**Срок:** 2-3 дня  
**Сложность:** 🟡 Средняя

### Задача 4.1: Модуль корректировки
**Целевой файл:** `models/intraday_adjuster.py`

**Концепция:**
```python
"""
Корректировка дневных прогнозов на основе последних H1 данных.

Логика:
1. Получаем дневной прогноз от Ensemble
2. Смотрим на последние 6 часов (H1 данные)
3. Корректируем прогноз на основе:
   - IVR vs прогноз (если IVR >> прогноза, увеличиваем)
   - VDS (если сильная асимметрия, корректируем направление)
   - Volume Spike (если аномальный объём, увеличиваем прогноз)
"""

class IntradayAdjuster:
    def __init__(self, adjustment_factor: float = 0.2):
        self.factor = adjustment_factor
    
    def adjust(self, daily_forecast: float, last_6h_features: Dict) -> float:
        """Корректировка прогноза."""
        adjustment = 0.0
        
        # IVR adjustment
        if last_6h_features['ivr'] > daily_forecast * 1.5:
            adjustment += self.factor * (last_6h_features['ivr'] / daily_forecast - 1)
        
        # Volume spike adjustment
        if last_6h_features['volume_zscore'] > 2.0:
            adjustment += self.factor * 0.5
        
        return daily_forecast * (1 + adjustment)
```

---

## ФАЗА 5: Early Warning System
**Срок:** 3-4 дня  
**Сложность:** 🔴 Высокая

### Задача 5.1: Модуль мониторинга
**Целевой файл:** `models/early_warning.py`

**Концепция:**
```python
"""
Real-time мониторинг H1 данных для раннего предупреждения.

Алерты:
- Volume Spike: z-score > 3.0
- POC Shift: сдвиг > 2 ATR за последние 3 часа
- Volatility Explosion: IVR > 2x дневного прогноза
- Correlation Break: корреляция с IMOEX < 0.3
"""

class EarlyWarningSystem:
    def __init__(self, thresholds: Dict):
        self.thresholds = thresholds
        self.alerts = []
    
    def check_alerts(self, current_h1_data: pd.DataFrame, daily_forecast: float) -> List[Alert]:
        """Проверка всех алертов."""
        alerts = []
        
        if current_h1_data['volume_zscore'] > self.thresholds['volume_spike']:
            alerts.append(Alert('VOLUME_SPIKE', severity='HIGH'))
        
        # ... другие проверки
        
        return alerts
```

---

## ФАЗА 6: API и интеграция
**Срок:** 3-5 дней  
**Сложность:** 🟡 Средняя

### Задача 6.1: REST API endpoint
**Целевой файл:** `api/predictions.py` или интеграция в `backend/`

**Endpoints:**
```
GET  /api/v1/predictions/{ticker}     # Прогноз для тикера
GET  /api/v1/predictions/all          # Прогнозы для всех тикеров
POST /api/v1/predictions/batch        # Batch прогнозы
GET  /api/v1/alerts/{ticker}          # Алерты EWS
```

---

### Задача 6.2: SHAP интеграция
**Источник:** `05_explainability/01_shap_explainer.ipynb`  
**Целевой файл:** `explainability/shap_explainer.py`

**Функции:**
```python
def explain_prediction(model, data: pd.DataFrame) -> Dict:
    """SHAP values для одного прогноза."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    
    return {
        'base_value': explainer.expected_value,
        'shap_values': shap_values,
        'feature_importance': get_feature_importance(shap_values)
    }
```

---

# 📋 ЧЕКЛИСТ РЕАЛИЗАЦИИ

## Фаза 1: D1 фичи (1-2 дня)
- [ ] 1.1: Добавить `directional_volatility` в `volatility_features.py`
- [ ] 1.2: Добавить окна 30, 60 дней
- [ ] 1.3: Создать `XX_calendar_features.ipynb`
- [ ] **Тест:** Пересоздать `processed_ml/` и проверить новые колонки
- [ ] **Тест:** Запустить `train_global_model.py` с новыми фичами

## Фаза 2: H1 фичи (3-5 дней)
- [ ] 2.1: Создать `07_intraday_features.ipynb`
- [ ] 2.2: Реализовать IVR, OPM, VDS, PRC, POCS
- [ ] 2.3: Создать `intraday_features.py`
- [ ] 2.4: Создать `Loaders/load_hourly.py`
- [ ] 2.5: Интегрировать в `feature_builder.py`
- [ ] **Тест:** Проверить агрегацию H1 → D1
- [ ] **Тест:** Запустить полный pipeline с H1 фичами

## Фаза 3: Ensemble (2-3 дня)
- [ ] 3.1: Создать `models/ensemble.py`
- [ ] 3.2: Интегрировать в `inference.py`
- [ ] 3.3: Обновить `run_backtest_pipeline.py`
- [ ] **Тест:** Сравнить метрики GARCH, LightGBM, Ensemble

## Фаза 4: Adjuster (2-3 дня)
- [ ] 4.1: Создать `models/intraday_adjuster.py`
- [ ] 4.2: Интегрировать в inference pipeline
- [ ] **Тест:** Проверить влияние корректировок на метрики

## Фаза 5: EWS (3-4 дня)
- [ ] 5.1: Создать `models/early_warning.py`
- [ ] 5.2: Определить пороги алертов
- [ ] **Тест:** Ретроспективный анализ алертов

## Фаза 6: API (3-5 дней)
- [ ] 6.1: REST API endpoints
- [ ] 6.2: SHAP production модуль
- [ ] 6.3: Интеграция с Go backend
- [ ] **Тест:** End-to-end тест API

---

# ⏱️ TIMELINE

```
Неделя 1:
├── День 1-2: Фаза 1 (D1 фичи)
├── День 3-5: Фаза 2 (H1 notebook + module)
└── День 5-7: Фаза 2 (интеграция H1)

Неделя 2:
├── День 1-3: Фаза 3 (Ensemble)
├── День 3-5: Фаза 4 (Adjuster)
└── День 5-7: Фаза 5 (EWS)

Неделя 3:
├── День 1-5: Фаза 6 (API + SHAP)
└── День 5-7: Тестирование + документация
```

**Общий срок:** ~3 недели

---

# 🎯 ПРИОРИТЕТЫ

## 🔴 КРИТИЧНО (блокирует production)
1. **Фаза 2:** Intraday H1 фичи — ключевой компонент для улучшения точности
2. **Фаза 3:** Ensemble — необходим для финального прогноза

## 🟡 ВАЖНО (улучшает качество)
3. **Фаза 1:** D1 фичи — quick wins
4. **Фаза 4:** Adjuster — улучшение real-time прогнозов

## 🟢 ЖЕЛАТЕЛЬНО (полнота функционала)
5. **Фаза 5:** EWS — дополнительный функционал
6. **Фаза 6:** API — интеграция с backend

---

# 📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

После реализации всех фаз:

| Метрика | Текущее | Ожидаемое |
|---------|---------|-----------|
| **Соответствие схеме** | 65% | 95% |
| **Покрытие фичей** | 51 признак | ~70 признаков |
| **Interval Coverage** | ~68% | >70% |
| **Quantile Loss** | baseline | -10-15% |
| **API Ready** | ❌ | ✅ |
| **Real-time** | ❌ | ✅ |

---

**Готово к реализации!** 🚀

Начинайте с **Фазы 1** для quick wins, затем переходите к **Фазе 2** (H1 фичи) как главному приоритету.

