# 🔄 Гайд по взаимосвязям и работе системы

**Версия:** 2.1  
**Дата обновления:** 2025-12-14 (добавлен модуль объяснимости)

---

## 🎯 Как работает система

### Общий принцип

```
Данные → Признаки → Модель → Прогнозы → Валидация
```

---

## 📊 Полный цикл работы

### Этап 1: Подготовка данных

**Что происходит:**
1. Загрузка OHLCV данных из `data/MOEX_DATA/`
2. Расчёт `log_return = log(close / close.shift(1))`
3. Сохранение в `data/processed/{TICKER}_ohlcv_returns.parquet`

**Модули:**
- `notebooks/01_data_loading.ipynb` (исследование)
- `features/Loaders/load_prices.py` (production)

**Результат:** 31 файл с обработанными данными

---

### Этап 2: Генерация признаков

**Что происходит:**
1. Загрузка обработанных данных
2. Генерация признаков через модули в `features/`
3. Объединение всех признаков в `feature_builder.py`
4. Сохранение в `data/processed_ml/{TICKER}_ml_features.parquet`

**Модули и их взаимосвязи:**

```
feature_builder.py (главный модуль)
    │
    ├── volatility_features.py
    │   ├── realized_volatility()      # RV на окнах 5, 10, 20
    │   ├── ewma_volatility()          # EWMA на окнах 10, 20
    │   ├── parkinson_volatility()     # Parkinson на окнах 10, 20
    │   └── garman_klass_volatility()  # GK на окнах 10, 20
    │
    ├── volume_features.py
    │   ├── volume_profile()           # POC, Value Area
    │   ├── volume_zscore()            # Z-score объёмов
    │   └── volume_spike_detection()   # Обнаружение всплесков
    │
    ├── market_features.py
    │   ├── calculate_beta()           # Beta к IMOEX
    │   ├── calculate_correlation()    # Корреляция с индексом
    │   └── index_volatility()         # Волатильность индекса
    │
    ├── trend_features.py
    │   ├── sma_ema_distances()        # Расстояния до SMA/EMA
    │   ├── momentum()                 # Momentum индикаторы
    │   └── trend_signals()            # Сигналы тренда
    │
    ├── calendar_features.py
    │   ├── day_of_week()              # День недели
    │   ├── month_features()           # Месяц, конец/начало месяца
    │   └── seasonality()              # Сезонность
    │
    └── intraday_features.py
        ├── intraday_volatility()      # IVR
        ├── opening_momentum()         # OPM
        └── price_reversal_count()     # PRC
```

**Результат:** ~66 признаков на тикер

---

### Этап 3: Обучение модели

**Что происходит:**
1. Загрузка всех `*_ml_features.parquet` файлов
2. Создание целевой переменной: `target_vol_5d` (волатильность на 5 дней вперёд)
3. Временной split: train до cutoff, test после
4. Обучение 3 квантильных моделей LightGBM (q16, q50, q84)
5. Сохранение моделей и генерация отчёта

**Модули:**
- `scripts/run_full_pipeline.py` (главный скрипт)
- `03_models/train_global_model.py` (обучение)
- `config/training_config.py` (параметры)

**Конфигурация:**
Все параметры в **одном файле**: `config/training_config.py`
- Train cutoff дата
- Гиперпараметры LightGBM
- Квантили
- Исключаемые тикеры

**Результат:**
- `data/models/global_lgbm_q*.txt` (3 модели)
- `reports/validation_report.csv` (метрики)
- `reports/feature_importance.csv` (важность признаков)

---

### Этап 4: Прогнозирование

**Что происходит:**
1. Загрузка обученных моделей
2. Загрузка признаков для тикера
3. Прогноз квантилей (q16, q50, q84)
4. Опционально: ансамбль с GARCH
5. Опционально: объяснения прогнозов через SHAP

**Модули:**
- `03_models/inference.py` (прогнозирование)
- `models/ensemble.py` (ансамбль LightGBM + GARCH)
- `explainability/shap_wrapper.py` (SHAP объяснения)
- `explainability/text_generator.py` (текстовые объяснения)

**Использование (базовый прогноз):**
```python
from inference import GlobalQuantileModel

model = GlobalQuantileModel(use_ensemble=True)
model.load_models()
predictions = model.predict_ensemble(data)
```

**Использование (с объяснениями):**
```python
from inference import GlobalQuantileModel

model = GlobalQuantileModel()
model.load_models()

# Прогноз с объяснениями
result = model.predict(
    data.tail(1),
    include_explanation=True,
    background_data=data.tail(100)  # Фоновые данные для SHAP
)

# result содержит:
# - 'forecast': DataFrame с прогнозами (q16, q50, q84)
# - 'explanation': Dict с текстовым объяснением и сырыми данными
```

**Результат:** 
- Прогнозы волатильности с интервалами
- Текстовые объяснения на русском языке
- JSON структура для фронтенда

---

### Этап 4.5: Объяснимость прогнозов (NEW!)

**Что происходит:**
1. Инициализация SHAP TreeExplainer с фоновым датасетом
2. Вычисление SHAP значений для каждого признака
3. Генерация текстового объяснения на русском языке
4. Формирование JSON структуры для фронтенда

**Модули:**
- `explainability/shap_wrapper.py` - `ShapExplainer` класс
  - Ленивая инициализация TreeExplainer (оптимизация производительности)
  - Метод `explain_local()` - вычисление SHAP значений
  - Метод `format_explanation()` - форматирование в список словарей
  
- `explainability/text_generator.py` - `ExplanationGenerator` класс
  - Словарь `FEATURE_DESCRIPTIONS` - переводы признаков на русский
  - Метод `generate_text()` - генерация текстового объяснения
  - Метод `generate_detailed_text()` - подробное объяснение с детализацией

**Интеграция в inference.py:**
- Метод `predict()` поддерживает параметр `include_explanation=True`
- Автоматическая инициализация explainer при первом использовании
- Обработка ошибок: при сбое объяснений возвращаются только прогнозы

**Формат результата:**
```python
{
    'forecast': DataFrame,  # Прогнозы q16, q50, q84
    'explanation': {
        'text': str,        # Текстовое объяснение на русском
        'raw_data': List[Dict]  # Сырые данные для визуализации
    }
}
```

**Тестирование:**
```bash
python scripts/test_explanation.py
```
Скрипт проверяет:
- Корректность работы объяснимости
- Структуру JSON ответа
- Обработку edge cases (NaN, нули)

---

### Этап 5: Валидация

**Что происходит:**
1. Загрузка прогнозов и фактических значений
2. Проверка калибровки квантилей (coverage)
3. Расчёт метрик точности (MAE, RMSE, correlation)
4. Анализ по тикерам

**Модули:**
- `scripts/validate_model.py`

**Метрики:**
- Coverage 68% (должно быть ~68%)
- MAE (Mean Absolute Error)
- Correlation (корреляция прогнозов с фактом)
- Стабильность по тикерам

---

## 🔗 Взаимосвязи модулей

### features/ → models/

```
feature_builder.py
    ↓ (генерирует признаки)
processed_ml/{TICKER}_ml_features.parquet
    ↓ (используется для обучения)
train_global_model.py
    ↓ (обучает модель)
global_lgbm_q*.txt
```

### models/ → inference/

```
train_global_model.py
    ↓ (сохраняет)
global_lgbm_q*.txt
    ↓ (загружает)
inference.py
    ↓ (делает прогнозы)
{TICKER}_predictions.csv
```

### config/ → все модули

```
training_config.py
    ↓ (используется)
train_global_model.py
    ↓ (применяет параметры)
Обучение модели
```

---

## 🚀 Типичные сценарии использования

### Сценарий 1: Полный pipeline

```bash
# 1. Генерация признаков
python scripts/run_full_pipeline.py

# 2. Валидация
python scripts/validate_model.py

# 3. Сравнение с предыдущей моделью
python scripts/compare_models.py
```

### Сценарий 2: Только обучение (features готовы)

```bash
# 1. Настройте параметры в config/training_config.py
# 2. Запустите обучение
python scripts/run_full_pipeline.py --skip-features --preset MORE_TRAIN
```

### Сценарий 3: Эксперимент с разными конфигурациями

```bash
# 1. Сохраните текущий отчёт
copy reports\validation_report.csv reports\validation_report_baseline.csv

# 2. Измените пресет в config/training_config.py
# Или используйте --preset
python scripts/run_full_pipeline.py --skip-features --preset REGULARIZED

# 3. Сравните результаты
python scripts/compare_models.py
```

### Сценарий 4: Прогноз с объяснениями

```bash
# 1. Тестирование модуля объяснимости
python scripts/test_explanation.py

# 2. В Python коде
from inference import GlobalQuantileModel
import pandas as pd

model = GlobalQuantileModel()
model.load_models()

# Загружаем данные
df = pd.read_parquet("data/processed_ml/SBER_ml_features.parquet")

# Прогноз с объяснениями
result = model.predict(
    df.tail(1),
    include_explanation=True,
    background_data=df.tail(100)
)

print(result['explanation']['text'])
# Вывод: "Прогноз волатильности (15.00%) сформирован в основном 
#         за счет внутридневного размаха цен (10 дней) и 
#         инерции тренда (20 дней)."
```

---

## 🔄 Потоки данных (детально)

### Поток 1: От сырых данных до признаков

```
MOEX API / CSV файлы
    ↓
data/MOEX_DATA/{TICKER}/1D/*.csv
    ↓
features/Loaders/load_prices.py
    ↓
Расчёт log_return
    ↓
data/processed/{TICKER}_ohlcv_returns.parquet
    ↓
features/feature_builder.py
    ├── volatility_features.py
    ├── volume_features.py
    ├── market_features.py
    ├── trend_features.py
    ├── calendar_features.py
    └── intraday_features.py
    ↓
data/processed_ml/{TICKER}_ml_features.parquet
```

### Поток 2: От признаков до модели

```
data/processed_ml/*.parquet (все тикеры)
    ↓
03_models/train_global_model.py
    ├── load_all_ticker_data()
    ├── create_target_variable()  # target_vol_5d
    ├── time_series_split()       # train/test
    └── train_quantile_models()   # q16, q50, q84
    ↓
data/models/global_lgbm_q*.txt
reports/validation_report.csv
```

### Поток 3: От модели до прогноза

```
data/models/global_lgbm_q*.txt
    ↓
03_models/inference.py
    ├── GlobalQuantileModel.load_models()
    ├── model.predict_ensemble()  # Базовый прогноз
    └── model.predict(include_explanation=True)  # С объяснениями
        ↓
        explainability/shap_wrapper.py
            ├── ShapExplainer (TreeExplainer)
            └── explain_local() → SHAP значения
        ↓
        explainability/text_generator.py
            ├── ExplanationGenerator
            └── generate_detailed_text() → Текст на русском
        ↓
    JSON структура для фронтенда
    data/models/{TICKER}_predictions.csv
```

---

## ⚙️ Конфигурация и параметры

### Где что настраивается

| Параметр | Файл | Описание |
|----------|------|----------|
| **Train/Test split** | `config/training_config.py` | Дата cutoff для разделения |
| **Гиперпараметры** | `config/training_config.py` | num_leaves, learning_rate, регуляризация |
| **Квантили** | `config/training_config.py` | [0.16, 0.50, 0.84] |
| **Исключаемые тикеры** | `config/training_config.py` | Список тикеров для исключения |
| **Пресеты** | `config/training_config.py` | BASELINE, MORE_TRAIN, REGULARIZED, NO_TICKER |

### Как изменить параметры

1. Откройте `config/training_config.py`
2. Измените нужные параметры:
   ```python
   TRAIN_CUTOFF_DATE = '2024-06-01'  # Для 70/30 split
   LGBM_PARAMS = {
       'num_leaves': 63,
       'learning_rate': 0.05,
       'lambda_l1': 0.5,  # Увеличить регуляризацию
       ...
   }
   ```
3. Или выберите пресет:
   ```python
   ACTIVE_PRESET = 'MORE_TRAIN'
   ```
4. Запустите обучение:
   ```bash
   python scripts/run_full_pipeline.py --skip-features
   ```

---

## 🔍 Отладка и диагностика

### Проблема: Модель не обучается

**Проверьте:**
1. Есть ли файлы в `data/processed_ml/`?
2. Правильно ли настроен `config/training_config.py`?
3. Достаточно ли данных для train/test split?

### Проблема: Низкая точность

**Проверьте:**
1. Запустите `scripts/validate_model.py` для детальной диагностики
2. Посмотрите `reports/validation_detailed.csv` - какие тикеры плохо работают?
3. Проверьте feature importance в `reports/feature_importance.csv`

### Проблема: Модель деградирует на новых данных

**Решение:**
1. Используйте пресет `MORE_TRAIN` (больше train данных)
2. Увеличьте регуляризацию (пресет `REGULARIZED`)
3. Исключите плохие тикеры в `EXCLUDE_TICKERS`

---

## 📚 Связанные документы

- [SYSTEM_STRUCTURE.md](SYSTEM_STRUCTURE.md) - Полная структура системы
- [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) - План разработки
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - Текущие проблемы и метрики

