# 📁 Полная структура системы ML Pipeline

**Версия:** 2.1  
**Дата обновления:** 2025-12-14  
**Статус:** Актуальная (добавлен модуль объяснимости)

---

## 🎯 Обзор архитектуры

Система разделена на **3 категории**:

1. **Исследовательские notebooks** (01-06) - для экспериментов и анализа
2. **Production модули** (features/, models/, scripts/) - готовый код для использования
3. **Данные и конфигурация** (data/, config/, reports/) - данные и настройки

**Принцип:** Notebooks → Production модули → Скрипты

---

## 📂 Полная структура директорий

```
ML/
│
├── 📜 scripts/                    # Исполняемые скрипты
│   ├── run_full_pipeline.py      # Главный пайплайн (Features → Training → Inference)
│   ├── validate_model.py         # Валидация модели
│   ├── compare_models.py          # Сравнение результатов экспериментов
│   ├── test_explanation.py        # Тестирование модуля объяснимости
│   └── README.md                 # Документация скриптов
│
├── 📓 notebooks/                  # Jupyter ноутбуки (исследования)
│   ├── 01_data_loading.ipynb     # Загрузка и предобработка данных
│   └── plots.ipynb               # Визуализация
│
├── 🔬 02_feature_engineering/    # Исследование признаков
│   ├── 01_volatility_features.ipynb
│   ├── 02_volume_features.ipynb
│   ├── 03_market_features.ipynb
│   ├── 04_trend_features.ipynb
│   ├── 05_targets.ipynb
│   └── 06_feature_aggregator.ipynb
│
├── 🤖 03_models/                  # Исследование моделей
│   ├── 01_baseline_models.ipynb
│   ├── 02_garch_model.ipynb
│   ├── 03_lightgbm_quantile.ipynb
│   ├── 04_ensemble_model.ipynb
│   ├── train_global_model.py     # Production: обучение модели
│   └── inference.py              # Production: прогнозирование
│
├── 📊 04_backtesting/             # Бэктестинг
│   ├── 01_signals_and_channels.ipynb
│   ├── 02_backtest_engine.ipynb
│   ├── 03_analysis_results.ipynb
│   └── run_backtest_pipeline.py   # Production: полный бэктест
│
├── 🔍 05_explainability/          # Объяснимость моделей
│   ├── 01_shap_explainer.ipynb
│   └── 02_feature_importance.ipynb
│
├── 🛠️ 06_utils/                   # Утилиты
│   ├── 01_helpers.ipynb
│   └── 02_validation_testing.ipynb
│
├── ⚙️ features/                    # Production: генерация признаков
│   ├── feature_builder.py        # Главный модуль (объединяет все признаки)
│   ├── volatility_features.py    # Признаки волатильности
│   ├── volume_features.py        # Признаки объёмов
│   ├── market_features.py        # Рыночные признаки
│   ├── trend_features.py         # Трендовые признаки
│   ├── calendar_features.py      # Календарные признаки
│   ├── intraday_features.py      # Внутридневные признаки (H1)
│   └── Loaders/                  # Загрузчики данных
│       ├── load_prices.py
│       ├── load_hourly.py
│       └── price_cleaner.py
│
├── 🧠 models/                     # Production: модели
│   └── ensemble.py               # Ансамбль LightGBM + GARCH
│
├── 📈 backtest/                   # Production: бэктестинг
│   └── __init__.py
│
├── 🔬 explainability/             # Production: объяснимость
│   ├── __init__.py
│   ├── shap_wrapper.py            # Обертка для SHAP объяснений
│   └── text_generator.py          # Генератор текстовых объяснений
│
├── ⚙️ config/                      # Конфигурация
│   ├── training_config.py        # Параметры обучения (ГЛАВНЫЙ ФАЙЛ!)
│   ├── config.py                 # Общие настройки
│   └── tickers_metadata.json     # Метаданные тикеров
│
├── 💾 data/                        # Данные
│   ├── MOEX_DATA/                # Исходные данные MOEX
│   ├── processed/                # Обработанные OHLCV + log_returns
│   ├── processed_ml/             # Данные с ML признаками
│   ├── backtest/                 # Данные для бэктестинга
│   └── models/                   # Обученные модели и прогнозы
│
├── 📊 reports/                     # Отчёты и результаты
│   ├── validation_report.csv     # Отчёт о валидации
│   ├── validation_detailed.csv   # Детальная статистика по тикерам
│   ├── feature_importance.csv    # Важность признаков
│   └── feature_importance.png    # График важности
│
├── 🛠️ tools/                       # Вспомогательные скрипты
│   ├── start_jupyter.bat
│   └── start_jupyter.ps1
│
├── 📚 docs/                        # Документация
│   ├── SYSTEM_STRUCTURE.md       # Этот файл
│   ├── SYSTEM_GUIDE.md           # Гайд по работе системы
│   ├── DEVELOPMENT_ROADMAP.md    # План разработки
│   └── CURRENT_STATUS.md         # Текущие проблемы и метрики
│
├── requirements.txt               # Зависимости Python
└── venv/                          # Виртуальное окружение
```

---

## 🔄 Потоки данных

### 1. Загрузка и предобработка
```
data/MOEX_DATA/{TICKER}/1D/*.csv
    ↓
notebooks/01_data_loading.ipynb
    ↓
data/processed/{TICKER}_ohlcv_returns.parquet
```

### 2. Генерация признаков
```
data/processed/*.parquet
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

### 3. Обучение модели
```
data/processed_ml/*.parquet
    ↓
scripts/run_full_pipeline.py
    ↓
03_models/train_global_model.py
    ↓
data/models/global_lgbm_q*.txt
reports/validation_report.csv
```

### 4. Прогнозирование
```
data/processed_ml/{TICKER}_ml_features.parquet
    ↓
03_models/inference.py
    ├── Прогноз квантилей (q16, q50, q84)
    └── Опционально: объяснения через explainability/
    ↓
data/models/{TICKER}_predictions.csv
JSON с прогнозами и объяснениями (для фронтенда)
```

---

## 📦 Production модули

### features/ - Генерация признаков

| Модуль | Функции | Признаки |
|--------|---------|----------|
| `volatility_features.py` | RV, EWMA, Parkinson, Garman-Klass | ~15 признаков |
| `volume_features.py` | Volume Profile, POC, Value Area | ~10 признаков |
| `market_features.py` | Beta, Correlation, Index Vol | ~8 признаков |
| `trend_features.py` | SMA/EMA, Momentum, RSI | ~15 признаков |
| `calendar_features.py` | Дни недели, месяцы, сезонность | ~10 признаков |
| `intraday_features.py` | H1 признаки (IVR, OPM, VDS) | ~8 признаков |
| `feature_builder.py` | Объединяет все модули | **~66 признаков** |

### models/ - Модели

| Модуль | Описание |
|--------|----------|
| `ensemble.py` | Ансамбль LightGBM + GARCH |
| `03_models/train_global_model.py` | Обучение квантильных моделей |
| `03_models/inference.py` | Прогнозирование |

### scripts/ - Исполняемые скрипты

| Скрипт | Назначение |
|--------|------------|
| `run_full_pipeline.py` | Полный цикл: Features → Training → Inference |
| `validate_model.py` | Валидация качества модели |
| `compare_models.py` | Сравнение результатов экспериментов |
| `test_explanation.py` | Тестирование модуля объяснимости и JSON контракта |

### explainability/ - Объяснимость моделей

| Модуль | Описание |
|--------|----------|
| `shap_wrapper.py` | Обертка для SHAP TreeExplainer с оптимизацией производительности |
| `text_generator.py` | Генератор текстовых объяснений на русском языке |

---

## ⚙️ Конфигурация

### config/training_config.py - Главный файл настроек

**Что настраивается:**
- Train/Test split (дата cutoff)
- Гиперпараметры LightGBM
- Квантили для прогноза
- Исключаемые тикеры
- Пресеты конфигурации

**Пресеты:**
- `BASELINE` - базовая конфигурация (60/40 split)
- `MORE_TRAIN` - больше train данных (70/30 split)
- `REGULARIZED` - сильная регуляризация
- `NO_TICKER` - без ticker_id признака

---

## 📊 Выходные данные

### reports/ - Отчёты

| Файл | Описание |
|------|----------|
| `validation_report.csv` | Основные метрики валидации |
| `validation_detailed.csv` | Детальная статистика по тикерам |
| `feature_importance.csv` | Важность признаков |
| `feature_importance.png` | График важности |

### data/models/ - Модели и прогнозы

| Файл | Описание |
|------|----------|
| `global_lgbm_q16.txt` | Модель для квантиля 0.16 |
| `global_lgbm_q50.txt` | Модель для квантиля 0.50 (медиана) |
| `global_lgbm_q84.txt` | Модель для квантиля 0.84 |
| `{TICKER}_predictions.csv` | Прогнозы для тикера |

---

## 🔗 Связанные документы

- [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) - Гайд по работе системы
- [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) - План разработки
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - Текущие проблемы и метрики

