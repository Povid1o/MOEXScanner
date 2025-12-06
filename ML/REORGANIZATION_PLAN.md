# 📁 План реорганизации структуры ML/

## 🎯 Цель
Создать понятную структуру, где легко найти нужные файлы.

## 📂 Новая структура

```
ML/
├── scripts/              # Все исполняемые скрипты
│   ├── run_full_pipeline.py
│   ├── run_experiment.py
│   ├── run_experiment.bat
│   ├── run_experiment.ps1
│   ├── validate_model.py
│   ├── compare_models.py
│   ├── analyze_feature_correlation.py
│   ├── example_usage.py
│   └── test_setup.py
│
├── notebooks/            # Все Jupyter ноутбуки
│   ├── 01_data_loading.ipynb
│   └── plots.ipynb
│
├── docs/                 # Вся документация
│   ├── README.md
│   ├── GUIDE.md
│   ├── QUICKSTART.md
│   ├── EXPERIMENTS_GUIDE.md
│   ├── PROJECT_STRUCTURE.md
│   ├── ARCHITECTURE_AUDIT.md
│   ├── DEVELOPMENT_ROADMAP.md
│   └── NOTEBOOKS_VS_PRODUCTION_AUDIT.md
│
├── config/               # Конфигурация (уже есть)
│   ├── training_config.py
│   ├── tickers_metadata.json
│   └── config.py (переместить сюда)
│
├── tools/                # Вспомогательные скрипты
│   ├── start_jupyter.bat
│   └── start_jupyter.ps1
│
├── 02_feature_engineering/  # (остаётся как есть)
├── 03_models/            # (остаётся как есть)
├── 04_backtesting/       # (остаётся как есть)
├── 05_explainability/    # (остаётся как есть)
├── 06_utils/             # (остаётся как есть)
│
├── features/             # Production модули (остаётся)
├── models/               # Production модули (остаётся)
├── utils/                # Production модули (остаётся)
├── explainability/       # Production модули (остаётся)
├── backtest/             # Production модули (остаётся)
│
├── data/                 # Данные (остаётся)
├── reports/              # Отчёты (остаётся)
├── venv/                 # Виртуальное окружение (остаётся)
│
├── requirements.txt      # (остаётся в корне)
└── __init__.py           # (остаётся в корне)
```

## 🔄 Что перемещается

### В scripts/:
- run_full_pipeline.py
- run_experiment.py
- run_experiment.bat
- run_experiment.ps1
- validate_model.py
- compare_models.py
- analyze_feature_correlation.py
- example_usage.py
- test_setup.py

### В notebooks/:
- 01_data_loading.ipynb
- plots.ipynb

### В docs/:
- README.md
- GUIDE.md
- QUICKSTART.md
- EXPERIMENTS_GUIDE.md
- PROJECT_STRUCTURE.md
- ARCHITECTURE_AUDIT.md
- DEVELOPMENT_ROADMAP.md
- NOTEBOOKS_VS_PRODUCTION_AUDIT.md

### В config/:
- config.py (из корня)

### В tools/:
- start_jupyter.bat
- start_jupyter.ps1

## ⚠️ Важно
После перемещения нужно обновить пути в скриптах!

