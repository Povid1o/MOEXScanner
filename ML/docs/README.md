# 📚 Документация ML Pipeline

**Версия:** 2.1  
**Дата обновления:** 2025-01-XX (добавлен модуль объяснимости)

---

## 📖 Основные документы

### 1. [SYSTEM_STRUCTURE.md](SYSTEM_STRUCTURE.md)
**Полная структура текущей системы**

Содержит:
- Полную структуру директорий
- Описание всех модулей
- Потоки данных
- Production модули и их функции

**Когда читать:** Для понимания архитектуры системы

---

### 2. [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)
**Гайд по взаимосвязям и работе системы**

Содержит:
- Как работает система (пошагово)
- Взаимосвязи между модулями
- Типичные сценарии использования
- Потоки данных (детально)
- Конфигурация и параметры
- Отладка и диагностика

**Когда читать:** Для понимания как система работает и как её использовать

---

### 3. [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)
**Роудмап девелопера - что осталось сделать**

Содержит:
- Текущий статус компонентов
- Приоритетные задачи (критичные, важные, желательные)
- Детальный план разработки
- Метрики успеха
- Процесс разработки

**Когда читать:** Для планирования разработки и понимания что делать дальше

---

### 4. [CURRENT_STATUS.md](CURRENT_STATUS.md)
**Текущие проблемы системы и актуальные метрики**

Содержит:
- Актуальные метрики модели
- Критические проблемы и их решения
- Предупреждения
- Что работает хорошо
- Рекомендации по улучшению

**Когда читать:** Для понимания текущего состояния системы и проблем

---

## 🚀 Быстрый старт

### 1. Настройка окружения

```bash
cd ML
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Запуск полного pipeline

```bash
# Генерация признаков + Обучение + Инференс
python scripts/run_full_pipeline.py

# Только обучение (features готовы)
python scripts/run_full_pipeline.py --skip-features --preset MORE_TRAIN
```

### 3. Валидация модели

```bash
python scripts/validate_model.py
```

### 4. Сравнение моделей

```bash
# Сохраните старый отчёт
copy reports\validation_report.csv reports\validation_report_baseline.csv

# Обучите новую модель
python scripts/run_full_pipeline.py --skip-features --preset REGULARIZED

# Сравните
python scripts/compare_models.py
```

---

## 📁 Структура проекта

```
ML/
├── scripts/          # Исполняемые скрипты
├── features/         # Production: генерация признаков
├── models/           # Production: модели
├── 03_models/       # Обучение и инференс
├── config/           # Конфигурация (training_config.py - главный!)
├── data/             # Данные
├── reports/          # Отчёты и результаты
└── docs/             # Документация (эта папка)
```

**Подробнее:** [SYSTEM_STRUCTURE.md](SYSTEM_STRUCTURE.md)

---

## ⚙️ Конфигурация

**Все параметры обучения в одном файле:** `config/training_config.py`

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

## 📊 Текущие метрики

**Модель:** MORE_TRAIN (70/30 split, cutoff: 2024-06-01)

| Метрика | Значение | Статус |
|---------|----------|--------|
| Coverage 68% | 66.0% | ✅ |
| MAE | 0.120 | ⚠️ |
| Correlation | 0.489 | ❌ |
| Bias | +4.14% | ⚠️ |

**Подробнее:** [CURRENT_STATUS.md](CURRENT_STATUS.md)

---

## 🔧 Основные скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/run_full_pipeline.py` | Полный цикл: Features → Training → Inference |
| `scripts/validate_model.py` | Валидация качества модели |
| `scripts/compare_models.py` | Сравнение результатов экспериментов |
| `scripts/test_explanation.py` | Тестирование модуля объяснимости и JSON контракта |

**Подробнее:** `scripts/README.md`

---

## 🔍 Объяснимость прогнозов (NEW!)

Система поддерживает генерацию текстовых объяснений прогнозов на русском языке через SHAP.

**Использование:**
```python
from inference import GlobalQuantileModel

model = GlobalQuantileModel()
model.load_models()

result = model.predict(
    data.tail(1),
    include_explanation=True,
    background_data=data.tail(100)
)

print(result['explanation']['text'])
```

**Модули:**
- `explainability/shap_wrapper.py` - SHAP объяснения
- `explainability/text_generator.py` - Генератор текстов

**Тестирование:**
```bash
python scripts/test_explanation.py
```

**Подробнее:** См. раздел "Объяснимость" в [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)

---

## 📚 Дополнительная информация

- **Структура системы:** [SYSTEM_STRUCTURE.md](SYSTEM_STRUCTURE.md)
- **Как работает:** [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)
- **Что делать дальше:** [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)
- **Текущие проблемы:** [CURRENT_STATUS.md](CURRENT_STATUS.md)

---

## ⚠️ Важные замечания

1. **Всегда используйте `--skip-features`** если признаки уже готовы
2. **Сохраняйте старые отчёты** перед новыми экспериментами
3. **Параметры настраиваются в `config/training_config.py`**
4. **Модель деградирует на новых данных** - нужен rolling retraining

---

**Последнее обновление:** 2025-12-06
