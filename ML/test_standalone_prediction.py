"""
Автономный скрипт для тестирования ML модели без Backend/Frontend
Делает прогноз на GAZP на 9 дней
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Добавляем пути к модулям
ML_ROOT = Path(__file__).parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))

print("=" * 80)
print("🧪 ТЕСТИРОВАНИЕ ML МОДЕЛИ (STANDALONE)")
print("=" * 80)
print()

# Шаг 1: Загрузка модели
print("📦 Загрузка модели...")
try:
    from models.ensemble import EnsembleModel
    from ML.models.ensemble import GlobalQuantileModel
except ImportError:
    # Альтернативный импорт
    sys.path.insert(0, str(ML_ROOT / "models"))
    from inference import GlobalQuantileModel

model = GlobalQuantileModel(
    use_ensemble=True,
    ensemble_weights={'lgbm': 0.7, 'garch': 0.3}
)
model.load_models()
print(f"✅ Модель загружена: {model.feature_names[:5]}... ({len(model.feature_names)} признаков)")
print()

# Шаг 2: Загрузка актуальных данных с MOEX API
print("📊 Загрузка актуальных данных с MOEX API...")
ticker = "GAZP"
target_date = datetime(2025, 12, 19)  # 19.12.2025 (вчерашняя дата)
from_date = target_date - timedelta(days=365)  # Последние 365 дней

print(f"   Тикер: {ticker}")
print(f"   Дата прогноза: {target_date.strftime('%Y-%m-%d')}")
print(f"   Период данных: {from_date.strftime('%Y-%m-%d')} - {target_date.strftime('%Y-%m-%d')}")
print()

# Запрос к MOEX History API для акций
# Для обычных акций используем boards/TQBR, для индексов - boards/SNDX
url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
params = {
    "from": from_date.strftime("%Y-%m-%d"),
    "till": target_date.strftime("%Y-%m-%d"),
    "limit": 10000  # Максимум записей
}

print(f"🌐 Запрос к MOEX API: {url}")
print(f"   Параметры: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # MOEX API возвращает данные в структуре: {"history": {"columns": [...], "data": [...]}}
    if "history" not in data:
        raise ValueError("Не найдена секция 'history' в ответе API")
    
    history = data["history"]
    columns = history.get("columns", [])
    rows = history.get("data", [])
    
    if not rows:
        raise ValueError("Нет данных в ответе API")
    
    print(f"✅ Получено {len(rows)} записей с MOEX API")
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # Переименовываем столбцы для совместимости
    column_mapping = {
        "TRADEDATE": "date",
        "OPEN": "open",
        "CLOSE": "close",
        "HIGH": "high",
        "LOW": "low",
        "VOLUME": "volume"
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Преобразуем дату
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "TRADEDATE" in df.columns:
        df["date"] = pd.to_datetime(df["TRADEDATE"])
    else:
        raise ValueError("Не найден столбец с датой")
    
    # Сортируем по дате
    df = df.sort_values("date")
    
    # Вычисляем log_return
    df['log_return'] = np.log(df['close']).diff().fillna(0)
    
    # Удаляем строки с NaN в критических столбцах
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    print(f"📈 Данные обработаны: {len(df)} записей")
    print(f"   Период: {df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Последняя цена: {df['close'].iloc[-1]:.2f} ₽")
    print()
    
except Exception as e:
    print(f"❌ Ошибка при загрузке данных с MOEX API: {e}")
    print("📝 Используем резервный вариант: загрузка из локального файла...")
    print()
    
    # Резервный вариант: загрузка из файла
    data_file = ML_ROOT / "data" / "MOEX_DATA" / ticker / "1D" / f"{ticker}_daily_2020-10-12_to_2025-10-11.csv"
    
    if data_file.exists():
        print(f"✅ Загружен файл: {data_file.name}")
        df = pd.read_csv(data_file)
        
        if 'date' not in df.columns:
            if 'begin' in df.columns:
                df['date'] = pd.to_datetime(df['begin'])
            elif 'end' in df.columns:
                df['date'] = pd.to_datetime(df['end'])
            else:
                df['date'] = pd.date_range(end=target_date, periods=len(df), freq='D')
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        df = df.sort_values('date')
        
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close']).diff().fillna(0)
        
        print(f"📈 Данные: {len(df)} записей с {df['date'].min()} по {df['date'].max()}")
    else:
        raise ValueError(f"Не удалось загрузить данные ни с API, ни из файла {data_file}")

print()

# Шаг 3: Генерация признаков
print("🔧 Генерация ML признаков...")
try:
    from features.feature_builder import build_all_features
except ImportError:
    from ML.features.feature_builder import build_all_features

ml_features, backtest = build_all_features(df, ticker, include_intraday=False)
print(f"✅ Признаки сгенерированы: {ml_features.shape}")
print(f"   ML признаков: {ml_features.shape[1]}")
print()

# Шаг 4: Подготовка данных для модели
print("🎯 Подготовка данных для прогноза...")
X_full = ml_features.tail(1).reset_index(drop=True)

# Выравниваем признаки с моделью
X = pd.DataFrame(index=X_full.index)
for f in model.feature_names:
    if f in X_full.columns:
        X[f] = X_full[f]
    else:
        X[f] = 0  # Заполняем нулями отсутствующие признаки

# Приводим категориальные признаки
cat_features = getattr(model, "CATEGORICAL_FEATURES", [])
for cf in cat_features:
    if cf in X.columns:
        try:
            X[cf] = X[cf].astype('category')
        except Exception:
            X[cf] = X[cf].astype(str).astype('category')

# Преобразуем числовые признаки
for col in X.columns:
    if str(X[col].dtype) not in ['category', 'bool']:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

print(f"✅ Данные подготовлены: {X.shape}")
print()

# Шаг 5: Прогнозирование
print("🚀 ЗАПУСК ПРОГНОЗИРОВАНИЯ...")
print(f"   Тикер: {ticker}")
print(f"   Горизонт: 9 дней")
print()

try:
    # Передаем только последнюю строку returns
    returns_for_garch = df['log_return'].tail(1)
    
    preds = model.predict_ensemble(
        X,
        returns=returns_for_garch,
        return_components=True
    )
    
    pred_row = preds.iloc[0]
    current_price = float(df['close'].iloc[-1])
    
    print("=" * 80)
    print("✅ РЕЗУЛЬТАТЫ ПРОГНОЗА")
    print("=" * 80)
    print()
    print(f"📊 Текущая цена: {current_price:.2f} ₽")
    print()
    print("🎯 Прогноз волатильности (аннуализированная):")
    print(f"   • Нижний квантиль (q16): {pred_row['pred_q16']:.4f}")
    print(f"   • Медиана (q50):         {pred_row['pred_q50']:.4f}")
    print(f"   • Верхний квантиль (q84): {pred_row['pred_q84']:.4f}")
    print(f"   • Ширина интервала:      {pred_row['interval_width']:.4f}")
    print()
    
    # Деаннуализация для 9 дней
    horizon = 9
    time_factor = np.sqrt(horizon / 252.0)
    
    vol_9d_q16 = pred_row['pred_q16'] * time_factor
    vol_9d_q50 = pred_row['pred_q50'] * time_factor
    vol_9d_q84 = pred_row['pred_q84'] * time_factor
    
    print(f"🗓️ Прогноз волатильности на {horizon} дней:")
    print(f"   • q16: {vol_9d_q16:.4f} ({vol_9d_q16*100:.2f}%)")
    print(f"   • q50: {vol_9d_q50:.4f} ({vol_9d_q50*100:.2f}%)")
    print(f"   • q84: {vol_9d_q84:.4f} ({vol_9d_q84*100:.2f}%)")
    print()
    
    # Ценовые каналы
    price_upper = current_price * (1 + vol_9d_q84)
    price_lower = current_price * (1 - vol_9d_q84)
    price_median_up = current_price * (1 + vol_9d_q50)
    price_median_down = current_price * (1 - vol_9d_q50)
    
    print(f"💰 Ценовые уровни через {horizon} дней:")
    print(f"   • Верхняя граница (84%): {price_upper:.2f} ₽")
    print(f"   • Медиана верх (50%):    {price_median_up:.2f} ₽")
    print(f"   • Текущая цена:          {current_price:.2f} ₽")
    print(f"   • Медиана низ (50%):     {price_median_down:.2f} ₽")
    print(f"   • Нижняя граница (84%):  {price_lower:.2f} ₽")
    print()
    
    print("=" * 80)
    print("✅ ТЕСТ УСПЕШНО ЗАВЕРШЁН!")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ ОШИБКА ПРИ ПРОГНОЗИРОВАНИИ: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

