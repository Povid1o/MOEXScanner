"""
FastAPI-сервис-адаптер между ML Core (GlobalQuantileModel из ML/03_models/inference.py)
и внешним миром (Go-бэкенд или другие клиенты).

Основная задача:
- Принять простой запрос по тикеру
- Загрузить последнюю строку фич для тикера
- Вызвать модель для квантильного прогноза и объяснения
- Преобразовать результат в удобный для Go контракт PredictionResponse

Все ключевые комментарии — на русском, как просили.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Пути и импорт ML Core
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ML"

# Добавляем директорию с inference.py в sys.path
# Важно: каталог называется "03_models", поэтому импортируем как модуль "inference"
MODELS_DIR = ML_DIR / "03_models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from inference import GlobalQuantileModel  # type: ignore
except ImportError as e:  # pragma: no cover - защита от окружения без ML
    raise RuntimeError(
        "Не удалось импортировать GlobalQuantileModel из ML/03_models/inference.py. "
        "Убедитесь, что структура проекта сохранена."
    ) from e


# ---------------------------------------------------------------------------
# Pydantic-модели данных (контракты ответа)
# ---------------------------------------------------------------------------


class PredictedVolatility(BaseModel):
    """Квантильный прогноз волатильности."""

    median: float          # q50
    lower_1sigma: float    # q16
    upper_1sigma: float    # q84
    lower_2sigma: float    # q50 - 2*(q50-q16)
    upper_2sigma: float    # q50 + 2*(q84-q50)


class Trend(BaseModel):
    """Упрощённое направление тренда."""

    direction: str         # "UP", "DOWN", "SIDE"
    confidence: str        # "HIGH", "MEDIUM", "LOW"
    strength: float        # 0.0–1.0 (насколько выражен тренд)


class TradingSignal(BaseModel):
    """Правил-based торговый сигнал на основе тренда и волатильности."""

    action: str            # "BUY", "SELL", "HOLD", "WAIT"
    entry: float           # Цена входа
    target: float          # Целевая цена (обычно верхний сигма-уровень)
    stop_loss: float       # Стоп-лосс (нижний сигма-уровень)
    reason: str            # Человеко-понятное объяснение


class TailRisk(BaseModel):
    """Оценка хвостового риска."""

    warning: bool
    probability: float     # Оценочная вероятность экстремального движения
    expected_loss: float   # Потенциальный размер потерь


class FeatureContribution(BaseModel):
    """Описание вклада отдельного признака в прогноз."""

    name: str
    impact: str            # "positive" или "negative"
    description: str       # Читабельное текстовое описание
    value: float           # Числовая величина вклада


class Explanation(BaseModel):
    """Высокоуровневая текстовая сводка + топ признаков."""

    summary: str
    top_features: List[FeatureContribution]


class PredictionResponse(BaseModel):
    """Основной контракт ответа сервиса-адаптера."""

    ticker: str
    date: str
    current_price: float
    volatility: PredictedVolatility
    trend: Trend
    signal: TradingSignal
    tail_risk: TailRisk
    explanation: Explanation


# Входной контракт (минимальный) — тикер
class PredictionRequest(BaseModel):
    ticker: str


# ---------------------------------------------------------------------------
# Инициализация FastAPI и глобальной модели
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Adapter API",
    description="Сервис-адаптер между ML Core и Go-бэкендом",
    version="0.1.0",
)

# Разрешаем CORS для удобства интеграции (Go, фронт и т.п.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный объект модели, загружаем один раз при старте приложения
global_model: Optional[GlobalQuantileModel] = None

# Директория с parquet-фичами
FEATURES_DIR = ML_DIR / "data" / "processed_ml"


@app.on_event("startup")
def load_global_model() -> None:
    """Загрузка обученных моделей при старте сервиса.

    Делается один раз, затем объект модели переиспользуется для всех запросов.
    """
    global global_model

    model = GlobalQuantileModel()
    model.load_models()

    global_model = model
    print("✅ GlobalQuantileModel успешно загружена")


# ---------------------------------------------------------------------------
# Вспомогательные функции адаптера
# ---------------------------------------------------------------------------


def _find_features_file_for_ticker(ticker: str) -> Path:
    """Поиск parquet-файла с фичами по тикеру.

    Стратегия:
    1) <TICKER>_ml_features.parquet
    2) Любой parquet в каталоге, содержащий имя тикера в названии
    """
    ticker = ticker.upper()

    direct = FEATURES_DIR / f"{ticker}_ml_features.parquet"
    if direct.exists():
        return direct

    candidates = sorted(FEATURES_DIR.glob(f"*{ticker}*.parquet"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Не найден parquet-файл с фичами для тикера {ticker}")


def _load_last_feature_row(ticker: str) -> pd.Series:
    """Загружает последнюю (по времени) строку фич для тикера.

    Если в таблице есть столбцы 'date' или 'timestamp', сортируем по ним,
    иначе используем естественный порядок индекса.
    """
    features_path = _find_features_file_for_ticker(ticker)
    df = pd.read_parquet(features_path)

    if "ticker" in df.columns:
        df_ticker = df[df["ticker"].str.upper() == ticker.upper()]
        if df_ticker.empty:
            df_ticker = df
    else:
        df_ticker = df

    # Определяем столбец сортировки
    sort_col = None
    for candidate in ("date", "timestamp", "DATETIME"):
        if candidate in df_ticker.columns:
            sort_col = candidate
            break

    if sort_col is not None:
        df_ticker = df_ticker.sort_values(sort_col)

    # Берём последнюю строку
    if df_ticker.empty:
        raise ValueError(f"Таблица фич для тикера {ticker} пуста")

    return df_ticker.iloc[-1]


def _extract_current_price(row: pd.Series) -> float:
    """Извлекает текущую цену из признаков.

    Ищем по нескольким возможным названиям колонок,
    если не находим — используем запасной вариант.
    """
    candidates = [
        "close",
        "close_price",
        "price_close",
        "last_price",
        "adj_close",
    ]
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])

    # Если ничего не нашли — пробуем 'open'
    if "open" in row.index and pd.notna(row["open"]):
        return float(row["open"])

    # Совсем без цены — возвращаем 0.0 (но в реальном проде лучше падать с ошибкой)
    return 0.0


def _build_volatility(pred_q16: float, pred_q50: float, pred_q84: float) -> PredictedVolatility:
    """Строим объект волатильности и 2-сигма уровни по линейной экстраполяции."""
    lower_2 = pred_q50 - 2.0 * (pred_q50 - pred_q16)
    upper_2 = pred_q50 + 2.0 * (pred_q84 - pred_q50)
    return PredictedVolatility(
        median=pred_q50,
        lower_1sigma=pred_q16,
        upper_1sigma=pred_q84,
        lower_2sigma=lower_2,
        upper_2sigma=upper_2,
    )


def _infer_trend(row: pd.Series) -> Trend:
    """Простейшая логика определения тренда.

    Приоритет:
    1) Если есть SMA/MA 50 (sma_50, ma_50 и т.п.) — сравниваем Close с ними
    2) Иначе — сравниваем Close с Open
    """
    close = float(row.get("close") or row.get("close_price") or row.get("last_price") or 0.0)
    open_price = float(row.get("open") or 0.0)

    sma_50 = None
    for col in ("sma_50", "ma_50", "sma_50d", "ma_50d"):
        if col in row.index and pd.notna(row[col]):
            sma_50 = float(row[col])
            break

    direction = "SIDE"
    strength = 0.3

    if sma_50 is not None and sma_50 > 0:
        diff = (close - sma_50) / sma_50
        if diff > 0.002:
            direction = "UP"
        elif diff < -0.002:
            direction = "DOWN"
        else:
            direction = "SIDE"

        # Нормируем силу тренда в диапазон [0, 1]
        strength = float(np.clip(abs(diff) / 0.05, 0.0, 1.0))
    elif open_price > 0:
        diff = (close - open_price) / open_price
        if diff > 0.001:
            direction = "UP"
        elif diff < -0.001:
            direction = "DOWN"
        else:
            direction = "SIDE"
        strength = float(np.clip(abs(diff) / 0.03, 0.0, 1.0))

    # Грубая шкала уверенности по силе тренда
    if strength >= 0.66:
        confidence = "HIGH"
    elif strength >= 0.33:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return Trend(direction=direction, confidence=confidence, strength=strength)


def _build_signal(
    current_price: float,
    vol: PredictedVolatility,
    trend: Trend,
) -> TradingSignal:
    """Правила формирования торгового сигнала.

    Основные правила:
    - Если q84 > 0.05 → WAIT (слишком высокая ожидаемая дневная волатильность)
    - Если тренд UP и волатильность умеренная → BUY
    - Иначе → HOLD
    """
    q16 = vol.lower_1sigma
    q50 = vol.median
    q84 = vol.upper_1sigma

    interval_width = q84 - q16

    # Базовые значения цен (если current_price == 0, всё равно считаем)
    entry = current_price
    target = current_price
    stop_loss = current_price

    # Слишком высокая ожидаемая дневная волатильность — лучше подождать
    if q84 > 0.05:
        action = "WAIT"
        reason = "Ожидается экстремальная волатильность (q84 > 5% дневного движения)"
    else:
        # Оцениваем «умеренность» волатильности
        moderate_vol = interval_width <= 0.03

        if trend.direction == "UP" and moderate_vol:
            action = "BUY"
            # Вход — небольшой откат от текущей цены
            entry = current_price * (1.0 - max(vol.lower_1sigma, 0.0))
            # Цель — движение к верхнему 2-сигма уровню
            target = current_price * (1.0 + max(vol.upper_2sigma, 0.0))
            # Стоп — 1-сигма вниз
            stop_loss = current_price * (1.0 - max(vol.lower_1sigma, 0.0))
            reason = "Восходящий тренд с умеренной волатильностью, целимся в верхний 2-sigma уровень"
        else:
            action = "HOLD"
            reason = "Нет чёткого тренда или волатильность повышена — лучше наблюдать"
            entry = current_price
            target = current_price
            stop_loss = current_price * (1.0 - max(vol.lower_1sigma, 0.0))

    return TradingSignal(
        action=action,
        entry=float(entry),
        target=float(target),
        stop_loss=float(stop_loss),
        reason=reason,
    )


def _build_tail_risk(vol: PredictedVolatility, current_price: float) -> TailRisk:
    """Оценка хвостового риска по асимметрии интервала (q84 vs q16)."""
    q16 = vol.lower_1sigma
    q50 = vol.median
    q84 = vol.upper_1sigma

    downside = q50 - q16
    upside = q84 - q50

    warning = False
    if downside > 0 and upside > downside * 1.5:
        warning = True

    # Простая оценка вероятности хвостового события по ширине интервала
    interval_width = q84 - q16
    probability = float(np.clip(interval_width / 0.10, 0.0, 1.0))

    # Потенциальный убыток — пропорционален волатильности
    expected_loss = float(current_price * max(vol.upper_2sigma, vol.median))

    return TailRisk(
        warning=warning,
        probability=probability,
        expected_loss=expected_loss,
    )


def _build_explanation(
    raw_explanation: dict,
    top_n: int = 5,
) -> Explanation:
    """Преобразует raw_data из ML Core в список FeatureContribution.

    Ожидаемый формат raw_explanation:
    {
        "text": "...",
        "raw_data": [
            {"feature": "...", "value": ..., "contribution": ...},
            ...
        ]
    }
    """
    summary_text = ""
    top_features: List[FeatureContribution] = []

    if not raw_explanation:
        return Explanation(summary=summary_text, top_features=top_features)

    summary_text = str(raw_explanation.get("text", "")) or ""
    raw_data = raw_explanation.get("raw_data") or []

    # Нормализуем к списку словарей
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    normalized = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        name = (
            item.get("feature")
            or item.get("name")
            or item.get("feature_name")
        )
        if not name:
            continue

        contrib = item.get("contribution")
        if contrib is None:
            # Иногда вклад может называться value/impact_value — пробуем их
            contrib = item.get("value", 0.0)

        try:
            contrib_f = float(contrib)
        except (TypeError, ValueError):
            contrib_f = 0.0

        impact = "positive" if contrib_f >= 0 else "negative"
        description = f"Признак {name} оказывает {impact} влияние на прогноз волатильности."

        normalized.append(
            (abs(contrib_f), FeatureContribution(
                name=str(name),
                impact=impact,
                description=description,
                value=contrib_f,
            ))
        )

    # Сортируем по модулю вклада и берём top_n
    normalized.sort(key=lambda x: x[0], reverse=True)
    top_features = [fc for _, fc in normalized[:top_n]]

    return Explanation(summary=summary_text, top_features=top_features)


# ---------------------------------------------------------------------------
# Основной эндпоинт /predict
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Главный эндпоинт инференса.

    Вход:  {"ticker": "SBER"}
    Выход: PredictionResponse с волатильностью, трендом, сигналом и объяснением.
    """
    if global_model is None:
        # Теоретически не должно случиться: модель грузится в событии startup
        raise HTTPException(status_code=500, detail="ML-модель ещё не загружена")

    ticker = request.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Поле 'ticker' не должно быть пустым")

    try:
        # A) Загрузка последней строки фич
        feature_row = _load_last_feature_row(ticker)
    except FileNotFoundError as e:
        # Ticker/фичи не найдены — отвечаем 500 с понятным описанием
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке фич: {e}")

    # Готовим DataFrame для модели (одна строка)
    X = feature_row.to_frame().T
    X.index = [0]

    # B) Вызов модели с объяснением
    try:
        result = global_model.predict(
            X,
            return_interval=True,
            include_explanation=True,
            background_data=None,  # можно передать X_train, если есть под рукой
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка во время инференса модели: {e}")

    forecast_df: pd.DataFrame = result["forecast"]
    explanation_raw: dict = result.get("explanation", {})  # type: ignore[assignment]

    if forecast_df.empty:
        raise HTTPException(status_code=500, detail="Модель вернула пустой прогноз")

    # Берём первую (и единственную) строку прогноза
    pred_row = forecast_df.iloc[0]

    # Извлекаем квантили
    try:
        q16 = float(pred_row["pred_q16"])
        q50 = float(pred_row["pred_q50"])
        q84 = float(pred_row["pred_q84"])
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"В прогнозе отсутствует ожидаемое поле квантиля: {e}",
        )

    # C) Адаптер-логика: волатильность, тренд, сигнал, хвостовой риск, объяснение

    # Волатильность и 2-сигма
    volatility = _build_volatility(q16, q50, q84)

    # Текущая цена из фич
    current_price = _extract_current_price(feature_row)

    # Дата — берём из фич, если есть, либо пустую строку
    date_value = ""
    for col in ("date", "timestamp", "DATETIME"):
        if col in feature_row.index and pd.notna(feature_row[col]):
            date_value = str(feature_row[col])
            break

    # Тренд
    trend = _infer_trend(feature_row)

    # Торговый сигнал
    signal = _build_signal(current_price=current_price, vol=volatility, trend=trend)

    # Хвостовой риск
    tail_risk = _build_tail_risk(vol=volatility, current_price=current_price)

    # Объяснение
    explanation = _build_explanation(explanation_raw)

    return PredictionResponse(
        ticker=ticker,
        date=date_value,
        current_price=current_price,
        volatility=volatility,
        trend=trend,
        signal=signal,
        tail_risk=tail_risk,
        explanation=explanation,
    )


@app.get("/health")
def health() -> dict:
    """Простой health-check эндпоинт."""
    return {
        "status": "ok",
        "model_loaded": global_model is not None,
    }


if __name__ == "__main__":
    # Локальный запуск:
    #   uvicorn app:app --reload
    #
    # Здесь оставляем только подсказку, без автозапуска,
    # чтобы служба могла управляться внешними инструментами.
    print(
        "Запустите сервис командой:\n"
        "  uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
    )


