from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from pathlib import Path
import sys
import traceback
import pandas as pd
import numpy as np


def _calculate_confidence(interval_width: float, pred_q16: float, pred_q84: float) -> float:
    """
    Calculate confidence based on prediction interval width.
    
    interval_width is the difference between q84 and q16 (in relative terms, e.g., 0.02 = 2%).
    Confidence should be inversely related to interval width:
    - Narrow interval (low uncertainty) -> high confidence
    - Wide interval (high uncertainty) -> low confidence
    
    Args:
        interval_width: Width of prediction interval (q84 - q16)
        pred_q16: Lower quantile prediction
        pred_q84: Upper quantile prediction
    
    Returns:
        Confidence value between 0.0 and 1.0
    """
    if interval_width <= 0:
        return 0.5  # Default confidence if interval is invalid
    
    # Normalize interval_width: typical values are 0.01-0.10 (1%-10%)
    # Confidence = 1.0 - normalized_width, clamped to [0.0, 1.0]
    # For interval_width = 0.02 (2%), confidence should be high (~0.8-0.9)
    # For interval_width = 0.10 (10%), confidence should be low (~0.0-0.2)
    
    # Normalize: divide by typical max width (0.15 = 15% daily volatility is very high)
    normalized_width = min(interval_width / 0.15, 1.0)
    confidence = max(0.0, min(1.0, 1.0 - normalized_width))
    
    return float(confidence)

# ensure ML package paths
ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))

app = FastAPI(title="MOEXScanner Local Model Server")


class Candle(BaseModel):
    timestamp: Optional[str]
    open: float
    close: float
    high: float
    low: float
    volume: float


class PredictRequest(BaseModel):
    ticker: str
    candles: List[Candle]
    timeframe: Optional[str] = "24"
    horizon: Optional[int] = 5
    date: Optional[str] = None


# global model holder
MODEL = None
FEATURE_NAMES = None


def build_dataframe(candles: List[Dict[str, Any]], payload_date: Optional[str], timeframe: str) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    # try multiple date columns
    date_col = None
    for col in ["date", "timestamp", "begin", "time", "datetime"]:
        if col in df.columns:
            date_col = col
            break
    if date_col is not None:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    if df.get("date") is None or df["date"].isna().all():
        # reconstruct from payload_date if possible
        try:
            last_date = pd.to_datetime(payload_date)
        except Exception:
            last_date = None
        if last_date is not None and len(df) > 0:
            freq = "D" if str(timeframe).startswith("24") else "H"
            dates = pd.date_range(end=last_date, periods=len(df), freq=freq)
            df["date"] = dates
        else:
            # fallback to integer index
            df["date"] = pd.RangeIndex(start=0, stop=len(df))

    # ensure required numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # compute log_return if missing
    if "log_return" not in df.columns:
        # sort by date if possible
        try:
            df = df.sort_values("date")
        except Exception:
            pass
        df["log_return"] = np.log(df["close"]).diff().fillna(0)

    return df


@app.on_event("startup")
def load_models_on_startup():
    global MODEL, FEATURE_NAMES
    try:
        from inference import GlobalQuantileModel
        
        # Проверяем доступность ансамбля
        try:
            from models.ensemble import EnsembleModel
            ENSEMBLE_AVAILABLE = True
        except ImportError:
            ENSEMBLE_AVAILABLE = False
            print("[serve_model] Warning: Ensemble module not available, using LightGBM only")
        
        # Инициализируем модель с ансамблем (если доступен)
        # Ансамбль комбинирует LightGBM (70%) + GARCH (30%)
        MODEL = GlobalQuantileModel(
            use_ensemble=ENSEMBLE_AVAILABLE,
            ensemble_weights={'lgbm': 0.7, 'garch': 0.3} if ENSEMBLE_AVAILABLE else None
        )
        MODEL.load_models()
        FEATURE_NAMES = MODEL.feature_names
        
        if ENSEMBLE_AVAILABLE:
            print("[serve_model] ✅ Models loaded successfully (Ensemble: LightGBM + GARCH)")
        else:
            print("[serve_model] ✅ Models loaded successfully (LightGBM only)")
    except Exception as e:
        print(f"[serve_model] Failed to load models at startup: {e}")
        traceback.print_exc()


@app.post("/predict_local")
def predict_local(req: PredictRequest):
    try:
        # build dataframe
        df = build_dataframe([c.dict() for c in req.candles], req.date, req.timeframe)

        # build features
        try:
            from features.feature_builder import build_all_features
        except Exception:
            # try alternative path
            from ML.features.feature_builder import build_all_features  # type: ignore

        ml_features, backtest = build_all_features(df, req.ticker, include_intraday=False)

        # ensure model loaded
        global MODEL
        if MODEL is None:
            from inference import GlobalQuantileModel
            try:
                from models.ensemble import EnsembleModel
                ENSEMBLE_AVAILABLE = True
            except ImportError:
                ENSEMBLE_AVAILABLE = False
            
            MODEL = GlobalQuantileModel(
                use_ensemble=ENSEMBLE_AVAILABLE,
                ensemble_weights={'lgbm': 0.7, 'garch': 0.3} if ENSEMBLE_AVAILABLE else None
            )
            MODEL.load_models()

        # predict last row
        # Align features with model.feature_names and ensure proper dtypes
        feature_names = getattr(MODEL, "feature_names", None)
        X_full = ml_features.tail(1).reset_index(drop=True)

        if feature_names is not None and len(feature_names) > 0:
            # create X with exactly the model features (fill missing with 0)
            X = pd.DataFrame(index=X_full.index)
            for f in feature_names:
                if f in X_full.columns:
                    X[f] = X_full[f]
                else:
                    # missing feature -> fill with 0
                    X[f] = 0
        else:
            X = X_full.copy()

        # cast categorical features to category if model declares them
        cat_features = getattr(MODEL, "CATEGORICAL_FEATURES", [])
        for cf in cat_features:
            if cf in X.columns:
                try:
                    X[cf] = X[cf].astype('category')
                except Exception:
                    # fallback: convert to string then category
                    X[cf] = X[cf].astype(str).astype('category')

        # ensure numeric columns are numeric and fill NaN
        for col in X.columns:
            if str(X[col].dtype) not in ['category', 'bool']:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        # Используем ансамблевый прогноз (LightGBM + GARCH), если доступен
        # Иначе используем только LightGBM
        if MODEL.use_ensemble and MODEL.ensemble is not None:
            # Ансамблевый прогноз: комбинирует LightGBM и GARCH
            preds = MODEL.predict_ensemble(X, returns=df['log_return'], return_components=True)
        else:
            # Только LightGBM прогноз
            preds = MODEL.predict(X, return_interval=True)
        
        pred_row = preds.iloc[0].to_dict()

        current_price = float(df.sort_values("date").iloc[-1]["close"])

        # build explanation/top_features using feature importance fallback
        explanation_payload = {"text": "", "top_features": []}
        try:
            imp_df = MODEL.get_feature_importance(top_n=5)
            total_imp = float(imp_df['importance'].sum()) if not imp_df.empty else 0.0
            top_features = []
            for _, row in imp_df.head(5).iterrows():
                fname = row['feature']
                importance = float(row['importance'])
                value = None
                if fname in X.columns:
                    try:
                        value = float(X[fname].iloc[0])
                    except Exception:
                        value = None
                contribution = (importance / total_imp) if total_imp > 0 else 0.0
                top_features.append({
                    'name': fname,
                    'value': value,
                    'contribution': contribution
                })
            explanation_payload['text'] = 'Top features by importance'
            explanation_payload['top_features'] = top_features
        except Exception:
            # keep empty explanation_payload on failure
            pass

        # Модель предсказывает аннуализированную волатильность (стандартное отклонение)
        # Необходимо деаннуализировать для горизонта прогноза
        pred_q16 = float(pred_row.get('pred_q16', 0))  # Нижний квантиль волатильности
        pred_q50 = float(pred_row.get('pred_q50', 0))  # Медианная волатильность
        pred_q84 = float(pred_row.get('pred_q84', 0))  # Верхний квантиль волатильности
        
        # Деаннуализация: используем правило "квадратного корня времени"
        # vol_horizon = vol_annual * sqrt(horizon / 252)
        # 252 - количество торговых дней в году
        horizon = req.horizon if req.horizon else 5  # По умолчанию 5 дней
        time_factor = np.sqrt(horizon / 252.0)
        
        vol_horizon_q16 = pred_q16 * time_factor
        vol_horizon_q50 = pred_q50 * time_factor
        vol_horizon_q84 = pred_q84 * time_factor
        
        # Волатильность - это мера разброса (magnitude), а не направление
        # Строим симметричные ценовые границы вокруг текущей цены
        
        # Верхние границы (оптимистичный сценарий / сопротивление):
        pred_price_high_vol_up = current_price * (1.0 + vol_horizon_q84)    # Верхняя граница с высокой волатильностью
        pred_price_median_up = current_price * (1.0 + vol_horizon_q50)      # Верхняя граница с медианной волатильностью
        
        # Нижние границы (пессимистичный сценарий / поддержка):
        pred_price_high_vol_down = current_price * (1.0 - vol_horizon_q84)  # Нижняя граница с высокой волатильностью
        pred_price_median_down = current_price * (1.0 - vol_horizon_q50)    # Нижняя граница с медианной волатильностью
        
        # Маппинг для обратной совместимости с downstream логикой:
        # pred_price_q84 используется как целевая цена (Target proxy)
        # pred_price_q16 используется как стоп-лосс (Stop Loss proxy)
        pred_price_q84 = pred_price_high_vol_up      # Верхняя граница канала
        pred_price_q16 = pred_price_high_vol_down    # Нижняя граница канала
        pred_price_q50 = current_price                # Медиана остаётся на текущей цене

        # ========================================================================
        # ПРОФЕССИОНАЛЬНАЯ ТОРГОВАЯ ЛОГИКА (Mean Reversion + Trend Following)
        # ========================================================================
        
        # Шаг A: Определение направления тренда
        # Используем медианную волатильность (pred_q50) для определения тренда
        # Если медианный прогноз показывает движение вверх от текущей цены, тренд восходящий
        trend_is_up = pred_q50 > 0  # pred_q50 - это волатильность, но проверяем её положительность
        
        # Альтернативно: используем сравнение медианной верхней границы с текущей ценой
        # Это более надёжный индикатор направления тренда
        trend_is_up = pred_price_median_up > current_price
        
        # Дополнительная проверка: если доступны SMA/EMA признаки, используем их для подтверждения
        # (опционально, если признаки есть в X)
        try:
            if 'sma_50' in X.columns:
                close_price = current_price  # Текущая цена закрытия
                sma_50 = float(X['sma_50'].iloc[0])
                if sma_50 > 0:
                    trend_confirmed = close_price > sma_50
                    # Если тренд не подтверждается техническими индикаторами, снижаем уверенность
                    if trend_is_up != trend_confirmed:
                        # Используем волатильность как основной сигнал, но отмечаем расхождение
                        pass
        except Exception:
            pass  # Если признаки недоступны, используем только pred_q50
        
        # Шаг B: Расчёт умной точки входа (Лимитный ордер)
        # НЕ входим по рыночной цене! Ждём откат/отскок
        
        if trend_is_up:
            # BUY сценарий: ждём отката (dip) для покупки
            # Входим на середине между текущей ценой и нижней границей волатильности
            entry = round((current_price + pred_price_high_vol_down) / 2.0, 2)
            action_candidate = "BUY"
        else:
            # SELL сценарий: ждём ралли для продажи
            # Входим на середине между текущей ценой и верхней границей волатильности
            entry = round((current_price + pred_price_high_vol_up) / 2.0, 2)
            action_candidate = "SELL"
        
        # Шаг C: Установка Target и Stop Loss
        
        if action_candidate == "BUY":
            # Для BUY:
            # Target - верхняя граница волатильности (сопротивление)
            target = round(pred_price_high_vol_up, 2)
            # Stop Loss - чуть ниже нижней границы волатильности (0.5% запас)
            stop_loss = round(pred_price_high_vol_down * 0.995, 2)
        else:  # SELL
            # Для SELL:
            # Target - нижняя граница волатильности (поддержка)
            target = round(pred_price_high_vol_down, 2)
            # Stop Loss - чуть выше верхней границы волатильности (0.5% запас)
            stop_loss = round(pred_price_high_vol_up * 1.005, 2)
        
        # Шаг D: Фильтр Risk/Reward соотношения
        # Проверяем, что потенциальная прибыль > 1.2 * потенциальный риск
        
        potential_profit = abs(target - entry)
        potential_risk = abs(entry - stop_loss)
        
        # Защита от деления на ноль или слишком малых значений
        if potential_risk < 0.01 or vol_horizon_q84 < 0.001:
            # Волатильность слишком мала или риск нулевой - не торгуем
            action = "HOLD"
            reason = "Волатильность слишком низкая или нулевой риск"
        elif potential_profit < (1.2 * potential_risk):
            # R/R ratio < 1.2 - не торгуем
            action = "HOLD"
            rr_ratio = round(potential_profit / potential_risk, 2) if potential_risk > 0 else 0
            reason = f"Низкое соотношение R/R: {rr_ratio:.2f} (требуется >= 1.2)"
        else:
            # R/R ratio приемлемый - выставляем сигнал
            action = action_candidate
            rr_ratio = round(potential_profit / potential_risk, 2)
            trend_direction = "восходящий" if trend_is_up else "нисходящий"
            reason = f"Умный вход в канале волатильности ({trend_direction} тренд, R/R={rr_ratio:.2f})"

        # 7. Контекст объёма (Volume Context)
        # Извлекаем признаки объёма из фичей для фронтенда
        try:
            volume_zscore = float(X['volume_zscore_20'].iloc[0]) if 'volume_zscore_20' in X.columns else 0.0
        except (KeyError, IndexError, TypeError):
            volume_zscore = 0.0
        
        try:
            volume_spike_raw = X['volume_spike'].iloc[0] if 'volume_spike' in X.columns else 0
            # Конвертируем numpy bool/int в native Python bool для JSON
            spike_detected = bool(int(volume_spike_raw))
        except (KeyError, IndexError, TypeError):
            spike_detected = False
        
        # poc_distance: используем placeholder 0.0 (vp_position может быть категориальным)
        poc_distance = 0.0
        
        # va_position: определяем позицию цены относительно Value Area
        try:
            vp_above_va = X['vp_above_va'].iloc[0] if 'vp_above_va' in X.columns else 0
            va_position = "above" if int(vp_above_va) == 1 else "inside/below"
        except (KeyError, IndexError, TypeError):
            va_position = "inside/below"
        
        response = {
            "ticker": req.ticker,
            "horizon": req.horizon,
            "predicted_volatility": {
                "median": float(pred_row.get('pred_q50', 0)),
                "lower_1sigma": float(pred_row.get('pred_q16', 0)),
                "upper_1sigma": float(pred_row.get('pred_q84', 0)),
                "lower_2sigma": float(pred_row.get('pred_q16', 0) - pred_row.get('interval_width', 0)),
                "upper_2sigma": float(pred_row.get('pred_q84', 0) + pred_row.get('interval_width', 0)),
            },
            "confidence": _calculate_confidence(pred_row.get('interval_width', 0), pred_q16, pred_q84),
            "channel": {
                "upper_2sigma": float(pred_price_q84 + pred_row.get('interval_width', 0)),
                "upper_1sigma": float(pred_price_q84),
                "current_price": current_price,
                "lower_1sigma": float(pred_price_q16),
                "lower_2sigma": float(pred_price_q16 - pred_row.get('interval_width', 0)),
            },
            "trading_signal": {
                "action": action,
                "entry": float(entry),
                "target": target,
                "stop_loss": stop_loss,
                "position_size": 0.1,
                "reason": reason
            },
            "volume_context": {
                "zscore": volume_zscore,
                "spike_detected": spike_detected,
                "poc_distance": poc_distance,
                "va_position": va_position
            },
            "raw_prediction": pred_row
        }
        # attach explanation
        response['explanation'] = explanation_payload

        return response

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("ML.scripts.serve_model:app", host="127.0.0.1", port=8000, reload=False)
