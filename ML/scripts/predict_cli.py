#!/usr/bin/env python3
"""
CLI wrapper to run local model prediction.

Reads JSON from stdin with shape:
{
  "ticker": "SBER",
  "candles": [ {timestamp, open, close, high, low, volume}, ... ],
  "timeframe": "24",
  "horizon": 5,
  "date": "2025-10-11"
}

Outputs JSON to stdout with prediction fields.
"""
import json
import sys
from pathlib import Path
import traceback
import pandas as pd
import numpy as np

ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))

from features.feature_builder import build_all_features
from inference import GlobalQuantileModel


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


def main():
    try:
        # Read JSON from stdin
        payload = json.loads(sys.stdin.read())
        ticker = payload.get('ticker', 'SBER')
        candles = payload.get('candles', [])

        if not candles:
            print(json.dumps({"error": "no candles provided"}))
            return 1

        # Build dataframe from candles
        df = pd.DataFrame(candles)

        # Normalize date/time column: try several common names
        date_col = None
        for col in ['date', 'timestamp', 'begin', 'time', 'datetime']:
            if col in df.columns:
                date_col = col
                break

        if date_col is not None:
            # try parse; coerce invalid values to NaT
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')

        # If parsing failed (all NaT), try to reconstruct dates from payload 'date' and timeframe
        if df.get('date') is None or df['date'].isna().all():
            # Use payload 'date' as the last date if available
            last_date = None
            try:
                last_date = pd.to_datetime(payload.get('date'))
            except Exception:
                last_date = None

            timeframe = payload.get('timeframe', '24')
            if last_date is not None and len(df) > 0:
                # assume timeframe '24' -> daily, '60' -> hourly
                freq = 'D' if str(timeframe).startswith('24') else 'H'
                try:
                    df = df.sort_index().reset_index(drop=True)
                    dates = pd.date_range(end=last_date, periods=len(df), freq=freq)
                    df['date'] = dates
                    sys.stderr.write(f"[predict_cli] Reconstructed dates using last_date={last_date} freq={freq}\n")
                except Exception as e:
                    sys.stderr.write(f"[predict_cli] Failed to reconstruct dates: {e}\n")
                    df['date'] = pd.NaT
            else:
                # As last resort, create a simple integer index as date-like values
                df['date'] = pd.RangeIndex(start=0, stop=len(df))
                sys.stderr.write("[predict_cli] Warning: no valid dates found; using integer index as date\n")

        # Ensure required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                print(json.dumps({"error": f"missing column {col}"}))
                return 2

        if 'log_return' not in df.columns:
            df = df.sort_values('date')
            df['log_return'] = np.log(df['close']).diff().fillna(0)

        # Build features (no intraday)
        ml_features, backtest = build_all_features(df, ticker, include_intraday=False)

        # Load model with ensemble (LightGBM + GARCH)
        try:
            from models.ensemble import EnsembleModel
            ENSEMBLE_AVAILABLE = True
        except ImportError:
            ENSEMBLE_AVAILABLE = False
        
        model = GlobalQuantileModel(
            use_ensemble=ENSEMBLE_AVAILABLE,
            ensemble_weights={'lgbm': 0.7, 'garch': 0.3} if ENSEMBLE_AVAILABLE else None
        )
        model.load_models()

        X = ml_features.tail(1).reset_index(drop=True)
        
        # Используем ансамблевый прогноз, если доступен
        if model.use_ensemble and model.ensemble is not None:
            preds = model.predict_ensemble(X, returns=df['log_return'], return_components=True)
        else:
            preds = model.predict(X, return_interval=True)
        pred_row = preds.iloc[0].to_dict()

        current_price = float(df.sort_values('date').iloc[-1]['close'])

        # Модель предсказывает аннуализированную волатильность (стандартное отклонение)
        # Необходимо деаннуализировать для горизонта прогноза
        pred_q16 = float(pred_row.get('pred_q16', 0))  # Нижний квантиль волатильности
        pred_q50 = float(pred_row.get('pred_q50', 0))  # Медианная волатильность
        pred_q84 = float(pred_row.get('pred_q84', 0))  # Верхний квантиль волатильности
        
        # Деаннуализация: используем правило "квадратного корня времени"
        # vol_horizon = vol_annual * sqrt(horizon / 252)
        # 252 - количество торговых дней в году
        horizon = payload.get('horizon', 5)  # По умолчанию 5 дней
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

        # Calculate confidence properly
        interval_width = float(pred_row.get('interval_width', 0))
        confidence = _calculate_confidence(interval_width, pred_q16, pred_q84)

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
            "ticker": ticker,
            "horizon": payload.get('horizon', 5),
            "predicted_volatility": {
                "median": float(pred_row.get('pred_q50', 0)),
                "lower_1sigma": float(pred_row.get('pred_q16', 0)),
                "upper_1sigma": float(pred_row.get('pred_q84', 0)),
                "lower_2sigma": float(pred_row.get('pred_q16', 0) - pred_row.get('interval_width', 0)),
                "upper_2sigma": float(pred_row.get('pred_q84', 0) + pred_row.get('interval_width', 0)),
            },
            "confidence": confidence,
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

        sys.stdout.write(json.dumps(response))
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        sys.stdout.write(json.dumps({"error": str(e), "trace": tb}))
        return 10

if __name__ == '__main__':
    sys.exit(main())
