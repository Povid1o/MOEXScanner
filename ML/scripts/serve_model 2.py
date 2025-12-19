from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
import pandas as pd
import numpy as np

# Local ML imports
import sys
ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))
sys.path.insert(0, str(ML_ROOT))

# Try importing the inference module from the ML package
try:
    from inference import GlobalQuantileModel
except Exception:
    try:
        from ML.03_models.inference import GlobalQuantileModel  # type: ignore
    except Exception as e:
        raise ImportError(f"Cannot import GlobalQuantileModel: {e}")


app = FastAPI(title="MOEXScanner Local Model")


class Candle(BaseModel):
    timestamp: str
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


@app.post("/predict_local")
def predict_local(req: PredictRequest):
    try:
        # Convert candles to dataframe
        rows = [c.dict() for c in req.candles]
        df = pd.DataFrame(rows)

        # Rename/ensure columns expected by feature builder
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])

        # Ensure required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing column: {col}")

        # Simple preprocessing: compute log_return if missing
        if 'log_return' not in df.columns:
            df = df.sort_values('date')
            df['log_return'] = np.log(df['close']).diff().fillna(0)

        # Import feature builder dynamically to avoid import path issues
        try:
            from features.feature_builder import build_all_features
        except Exception as e:
            # Try alternative path
            from ML.features.feature_builder import build_all_features  # type: ignore

        # Build ML features
        ml_features, backtest = build_all_features(df, req.ticker, include_intraday=False)

        # Load model (singleton per process would be better)
        model = GlobalQuantileModel()
        model.load_models()

        # Predict for the last available row
        X = ml_features.tail(1).reset_index(drop=True)
        preds = model.predict(X, return_interval=True)

        # Prepare JSON response similar to previous AI schema (simplified)
        pred_row = preds.iloc[0]

        current_price = float(df.sort_values('date').iloc[-1]['close'])

        response = {
            "ticker": req.ticker,
            "horizon": req.horizon,
            "predicted_volatility": {
                "median": float(pred_row['pred_q50']),
                "lower_1sigma": float(pred_row['pred_q16']),
                "upper_1sigma": float(pred_row['pred_q84']),
                "lower_2sigma": float(pred_row['pred_q16'] - pred_row['interval_width']),
                "upper_2sigma": float(pred_row['pred_q84'] + pred_row['interval_width']),
            },
            "confidence": max(0.0, min(1.0, 1.0 - float(pred_row['interval_width']) / (abs(current_price) + 1e-9))),
            "trend": {
                "direction": ("uptrend" if X['log_return'].iloc[-1] > 0 else "downtrend" if X['log_return'].iloc[-1] < 0 else "sideways"),
                "confidence": "medium",
                "strength": float(min(1.0, abs(X['log_return'].iloc[-1]) * 10))
            },
            "channel": {
                "upper_2sigma": float(pred_row['pred_q84'] + pred_row['interval_width']),
                "upper_1sigma": float(pred_row['pred_q84']),
                "current_price": current_price,
                "lower_1sigma": float(pred_row['pred_q16']),
                "lower_2sigma": float(pred_row['pred_q16'] - pred_row['interval_width'])
            },
            "trading_signal": {
                "action": ("BUY" if current_price < pred_row['pred_q16'] else ("SELL" if current_price > pred_row['pred_q84'] else "HOLD")),
                "entry": current_price,
                "target": float(current_price * (1.0 + 0.05)),
                "stop_loss": float(current_price * (1.0 - 0.03)),
                "position_size": 0.1,
                "reason": "Автоматическое правило: сравнение текущей цены и оценочных квантилей"
            },
            "tail_risk": {
                "warning": False,
                "probability": 0.01,
                "expected_loss": None
            },
            "volume_context": {
                "zscore": 0.0,
                "spike_detected": False,
                "poc_distance": 0.0,
                "va_position": "inside"
            },
            "explanation": {
                "text": "Местные прогнозы на основе обученных квантильных моделей LightGBM",
                "top_features": []
            }
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # Run with: python ML/scripts/serve_model.py
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
