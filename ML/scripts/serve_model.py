from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from pathlib import Path
import sys
import traceback
import pandas as pd
import numpy as np

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
        # instantiate and load models
        MODEL = GlobalQuantileModel()
        MODEL.load_models()
        FEATURE_NAMES = MODEL.feature_names
        print("[serve_model] Models loaded successfully")
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
            MODEL = GlobalQuantileModel()
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

        # determine action based on predicted quantiles
        action = ("BUY" if current_price < pred_row.get('pred_q16', 0) else ("SELL" if current_price > pred_row.get('pred_q84', 0) else "HOLD"))

        # set target and stop loss depending on action direction
        if action == "BUY":
            target = float(current_price * (1.0 + 0.05))
            stop_loss = float(current_price * (1.0 - 0.03))
        elif action == "SELL":
            target = float(current_price * (1.0 - 0.05))
            stop_loss = float(current_price * (1.0 + 0.03))
        else:
            target = None
            stop_loss = None

        # interpret model quantiles as relative predictions (returns/volatility)
        pred_q16 = float(pred_row.get('pred_q16', 0))
        pred_q50 = float(pred_row.get('pred_q50', 0))
        pred_q84 = float(pred_row.get('pred_q84', 0))
        pred_price_q16 = current_price * (1.0 + pred_q16)
        pred_price_q50 = current_price * (1.0 + pred_q50)
        pred_price_q84 = current_price * (1.0 + pred_q84)

        # Decide action by comparing current price with predicted price quantiles
        action = ("BUY" if pred_price_q84 > current_price else ("SELL" if pred_price_q16 < current_price else "HOLD"))

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
            "confidence": max(0.0, min(1.0, 1.0 - float(pred_row.get('interval_width', 0)) / (abs(current_price) + 1e-9))),
            "channel": {
                "upper_2sigma": float(pred_price_q84 + pred_row.get('interval_width', 0)),
                "upper_1sigma": float(pred_price_q84),
                "current_price": current_price,
                "lower_1sigma": float(pred_price_q16),
                "lower_2sigma": float(pred_price_q16 - pred_row.get('interval_width', 0)),
            },
            "trading_signal": {
                "action": action,
                "entry": current_price,
                "target": (float(current_price * (1.0 + 0.05)) if action == "BUY" else (float(current_price * (1.0 - 0.05)) if action == "SELL" else None)),
                "stop_loss": (float(current_price * (1.0 - 0.03)) if action == "BUY" else (float(current_price * (1.0 + 0.03)) if action == "SELL" else None)),
                "position_size": 0.1,
                "reason": "automatic-rule: compare current price with predicted price quantiles"
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
