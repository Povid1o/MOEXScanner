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
        # interpret model quantiles as relative predictions (returns/volatility)
        # convert to predicted price quantiles to compare with current price
        pred_q16 = float(pred_row.get('pred_q16', 0))
        pred_q50 = float(pred_row.get('pred_q50', 0))
        pred_q84 = float(pred_row.get('pred_q84', 0))
        pred_price_q16 = current_price * (1.0 + pred_q16)
        pred_price_q50 = current_price * (1.0 + pred_q50)
        pred_price_q84 = current_price * (1.0 + pred_q84)

        # Decide action by comparing current price with predicted price quantiles
        action = ("BUY" if pred_price_q84 > current_price else ("SELL" if pred_price_q16 < current_price else "HOLD"))

        response = {
            "ticker": ticker,
from pathlib import Path
import traceback

ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))

def main():
    try:
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
        from inference import GlobalQuantileModel

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

        # Load model and predict
        model = GlobalQuantileModel()
        model.load_models()

        X = ml_features.tail(1).reset_index(drop=True)
        preds = model.predict(X, return_interval=True)
        pred_row = preds.iloc[0].to_dict()

        current_price = float(df.sort_values('date').iloc[-1]['close'])

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
            "confidence": max(0.0, min(1.0, 1.0 - float(pred_row.get('interval_width', 0)) / (abs(current_price) + 1e-9))),
            "channel": {
                "upper_2sigma": float(pred_row.get('pred_q84', 0) + pred_row.get('interval_width', 0)),
                "upper_1sigma": float(pred_row.get('pred_q84', 0)),
                "current_price": current_price,
                "lower_1sigma": float(pred_row.get('pred_q16', 0)),
                "lower_2sigma": float(pred_row.get('pred_q16', 0) - pred_row.get('interval_width', 0)),
            },
            "trading_signal": {
                "action": action,
                "entry": current_price,
                "target": target,
                "stop_loss": stop_loss,
                "position_size": 0.1,
                "reason": "automatic-rule: compare current price with predicted quantiles"
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
