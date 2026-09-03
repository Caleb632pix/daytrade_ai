import os
from flask import Flask, jsonify, request

from data import fetch_intraday
from features import add_features, FEATURE_COLUMNS
from labels import add_target
from model import train_model

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "service": "daytrade_ai",
        "message": "DayTrade AI is running"
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.get("/predict")
def predict():
    ticker = request.args.get("ticker", "AAPL").upper()

    try:
        # 1. Get recent 1-minute market data
        raw = fetch_intraday(
            ticker,
            interval="1m",
            period="5d"
        )

        # 2. Create technical features
        featured = add_features(raw)

        # 3. Create historical training labels
        labeled = add_target(
            featured,
            horizon_bars=15,
            threshold_pct=0.003
        )

        # 4. Train model using historical data
        model = train_model(
            labeled,
            feature_cols=FEATURE_COLUMNS
        )

        # 5. Get the latest available market row
        latest = featured.dropna(
            subset=FEATURE_COLUMNS
        ).iloc[-1]

        # 6. Ask the model for probability
        probability = float(
            model.predict_proba(
                latest[FEATURE_COLUMNS].to_frame().T
            )[0][1]
        )

        # 7. Convert probability into a simple signal
        if probability >= 0.65:
            signal = "BUY"
        elif probability <= 0.35:
            signal = "WAIT"
        else:
            signal = "WAIT"

        return jsonify({
            "status": "success",
            "ticker": ticker,
            "price": float(latest["close"]),
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 2),
            "signal": signal,
            "target": ">= 0.3% rise",
            "horizon": "15 minutes",
            "data_interval": "1 minute"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "ticker": ticker,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port
            )
