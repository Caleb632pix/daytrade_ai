"""
main.py — Orchestrates: data -> features -> labels -> walk-forward model -> backtest.

Run with real data:
    python main.py --ticker AAPL

Run with synthetic data (no internet needed — validates the pipeline logic):
    python main.py --synthetic
"""

import argparse
import numpy as np
import pandas as pd

from features import add_features, FEATURE_COLUMNS
from labels import add_target
from model import walk_forward_predict
from backtest import run_backtest


def make_synthetic_ohlcv(n_bars: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Random-walk-with-drift synthetic 1-min bars, purely for pipeline testing.
    Do NOT use this to judge whether the strategy is profitable — it has no
    real market structure. It only proves the code runs end-to-end without bugs."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.00002, scale=0.0015, size=n_bars)
    close = 100 * np.cumprod(1 + returns)

    high = close * (1 + np.abs(rng.normal(0, 0.001, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n_bars)))
    open_ = close + rng.normal(0, 0.05, n_bars)
    volume = rng.integers(1000, 50000, n_bars).astype(float)

    idx = pd.date_range("2026-08-01 09:30", periods=n_bars, freq="1min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def run_pipeline(df: pd.DataFrame):
    df = add_features(df)
    df = add_target(df, horizon_bars=15, threshold_pct=0.003)

    preds = walk_forward_predict(df, feature_cols=FEATURE_COLUMNS, n_folds=5, min_train_size=500)

    print(f"\nOut-of-sample rows: {len(preds)}")
    print(f"Base rate (target hit naturally): {preds['target'].mean():.3f}")
    print(f"Mean predicted probability: {preds['pred_proba'].mean():.3f}")

    results = run_backtest(preds)
    print("\n--- Backtest Results (out-of-sample, walk-forward) ---")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return preds, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--interval", type=str, default="1m")
    parser.add_argument("--period", type=str, default="5d")
    args = parser.parse_args()

    if args.synthetic or not args.ticker:
        print("Running on SYNTHETIC data (pipeline validation only, not a real strategy test)...")
        raw = make_synthetic_ohlcv()
    else:
        from data import fetch_intraday
        raw = fetch_intraday(args.ticker, interval=args.interval, period=args.period)

    run_pipeline(raw)
