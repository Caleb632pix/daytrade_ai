"""
labels.py — Defines what the model actually predicts.

Per your point #10: never predict "will it go up". Predict something measurable:

    "Will price rise by >= threshold_pct within the next horizon_bars bars?"

This produces a binary classification target. The model outputs a PROBABILITY,
not a hard yes/no — that probability is what the backtest thresholds against.
"""

import pandas as pd


def add_target(df: pd.DataFrame, horizon_bars: int = 15, threshold_pct: float = 0.003) -> pd.DataFrame:
    """
    horizon_bars=15, threshold_pct=0.003 on 1-min bars means:
    "Does price rise >= 0.3% at any point in the next 15 minutes?"
    (Using max-forward-return, not just the close N bars later, since day
    trading usually exits on a favorable spike rather than waiting exactly N bars.)
    """
    df = df.copy()
    future_max = df["close"].shift(-1).rolling(horizon_bars, min_periods=1).max().shift(-(horizon_bars - 1))
    forward_return = (future_max - df["close"]) / df["close"]
    df["forward_return"] = forward_return
    df["target"] = (forward_return >= threshold_pct).astype(int)

    # Drop the last horizon_bars rows — their target is incomplete (lookahead not yet available)
    df = df.iloc[:-horizon_bars]
    return df
