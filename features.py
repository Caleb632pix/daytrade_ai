"""
features.py — Technical indicator feature engineering.

Deliberately uses a SMALL set of complementary signals rather than dumping in
every indicator that exists (correlated indicators add noise, not information):

  Trend:      SMA20/50, MACD
  Momentum:   RSI
  Volatility: ATR, Bollinger Band width
  Volume:     relative volume, VWAP distance
  Structure:  distance from recent high/low (breakout proximity)

All features are computed causally (no lookahead) — every value at time t only
uses data up to and including t.
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Trend ---
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["price_vs_sma20"] = (df["close"] - df["sma20"]) / df["sma20"]
    df["sma20_vs_sma50"] = (df["sma20"] - df["sma50"]) / df["sma50"]

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Momentum: RSI(14) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # --- Volatility: ATR(14) & Bollinger width ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_width"] = (4 * bb_std) / bb_mid  # (upper-lower)/mid

    # --- Volume ---
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["rel_volume"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

    # --- Structure: breakout proximity ---
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["dist_from_high20"] = (df["high_20"] - df["close"]) / df["close"]
    df["dist_from_low20"] = (df["close"] - df["low_20"]) / df["close"]

    return df


FEATURE_COLUMNS = [
    "price_vs_sma20", "sma20_vs_sma50", "macd_hist",
    "rsi14", "atr_pct", "bb_width",
    "rel_volume", "price_vs_vwap",
    "dist_from_high20", "dist_from_low20",
]
