"""
data.py — Intraday OHLCV data fetching.

Prototype source: yfinance (free, no API key, ~7-60 days of intraday history).
Production source: swap `fetch_intraday` internals for Alpaca's market data API
(alpaca-py) once you have an account — same function signature, same output shape,
so nothing downstream (features.py, model.py, backtest.py) needs to change.
"""

import pandas as pd


def fetch_intraday(ticker: str, interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    """
    Fetch intraday OHLCV data.

    Parameters
    ----------
    ticker : e.g. "AAPL"
    interval : "1m", "5m", "15m" (yfinance limits 1m data to the last 7 days)
    period : how far back, e.g. "5d", "7d", "60d" (60d max for 5m/15m bars)

    Returns
    -------
    DataFrame indexed by timestamp with columns: open, high, low, close, volume
    """
    import yfinance as yf

    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/interval/period limits.")

    # yfinance sometimes returns MultiIndex columns for single tickers depending on version
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "timestamp"
    return df


def fetch_alpaca_intraday(ticker: str, timeframe: str, start: str, end: str,
                           api_key: str, secret_key: str) -> pd.DataFrame:
    """
    Placeholder for production data source. Requires `pip install alpaca-py`
    and a free Alpaca account (paper trading keys work for historical data too).

    Fill this in when you're ready to move off yfinance — output must match
    fetch_intraday()'s shape: DataFrame[open, high, low, close, volume] indexed by timestamp.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(req).df
    bars = bars.reset_index().set_index("timestamp")
    return bars[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    df = fetch_intraday("AAPL", interval="1m", period="5d")
    print(df.tail())
    print(f"\n{len(df)} rows fetched.")
