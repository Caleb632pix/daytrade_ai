# Day-Trading Prediction Model + Backtester

## What this is
A first working slice of the pipeline you sketched: `Data → Features → Labels → Model → Backtest`.
It's deliberately simple (Random Forest, 10 features, single-ticker) so you can trust it,
debug it, and see where it actually has edge before adding complexity.

## Quick start
```bash
pip install -r requirements.txt

# Sanity-check the code with synthetic data (no internet, no real signal expected)
python main.py --synthetic

# Real data (yfinance 1-min bars only go back ~7 days)
python main.py --ticker AAPL --interval 1m --period 5d
```

## What it predicts
Not "will AAPL go up" — it predicts:
> Probability that price rises ≥0.3% at some point in the next 15 minutes.

Change this in `labels.py` (`horizon_bars`, `threshold_pct`) to match your actual holding period.

## Files
| File | Purpose |
|---|---|
| `data.py` | Fetches OHLCV bars (yfinance now, Alpaca stub for later) |
| `features.py` | 10 technical features — trend, momentum, volatility, volume, structure |
| `labels.py` | Defines the prediction target |
| `model.py` | Random Forest + walk-forward validation (no shuffled train/test — that's leakage) |
| `backtest.py` | Entry rules, ATR-based stop/target, fixed-fractional position sizing, costs |
| `main.py` | Runs the full pipeline end to end |

## What's NOT here yet (next steps, in priority order)
1. **Multi-ticker support** — right now it's one symbol at a time.
2. **Fundamental & macro features** — currently pure technicals. Adding sector
   relative strength or VIX level as a feature is a natural next step.
3. **News/catalyst integration** — this is where the fake-news scoring pipeline
   we built earlier plugs in: a verified-news sentiment score could become
   another feature column, or a hard filter (e.g. don't trade on Tier 3/4 sources).
4. **Live paper-trading loop** — connect `data.py`'s Alpaca stub to Alpaca's
   paper-trading order API and run this on a schedule instead of a one-shot backtest.
5. **Hyperparameter tuning + walk-forward optimization** — current settings
   (prob_threshold=0.65, ATR multipliers, etc.) are reasonable starting points,
   not tuned. Tune them on a validation slice, never on the final test slice.

## Before risking real money
- Run `--synthetic` only to confirm the code works — it proves nothing about
  edge, since the data has no real structure.
- On real data, check `win_rate`, `profit_factor`, and `max_drawdown_pct`
  across *multiple tickers and multiple time windows*, not just one lucky run.
- Forward-test on Alpaca's paper trading for weeks before going live.
- No model here should ever get a direct, unreviewed path to order execution —
  keep a human or a hard rule-based gate (like the fake-news risk filter) between
  a signal and a live trade.
