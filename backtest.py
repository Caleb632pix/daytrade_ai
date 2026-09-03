"""
backtest.py — Walk-forward backtest with risk management as a first-class rule,
not an afterthought.

Trade entry logic mirrors your own spec (#10):
    Enter LONG only if:
        - model probability > prob_threshold
        - price above sma20 (trend confirmation)
        - relative volume confirms the move (rel_volume > vol_threshold)
        - risk/reward >= min_rr
    Otherwise: no trade.

Every trade uses a fixed stop-loss and profit target derived from ATR (so risk
scales with current volatility, not a flat dollar amount), and position size is
derived from a fixed max-risk-per-trade rule, not gut feel.

Includes commission + slippage — a strategy that only wins before costs isn't
a strategy.
"""

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    prob_threshold: float = 0.65,
    vol_threshold: float = 1.2,
    min_rr: float = 2.0,
    atr_stop_mult: float = 1.0,
    atr_target_mult: float = 2.0,   # target = 2x the stop distance -> RR of 2 by construction
    account_size: float = 10_000,
    risk_per_trade_pct: float = 0.01,   # risk 1% of account per trade
    commission_per_share: float = 0.0,
    slippage_pct: float = 0.0005,       # 0.05% assumed slippage per fill
    max_hold_bars: int = 15,
) -> dict:
    df = df.reset_index(drop=True)
    trades = []
    equity = account_size
    equity_curve = [equity]

    i = 0
    while i < len(df) - max_hold_bars:
        row = df.iloc[i]

        entry_signal = (
            row.get("pred_proba", 0) > prob_threshold
            and row["close"] > row["sma20"]
            and row.get("rel_volume", 0) > vol_threshold
        )

        if not entry_signal:
            equity_curve.append(equity)
            i += 1
            continue

        entry_price = row["close"] * (1 + slippage_pct)
        stop_distance = row["atr14"] * atr_stop_mult
        target_distance = row["atr14"] * atr_target_mult
        rr = target_distance / stop_distance if stop_distance > 0 else 0

        if rr < min_rr or stop_distance <= 0 or np.isnan(stop_distance):
            equity_curve.append(equity)
            i += 1
            continue

        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance

        # --- Position sizing from fixed fractional risk ---
        risk_dollars = equity * risk_per_trade_pct
        shares = max(int(risk_dollars / stop_distance), 0)
        if shares == 0:
            equity_curve.append(equity)
            i += 1
            continue

        # --- Walk forward bar-by-bar to find exit ---
        exit_price, exit_reason = None, None
        for j in range(i + 1, min(i + 1 + max_hold_bars, len(df))):
            bar = df.iloc[j]
            if bar["low"] <= stop_price:
                exit_price, exit_reason = stop_price, "stop"
                break
            if bar["high"] >= target_price:
                exit_price, exit_reason = target_price, "target"
                break
        if exit_price is None:
            exit_price = df.iloc[min(i + max_hold_bars, len(df) - 1)]["close"]
            exit_reason = "timeout"

        exit_price *= (1 - slippage_pct)
        gross_pnl = (exit_price - entry_price) * shares
        costs = commission_per_share * shares * 2  # entry + exit
        net_pnl = gross_pnl - costs

        equity += net_pnl
        equity_curve.append(equity)

        trades.append({
            "entry_idx": i, "entry_price": entry_price, "exit_price": exit_price,
            "shares": shares, "exit_reason": exit_reason, "pnl": net_pnl,
        })

        i += max_hold_bars  # avoid overlapping positions; move past this trade's window

    return summarize(trades, equity_curve, account_size)


def summarize(trades: list, equity_curve: list, account_size: float) -> dict:
    if not trades:
        return {"num_trades": 0, "message": "No trades met entry criteria in this window."}

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max
    max_drawdown = drawdown.min()

    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 390)) if returns.std() > 0 else float("nan")
    # 252*390 annualizes assuming ~390 1-min bars/trading day — adjust if using a different bar interval

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "profit_factor": (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "total_pnl": pnls.sum(),
        "final_equity": equity_curve[-1],
        "return_pct": (equity_curve[-1] - account_size) / account_size,
        "max_drawdown_pct": max_drawdown,
        "sharpe_estimate": sharpe,
    }
