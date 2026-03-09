"""
backtesting/engine.py
──────────────────────
Event-driven backtesting engine for EarnML signal strategies.

Simulates trading based on ML model signals and computes:
  • Total return, annualised return
  • Sharpe ratio, Sortino ratio
  • Maximum drawdown, Calmar ratio
  • Win rate, profit factor
  • Per-trade log

Two strategies:
  1. SignalStrategy   — buy/sell based on composite ML signal
  2. EarningsStrategy — trade around predicted earnings beats

Why this matters for JPMorgan:
  Raw model accuracy is meaningless without understanding
  the *economic value* of the signal. A model that's 55%
  accurate but trades at the right times can beat a 65%
  accurate model with bad timing. Backtesting demonstrates
  you understand the full ML → value chain.

Usage:
  python backtesting/engine.py --ticker AAPL --strategy signal
"""

import sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker:      str
    entry_date:  pd.Timestamp
    entry_price: float
    direction:   int           # +1 = long, -1 = short
    exit_date:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    signal:      str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None: return 0.0
        return self.direction * (self.exit_price - self.entry_price) / self.entry_price

    @property
    def hold_days(self) -> int:
        if self.exit_date is None: return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    ticker:           str
    strategy:         str
    start_date:       str
    end_date:         str
    initial_capital:  float
    final_capital:    float
    trades:           list = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades: return 0.0
        return sum(1 for t in self.trades if t.pnl_pct > 0) / len(self.trades)

    def summary(self) -> dict:
        rets = [t.pnl_pct for t in self.trades]
        return {
            "ticker":         self.ticker,
            "strategy":       self.strategy,
            "total_return":   round(self.total_return * 100, 2),
            "n_trades":       self.n_trades,
            "win_rate":       round(self.win_rate * 100, 1),
            "avg_return":     round(np.mean(rets) * 100, 3) if rets else 0,
            "avg_hold_days":  round(np.mean([t.hold_days for t in self.trades]), 1) if self.trades else 0,
        }


# ─────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────

def compute_metrics(equity_curve: pd.Series, risk_free: float = 0.04) -> dict:
    """
    Compute full suite of performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve : daily portfolio value
    risk_free    : annual risk-free rate (default 4% = current T-bill)
    """
    daily_returns = equity_curve.pct_change().dropna()

    if len(daily_returns) < 2:
        return {"error": "insufficient data"}

    # ── Returns ──────────────────────────────────────────────────
    total_ret   = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years     = len(daily_returns) / 252
    ann_ret     = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    ann_vol     = daily_returns.std() * np.sqrt(252)

    # ── Sharpe ───────────────────────────────────────────────────
    rf_daily    = (1 + risk_free) ** (1/252) - 1
    excess      = daily_returns - rf_daily
    sharpe      = excess.mean() / (excess.std() + 1e-9) * np.sqrt(252)

    # ── Sortino (downside deviation only) ────────────────────────
    downside    = daily_returns[daily_returns < rf_daily]
    sortino     = excess.mean() / (downside.std() + 1e-9) * np.sqrt(252)

    # ── Max Drawdown ─────────────────────────────────────────────
    peaks       = equity_curve.cummax()
    drawdowns   = (equity_curve - peaks) / peaks
    max_dd      = drawdowns.min()
    dd_duration = _max_drawdown_duration(drawdowns)

    # ── Calmar = Ann Return / |Max Drawdown| ─────────────────────
    calmar      = ann_ret / (abs(max_dd) + 1e-9)

    # ── Win / Loss stats ─────────────────────────────────────────
    wins        = daily_returns[daily_returns > 0]
    losses      = daily_returns[daily_returns < 0]
    profit_factor = wins.sum() / (abs(losses.sum()) + 1e-9)

    return {
        "total_return":     round(total_ret * 100, 2),
        "annualised_return":round(ann_ret  * 100, 2),
        "annualised_vol":   round(ann_vol  * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "sortino_ratio":    round(sortino, 3),
        "calmar_ratio":     round(calmar, 3),
        "max_drawdown":     round(max_dd * 100, 2),
        "drawdown_duration_days": dd_duration,
        "profit_factor":    round(profit_factor, 3),
        "win_rate_daily":   round((daily_returns > 0).mean() * 100, 1),
        "best_day":         round(daily_returns.max() * 100, 2),
        "worst_day":        round(daily_returns.min() * 100, 2),
    }


def _max_drawdown_duration(drawdowns: pd.Series) -> int:
    """Number of consecutive days in the deepest drawdown."""
    in_dd     = drawdowns < 0
    max_dur   = 0
    current   = 0
    for v in in_dd:
        if v:
            current += 1
            max_dur = max(max_dur, current)
        else:
            current = 0
    return max_dur


# ─────────────────────────────────────────────────────────────────
# Strategy 1: Signal-based (long when BULLISH, flat otherwise)
# ─────────────────────────────────────────────────────────────────

def run_signal_strategy(
    ticker: str,
    price_df: pd.DataFrame,
    signal_df: pd.DataFrame,   # needs: date index, 'signal' col
    initial_capital: float = 10_000,
    hold_days: int = 5,
    transaction_cost: float = 0.001,  # 10 bps each way
) -> BacktestResult:
    """
    Simple signal strategy:
      BULLISH/STRONG_BUY → buy, hold N days
      BEARISH/STRONG_SELL → short (or stay flat)
      NEUTRAL → stay flat

    Includes transaction costs (realistic for retail).
    """
    capital  = initial_capital
    equity   = []
    trades   = []
    position = None

    for date in signal_df.index:
        if date not in price_df.index:
            continue

        price  = float(price_df.loc[date, "Close"])
        signal = signal_df.loc[date, "signal"]
        equity.append({"date": date, "equity": capital})

        # Close open position if hold period expired
        if position and position.is_open:
            days_held = (date - position.entry_date).days
            if days_held >= hold_days:
                position.exit_date  = date
                position.exit_price = price * (1 - transaction_cost)
                capital *= (1 + position.pnl_pct)
                trades.append(position)
                position = None

        # Open new position
        if position is None:
            direction = None
            if signal in {"BULLISH", "STRONG_BUY"}:
                direction = 1
            elif signal in {"BEARISH", "STRONG_SELL"}:
                direction = -1  # short

            if direction:
                entry_p  = price * (1 + transaction_cost)
                position = Trade(ticker, date, entry_p, direction, signal=signal)

    # Close any remaining position
    if position and position.is_open:
        last_price = float(price_df["Close"].iloc[-1])
        position.exit_date  = price_df.index[-1]
        position.exit_price = last_price
        capital *= (1 + position.pnl_pct)
        trades.append(position)

    equity_series = pd.Series(
        [e["equity"] for e in equity],
        index=[e["date"] for e in equity]
    )

    result = BacktestResult(
        ticker=ticker, strategy="SignalStrategy",
        start_date=str(price_df.index[0].date()),
        end_date=str(price_df.index[-1].date()),
        initial_capital=initial_capital,
        final_capital=capital,
        trades=trades,
    )

    result.equity_curve = equity_series
    result.metrics      = compute_metrics(equity_series)
    return result


# ─────────────────────────────────────────────────────────────────
# Strategy 2: Buy-and-hold benchmark
# ─────────────────────────────────────────────────────────────────

def run_buy_and_hold(
    ticker: str,
    price_df: pd.DataFrame,
    initial_capital: float = 10_000,
) -> dict:
    """Buy on first day, hold till last. Used as benchmark."""
    shares = initial_capital / float(price_df["Close"].iloc[0])
    equity = price_df["Close"] * shares
    metrics = compute_metrics(equity)
    metrics["strategy"] = "Buy & Hold"
    metrics["ticker"]   = ticker
    return metrics, equity


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def plot_backtest(result: BacktestResult, bah_equity: pd.Series, ticker: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{ticker} — Backtest Results", fontsize=14)

    # Equity curves
    ec = result.equity_curve
    ec_norm = ec / ec.iloc[0] * 100
    bah_norm = bah_equity / bah_equity.iloc[0] * 100

    axes[0].plot(ec_norm.index, ec_norm.values, color="#00c896", linewidth=1.5, label="ML Strategy")
    axes[0].plot(bah_norm.index, bah_norm.values, color="#94a3b8", linewidth=1, linestyle="--", label="Buy & Hold", alpha=0.8)
    axes[0].axhline(100, color="gray", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("Equity (base 100)")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # Drawdown
    peaks = ec.cummax()
    dd    = (ec - peaks) / peaks * 100
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4, color="#ff4d6d")
    axes[1].set_ylabel("Drawdown %")
    axes[1].spines[["top", "right"]].set_visible(False)

    # Trade returns
    if result.trades:
        trade_rets = [t.pnl_pct * 100 for t in result.trades]
        colours    = ["#00c896" if r > 0 else "#ff4d6d" for r in trade_rets]
        axes[2].bar(range(len(trade_rets)), trade_rets, color=colours, width=0.7)
        axes[2].axhline(0, color="gray", linewidth=0.5)
        axes[2].set_ylabel("Trade Return %")
        axes[2].set_xlabel("Trade #")
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = RESULTS_DIR / f"{ticker}_backtest.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[backtest] Chart → {out}")


# ─────────────────────────────────────────────────────────────────
# Full runner
# ─────────────────────────────────────────────────────────────────

def run_backtest(ticker: str) -> dict:
    """
    End-to-end backtest using simulated signals from the ML pipeline.
    In production, signals would come from the trained model.
    Here we simulate realistic signal distribution.
    """
    from ml.data.pipeline import fetch_price, engineer_features
    print(f"\n[backtest] Running for {ticker}…")

    price_df = fetch_price(ticker, period="2y")
    feat_df  = engineer_features(price_df)

    # Simulate realistic ML signals based on technical indicators
    # In production: replace with actual model.predict() calls
    np.random.seed(42)
    signals = []
    for _, row in feat_df.iterrows():
        # Simple heuristic signal combining RSI + MACD + momentum
        score = 0
        if "rsi_14" in feat_df.columns:
            if row.get("rsi_14", 50) < 35:   score += 1
            elif row.get("rsi_14", 50) > 70: score -= 1
        if "macd_cross" in feat_df.columns:
            score += int(row.get("macd_cross", 0))
        if "golden_cross" in feat_df.columns:
            score += int(row.get("golden_cross", 0))
        if "vs_sma20" in feat_df.columns:
            score += 1 if row.get("vs_sma20", 0) > 0.02 else (-1 if row.get("vs_sma20",0) < -0.02 else 0)

        if   score >= 2:  sig = "BULLISH"
        elif score <= -2: sig = "BEARISH"
        else:             sig = "NEUTRAL"
        signals.append(sig)

    signal_df = pd.DataFrame({"signal": signals}, index=feat_df.index)

    # Run ML strategy
    result = run_signal_strategy(ticker, price_df, signal_df)

    # Benchmark
    bah_metrics, bah_equity = run_buy_and_hold(ticker, price_df)

    # Plot
    if hasattr(result, "equity_curve"):
        plot_backtest(result, bah_equity, ticker)

    summary = result.summary()
    summary.update({"ml_" + k: v for k, v in result.metrics.items()})

    print(f"\n── Backtest Results: {ticker} ──────────────────────────────")
    print(f"  ML Strategy:")
    for k, v in result.metrics.items():
        print(f"    {k:<30} {v}")
    print(f"\n  Buy & Hold Benchmark:")
    for k, v in bah_metrics.items():
        if k not in ("strategy", "ticker"):
            print(f"    {k:<30} {v}")

    return {"ml": result.metrics, "benchmark": bah_metrics, "trades": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()
    run_backtest(args.ticker)
