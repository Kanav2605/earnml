"""
ml/models/volatility_model.py
──────────────────────────────
GARCH(1,1) volatility forecasting.

GARCH (Generalised AutoRegressive Conditional Heteroskedasticity)
models the *volatility clustering* phenomenon — when markets are
volatile, they tend to stay volatile.

Outputs:
  - Current annualised volatility
  - 5-day and 10-day volatility forecast
  - VaR (Value at Risk) at 95% and 99% confidence
  - Regime label: LOW / NORMAL / ELEVATED / EXTREME

Usage:
  python ml/models/volatility_model.py --ticker TSLA
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from arch import arch_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.data.fetch_data import fetch_stock_data

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Fit GARCH model
# ─────────────────────────────────────────────────────────────

def fit_garch(ticker: str, period: str = "3y"):
    """
    Fit a GARCH(1,1) model on daily log returns.
    Returns the fitted result object.
    """
    df = fetch_stock_data(ticker, period=period)
    returns = 100 * np.log(df["Close"] / df["Close"].shift(1)).dropna()

    print(f"[garch] {ticker}: {len(returns)} return observations")
    print(f"[garch] Return stats — Mean: {returns.mean():.4f}%  Std: {returns.std():.4f}%")

    garch = arch_model(
        returns,
        vol="Garch",
        p=1, q=1,
        mean="AR",
        lags=1,
        dist="t",    # Student-t captures fat tails
        rescale=True
    )

    result = garch.fit(
        disp="off",
        show_warning=False,
        options={"maxiter": 500}
    )

    return result, returns


# ─────────────────────────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────────────────────────

def forecast_volatility(ticker: str, horizon: int = 10) -> dict:
    """
    Fit GARCH and return forward volatility forecasts + VaR.
    """
    result, returns = fit_garch(ticker)

    # ── Annualised historical volatility ──────────────────────
    hist_vol_30d = returns.rolling(30).std().iloc[-1] * np.sqrt(252)
    hist_vol_90d = returns.rolling(90).std().iloc[-1] * np.sqrt(252)

    # ── GARCH forecast ────────────────────────────────────────
    fcast = result.forecast(horizon=horizon, reindex=False)
    var_fcast = fcast.variance.iloc[-1].values   # daily variance (scaled)

    # Annualise (scale by 252, undo the rescaling factor)
    scale = result.scale
    daily_vols     = np.sqrt(var_fcast) / scale      # daily vol %
    annualised_5d  = np.sqrt(np.mean(var_fcast[:5]))  * np.sqrt(252) / scale
    annualised_10d = np.sqrt(np.mean(var_fcast[:10])) * np.sqrt(252) / scale

    # ── Value at Risk (historical simulation) ─────────────────
    # 1-day 95% and 99% VaR
    var_95 = np.percentile(returns, 5)   # 5th percentile loss
    var_99 = np.percentile(returns, 1)   # 1st percentile loss

    # ── Volatility regime ─────────────────────────────────────
    current_vol = hist_vol_30d
    if current_vol < 15:
        regime = "LOW"
    elif current_vol < 25:
        regime = "NORMAL"
    elif current_vol < 40:
        regime = "ELEVATED"
    else:
        regime = "EXTREME"

    # ── Current conditional vol (last fitted) ─────────────────
    cond_vol = np.sqrt(result.conditional_volatility.iloc[-1]) * np.sqrt(252) / scale

    _plot_volatility(result, returns, ticker, scale)

    return {
        "ticker":                  ticker,
        "current_conditional_vol": round(float(cond_vol), 2),
        "hist_vol_30d":            round(float(hist_vol_30d), 2),
        "hist_vol_90d":            round(float(hist_vol_90d), 2),
        "garch_forecast_5d":       round(float(annualised_5d), 2),
        "garch_forecast_10d":      round(float(annualised_10d), 2),
        "daily_forecast":          [round(v * np.sqrt(252), 2) for v in daily_vols[:horizon]],
        "var_95_1d":               round(float(var_95), 2),
        "var_99_1d":               round(float(var_99), 2),
        "regime":                  regime,
        "model":                   "GARCH(1,1)-t",
        "aic":                     round(result.aic, 2),
        "bic":                     round(result.bic, 2),
    }


def _plot_volatility(result, returns, ticker, scale):
    """Plot conditional volatility over time."""
    cond_vol = result.conditional_volatility / scale * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(returns.index, returns.values, color="#334155", linewidth=0.7, alpha=0.8)
    axes[0].set_title(f"{ticker} — Daily Returns (%)", fontsize=12)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(cond_vol.index, cond_vol.values, color="#C9A84C", linewidth=1.2)
    axes[1].fill_between(cond_vol.index, cond_vol.values, alpha=0.15, color="#C9A84C")
    axes[1].set_title(f"GARCH(1,1) Conditional Volatility (Annualised %)", fontsize=12)
    axes[1].set_xlabel("Date")
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = MODELS_DIR / f"{ticker}_garch_volatility.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[garch] Volatility chart → {out}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GARCH Volatility Model")
    parser.add_argument("--ticker",  default="AAPL")
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    result = forecast_volatility(args.ticker, args.horizon)
    print(f"\n── GARCH Volatility Forecast: {args.ticker} ──────────────")
    for k, v in result.items():
        if k != "daily_forecast":
            print(f"  {k:<35} {v}")
    print(f"\n  Daily forecast (annualised %):")
    for i, v in enumerate(result["daily_forecast"], 1):
        print(f"    Day {i:>2}: {v:.2f}%")
