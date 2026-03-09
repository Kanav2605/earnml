"""
ml/data/pipeline.py
────────────────────
Multi-source data pipeline with in-memory caching.

Sources:
  1. Yahoo Finance      — OHLCV, fundamentals, earnings history
  2. FRED (via pandas-datareader) — macro: VIX, T-bill rates, CPI
  3. Yahoo RSS / yfinance.news   — news headlines for sentiment
  4. Computed                    — 40+ technical + macro features

All fetched data is cached in-memory (TTL 5 min) to avoid
rate-limiting during development and model iteration.
"""

import warnings, time, functools
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
import requests, feedparser
from cachetools import TTLCache, cached
warnings.filterwarnings("ignore")

# ─── in-memory cache (TTL = 5 min per ticker) ───────────────────────────────
_cache = TTLCache(maxsize=64, ttl=300)

def _cache_key(*args, **kwargs):
    return str(args) + str(sorted(kwargs.items()))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRICE  +  FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price(ticker: str, period: str = "3y") -> pd.DataFrame:
    key = f"price:{ticker}:{period}"
    if key in _cache:
        return _cache[key]
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(inplace=True)
    _cache[key] = df
    print(f"[pipeline] {ticker} price: {len(df)} rows")
    return df


def fetch_fundamentals(ticker: str) -> dict:
    key = f"fund:{ticker}"
    if key in _cache:
        return _cache[key]
    info = yf.Ticker(ticker).info or {}
    fields = [
        "trailingPE","forwardPE","priceToBook","priceToSalesTrailing12Months",
        "enterpriseToEbitda","enterpriseToRevenue",
        "revenueGrowth","earningsGrowth","earningsQuarterlyGrowth",
        "grossMargins","operatingMargins","profitMargins","ebitdaMargins",
        "debtToEquity","currentRatio","quickRatio","returnOnEquity","returnOnAssets",
        "trailingEps","forwardEps","bookValue",
        "fiftyTwoWeekHigh","fiftyTwoWeekLow","beta",
        "shortRatio","shortPercentOfFloat",
        "heldPercentInstitutions","heldPercentInsiders",
        "sharesPercentSharesOut","floatShares",
        "dividendYield","payoutRatio",
        "marketCap","totalRevenue","totalDebt","freeCashflow",
    ]
    result = {k: info.get(k) for k in fields}
    _cache[key] = result
    return result


def fetch_earnings_history(ticker: str) -> pd.DataFrame:
    key = f"earn:{ticker}"
    if key in _cache:
        return _cache[key]
    try:
        eh = yf.Ticker(ticker).earnings_history
        if eh is None or eh.empty:
            return pd.DataFrame()
        eh.index = pd.to_datetime(eh.index).tz_localize(None)
        eh.sort_index(inplace=True)
        _cache[key] = eh
        return eh
    except:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MACRO  DATA  (VIX, rates, market-wide context)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_macro(start: str = "2019-01-01") -> pd.DataFrame:
    """
    Pull macro context from Yahoo Finance proxies.
    VIX, S&P 500, 10Y Treasury yield, DXY (US Dollar Index).
    """
    key = f"macro:{start}"
    if key in _cache:
        return _cache[key]

    tickers = {
        "^VIX":  "vix",
        "^GSPC": "spx",
        "^TNX":  "t10y",
        "DX-Y.NYB": "dxy",
    }
    frames = []
    for sym, name in tickers.items():
        try:
            df = yf.download(sym, start=start, auto_adjust=True, progress=False)
            df = df[["Close"]].rename(columns={"Close": name})
            df.index = pd.to_datetime(df.index).tz_localize(None)
            frames.append(df)
        except Exception as e:
            print(f"[pipeline] macro {sym} failed: {e}")

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1).ffill()
    # Add derived macro features
    if "vix" in macro.columns:
        macro["vix_ma20"]       = macro["vix"].rolling(20).mean()
        macro["vix_regime"]     = (macro["vix"] > 25).astype(int)  # 1=fear
        macro["vix_change_5d"]  = macro["vix"].pct_change(5)
    if "spx" in macro.columns:
        macro["spx_ret_20d"]    = macro["spx"].pct_change(20)
        macro["spx_above_200d"] = (macro["spx"] > macro["spx"].rolling(200).mean()).astype(int)
    if "t10y" in macro.columns:
        macro["rate_change_20d"]= macro["t10y"].diff(20)
        macro["rate_regime"]    = (macro["t10y"] > 4.0).astype(int)  # 1=high rate env

    macro.dropna(how="all", inplace=True)
    _cache[key] = macro
    print(f"[pipeline] Macro data: {len(macro)} rows, {list(macro.columns)}")
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NEWS  HEADLINES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_headlines(ticker: str, max_items: int = 40) -> list[dict]:
    key = f"news:{ticker}"
    if key in _cache:
        return _cache[key]

    headlines = []
    headers = {"User-Agent": "Mozilla/5.0"}

    # Source 1: Yahoo Finance RSS
    for url in [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
    ]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                headlines.append({
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "source": "Yahoo Finance RSS",
                    "ticker": ticker,
                })
            if headlines:
                break
        except:
            pass

    # Source 2: yfinance fallback
    if not headlines:
        try:
            news = yf.Ticker(ticker).news or []
            for n in news[:max_items]:
                headlines.append({
                    "title": n.get("title", ""),
                    "published": str(n.get("providerPublishTime", "")),
                    "source": n.get("publisher", "yfinance"),
                    "ticker": ticker,
                })
        except:
            pass

    print(f"[pipeline] {ticker} headlines: {len(headlines)}")
    _cache[key] = headlines
    return headlines


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TECHNICAL  FEATURE  ENGINEERING  (40+ features)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price DataFrame, compute 40+ technical features.
    All operations are vectorised with pandas/numpy.
    """
    d = df.copy()
    c = d["Close"]; h = d["High"]; lo = d["Low"]; v = d["Volume"]

    # ── Returns ──────────────────────────────────────────────────────────────
    for n in [1, 2, 3, 5, 10, 20, 60]:
        d[f"ret_{n}d"] = c.pct_change(n)

    # ── Log returns ──────────────────────────────────────────────────────────
    d["log_ret_1d"] = np.log(c / c.shift(1))

    # ── Moving averages ───────────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100, 200]:
        d[f"sma_{w}"]  = c.rolling(w).mean()
        d[f"ema_{w}"]  = c.ewm(span=w, adjust=False).mean()

    # ── Price vs MAs ─────────────────────────────────────────────────────────
    for w in [20, 50, 200]:
        d[f"vs_sma{w}"] = c / d[f"sma_{w}"] - 1

    # ── MA crossovers (binary signals) ───────────────────────────────────────
    d["golden_cross"]  = (d["sma_50"] > d["sma_200"]).astype(int)
    d["ema_cross_5_20"]= (d["ema_5"]  > d["ema_20"]).astype(int)

    # ── RSI (multiple windows) ────────────────────────────────────────────────
    for w in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(w).mean()
        loss  = (-delta.clip(upper=0)).rolling(w).mean()
        d[f"rsi_{w}"]   = 100 - (100 / (1 + gain / (loss + 1e-9)))
    d["rsi_oversold"]   = (d["rsi_14"] < 30).astype(int)
    d["rsi_overbought"] = (d["rsi_14"] > 70).astype(int)

    # ── MACD ─────────────────────────────────────────────────────────────────
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]        = e12 - e26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]
    d["macd_cross"]  = (d["macd"] > d["macd_signal"]).astype(int)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d["bb_upper"]  = bb_mid + 2 * bb_std
    d["bb_lower"]  = bb_mid - 2 * bb_std
    d["bb_width"]  = (d["bb_upper"] - d["bb_lower"]) / bb_mid
    d["bb_pct"]    = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)
    d["bb_squeeze"]= (d["bb_width"] < d["bb_width"].rolling(120).quantile(0.2)).astype(int)

    # ── ATR ───────────────────────────────────────────────────────────────────
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    d["atr_14"]   = tr.rolling(14).mean()
    d["atr_pct"]  = d["atr_14"] / c
    d["atr_ratio"]= d["atr_14"] / d["atr_14"].rolling(60).mean()  # normalised

    # ── Stochastic ────────────────────────────────────────────────────────────
    ll14 = lo.rolling(14).min(); hh14 = h.rolling(14).max()
    d["stoch_k"]  = 100 * (c - ll14) / (hh14 - ll14 + 1e-9)
    d["stoch_d"]  = d["stoch_k"].rolling(3).mean()
    d["stoch_cross"]= (d["stoch_k"] > d["stoch_d"]).astype(int)

    # ── Volume ────────────────────────────────────────────────────────────────
    d["vol_sma20"]  = v.rolling(20).mean()
    d["vol_ratio"]  = v / (d["vol_sma20"] + 1)
    d["vol_surge"]  = (d["vol_ratio"] > 2).astype(int)
    d["vol_ret"]    = v.pct_change(1)

    # ── On-Balance Volume (OBV) ───────────────────────────────────────────────
    obv = (np.sign(c.diff()) * v).cumsum()
    d["obv"]        = obv
    d["obv_sma20"]  = obv.rolling(20).mean()
    d["obv_trend"]  = (obv > d["obv_sma20"]).astype(int)

    # ── Volatility regimes ────────────────────────────────────────────────────
    for w in [10, 20, 60]:
        d[f"vol_{w}d"] = d["log_ret_1d"].rolling(w).std() * np.sqrt(252)
    d["vol_ratio_10_60"] = d["vol_10d"] / (d["vol_60d"] + 1e-9)  # vol term structure

    # ── 52-week positioning ───────────────────────────────────────────────────
    d["high_52w"]        = c.rolling(252).max()
    d["low_52w"]         = c.rolling(252).min()
    d["from_52w_high"]   = c / d["high_52w"] - 1
    d["from_52w_low"]    = c / d["low_52w"]  - 1
    d["range_52w_pct"]   = (c - d["low_52w"]) / (d["high_52w"] - d["low_52w"] + 1e-9)

    # ── Gap ───────────────────────────────────────────────────────────────────
    d["overnight_gap"]   = (d["Open"] / c.shift(1) - 1) if "Open" in d.columns else 0
    d["intraday_range"]  = (h - lo) / (c + 1e-9)

    d.dropna(inplace=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MERGE  ALL  SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def build_master_dataset(ticker: str, forward_days: int = 5) -> pd.DataFrame:
    """
    Full pipeline: price → features → macro merge → label.
    Returns a model-ready DataFrame.
    """
    price_df  = fetch_price(ticker, period="5y")
    feat_df   = engineer_features(price_df)
    macro_df  = fetch_macro()

    # Align macro to price dates
    if not macro_df.empty:
        macro_aligned = macro_df.reindex(feat_df.index, method="ffill")
        feat_df = pd.concat([feat_df, macro_aligned], axis=1)

    # Broadcast fundamentals
    funds = fetch_fundamentals(ticker)
    for k, v in funds.items():
        if v is not None:
            feat_df[f"f_{k}"] = float(v)

    # Label: forward return direction
    feat_df["fwd_ret"]   = feat_df["Close"].pct_change(forward_days).shift(-forward_days)
    feat_df["target"]    = (feat_df["fwd_ret"] > 0).astype(int)
    feat_df.dropna(inplace=True)

    print(f"[pipeline] Master dataset: {feat_df.shape[0]} rows × {feat_df.shape[1]} cols")
    return feat_df


if __name__ == "__main__":
    df = build_master_dataset("AAPL")
    import yfinance as yf

    # Download VIX data
    vix_data = yf.download("^VIX", period="5y", progress=False)

    # Add VIX close price to dataframe
    df = df.join(vix_data["Close"].rename(columns={"Close": "vix"}))
    # Download S&P 500 data
    spx_data = yf.download("^GSPC", period="5y", progress=False)

    # Calculate 20-day return
    spx_ret = spx_data["Close"].pct_change(20)

    # Add to dataframe
    df["spx_ret_20d"] = spx_data["Close"].pct_change(20)
    print("\nAll dataframe columns:")
    print(df.columns.tolist())
    cols = ["Close","rsi_14","macd","vix","spx_ret_20d","target"]
    existing_cols = [c for c in cols if c in df.columns]

    print(df[existing_cols].tail(5))
    df = df.dropna()
    print("Columns in dataframe:")
    print(df.columns)
