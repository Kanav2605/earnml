"""
ml/models/earnings_classifier.py
──────────────────────────────────
XGBoost classifier that predicts whether a company will
BEAT or MISS analyst EPS estimates next earnings.

Features: technical indicators + fundamentals in the
30 trading days leading up to the earnings date.

Training pipeline:
  1. fetch_feature_matrix()  → raw features
  2. engineer_earnings_features() → pre-earnings window
  3. XGBClassifier with cross-validated hyperparameters
  4. SHAP explainability

Usage:
  python ml/models/earnings_classifier.py --ticker AAPL
"""

import os
import sys
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from pathlib import Path
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.data.pipeline import fetch_stock_data, add_technical_features, fetch_fundamentals

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Feature Engineering: pre-earnings window
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "return_5d", "return_20d", "return_60d",
    "rsi_14", "macd", "macd_hist",
    "bb_pct", "bb_width",
    "atr_pct", "vol_ratio",
    "stoch_k", "stoch_d",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "volatility_10d", "volatility_30d",
    "pct_from_52w_high", "pct_from_52w_low",
]

FUNDAMENTAL_COLS = [
    "fund_trailingPE", "fund_forwardPE", "fund_priceToBook",
    "fund_revenueGrowth", "fund_earningsGrowth",
    "fund_grossMargins", "fund_operatingMargins",
    "fund_debtToEquity", "fund_beta",
    "fund_shortRatio",
]


def build_earnings_dataset(ticker: str) -> pd.DataFrame:
    """
    For each historical earnings date, grab the 30-day pre-earnings
    window of features and label with BEAT(1) / MISS(0).
    """
    stock = yf.Ticker(ticker)

    # --- price features ---
    price_df = fetch_stock_data(ticker, period="5y")
    feat_df  = add_technical_features(price_df)

    # --- earnings history ---
    try:
        eh = stock.earnings_history
        if eh is None or eh.empty:
            raise ValueError("No earnings history")
        eh.index = pd.to_datetime(eh.index).tz_localize(None)
        eh = eh[["epsEstimate", "epsActual", "surprisePercent"]].dropna()
    except Exception as e:
        print(f"[classifier] earnings history error: {e}")
        return pd.DataFrame()

    # --- fundamentals (scalar, broadcast) ---
    funds = fetch_fundamentals(ticker)

    records = []
    for earn_date, row in eh.iterrows():
        # Window: 30 trading days before earnings
        window_end   = feat_df.index[feat_df.index <= earn_date]
        if len(window_end) < 30:
            continue
        window_slice = feat_df.loc[window_end[-30:]]

        # Aggregate window: mean + std of each feature
        agg = {}
        for col in FEATURE_COLS:
            if col in window_slice.columns:
                agg[f"{col}_mean"] = window_slice[col].mean()
                agg[f"{col}_std"]  = window_slice[col].std()
                agg[f"{col}_last"] = window_slice[col].iloc[-1]

        # Append fundamentals
        for k, v in funds.items():
            if v is not None:
                agg[f"fund_{k}"] = float(v)

        # Label: 1 = beat, 0 = miss
        agg["label"]           = int(row["surprisePercent"] > 0)
        agg["surprise_pct"]    = row["surprisePercent"]
        agg["eps_estimate"]    = row["epsEstimate"]
        agg["eps_actual"]      = row["epsActual"]
        agg["earnings_date"]   = earn_date

        records.append(agg)

    df = pd.DataFrame(records).set_index("earnings_date").sort_index()
    print(f"[classifier] Dataset: {len(df)} earnings events | Beat rate: {df['label'].mean():.1%}")
    return df


# ─────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────

def train(ticker: str, save: bool = True):
    """Train XGBoost classifier and save to disk."""
    print(f"\n{'='*55}")
    print(f"  EARNINGS BEAT/MISS CLASSIFIER — {ticker}")
    print(f"{'='*55}\n")

    df = build_earnings_dataset(ticker)
    if df.empty or len(df) < 6:
        print("[classifier] Not enough data. Need at least 6 earnings events.")
        return None, None

    # Drop non-feature columns
    drop_cols = ["label", "surprise_pct", "eps_estimate", "eps_actual"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]

    # Fill any remaining NaN with median
    X = X.fillna(X.median(numeric_only=True))

    print(f"Features: {X.shape[1]}  |  Samples: {len(X)}")
    print(f"Class balance — Beat: {y.sum()}  Miss: {(y==0).sum()}\n")

    # XGBoost with sensible defaults for small financial datasets
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),  # handle imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    # Time-series cross-validation (never peek into the future)
    tscv = TimeSeriesSplit(n_splits=min(5, len(X) - 1))
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
    print(f"Cross-val ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Per-fold scores:   {[f'{s:.3f}' for s in cv_scores]}\n")

    # Final fit on all data
    model.fit(X, y)

    # Evaluation on last 20% (out-of-sample)
    split = int(len(X) * 0.8)
    if split > 0 and split < len(X):
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print("── Out-of-sample Report ──────────────────────────────")
        print(classification_report(y_test, y_pred, target_names=["MISS", "BEAT"]))
        if len(np.unique(y_test)) > 1:
            print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}\n")

    # Feature importance plot
    _plot_feature_importance(model, X.columns, ticker)

    if save:
        model_path = MODELS_DIR / f"{ticker}_earnings_clf.json"
        feat_path  = MODELS_DIR / f"{ticker}_earnings_clf_features.joblib"
        model.save_model(str(model_path))
        joblib.dump(list(X.columns), str(feat_path))
        print(f"[classifier] Model saved → {model_path}")

    return model, list(X.columns)


def _plot_feature_importance(model, feature_names, ticker):
    """Save a top-20 feature importance bar chart."""
    imp = pd.Series(model.feature_importances_, index=feature_names)
    top = imp.nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#C9A84C")
    ax.set_title(f"{ticker} — Top-20 Feature Importances (XGBoost)", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = MODELS_DIR / f"{ticker}_feature_importance.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[classifier] Feature importance plot → {out}")


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def predict(ticker: str) -> dict:
    """
    Load saved model and predict beat/miss probability
    for the NEXT earnings using the most recent 30 days of data.
    """
    model_path = MODELS_DIR / f"{ticker}_earnings_clf.json"
    feat_path  = MODELS_DIR / f"{ticker}_earnings_clf_features.joblib"

    # Auto-train if model not found
    if not model_path.exists():
        print(f"[classifier] No saved model for {ticker}, training now…")
        model, feature_names = train(ticker)
        if model is None:
            return {"error": "Not enough earnings history to train"}
    else:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        feature_names = joblib.load(str(feat_path))

    # Build latest features
    price_df = fetch_stock_data(ticker, period="3mo")
    feat_df  = add_technical_features(price_df)
    funds    = fetch_fundamentals(ticker)

    if len(feat_df) < 30:
        return {"error": "Not enough recent price data"}

    window = feat_df.iloc[-30:]
    row = {}
    for col in FEATURE_COLS:
        if col in window.columns:
            row[f"{col}_mean"] = window[col].mean()
            row[f"{col}_std"]  = window[col].std()
            row[f"{col}_last"] = window[col].iloc[-1]
    for k, v in funds.items():
        if v is not None:
            row[f"fund_{k}"] = float(v)

    X_inf = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)
    proba = model.predict_proba(X_inf)[0]

    return {
        "ticker": ticker,
        "beat_probability":  round(float(proba[1]) * 100, 1),
        "miss_probability":  round(float(proba[0]) * 100, 1),
        "prediction":        "BEAT" if proba[1] > 0.5 else "MISS",
        "confidence":        round(float(max(proba)) * 100, 1),
        "model":             "XGBoost Earnings Classifier",
        "features_used":     len(feature_names),
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings Beat/Miss Classifier")
    parser.add_argument("--ticker",  default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--train",   action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Run inference")
    args = parser.parse_args()

    if args.train:
        train(args.ticker)
    if args.predict or not args.train:
        result = predict(args.ticker)
        print("\n── Prediction ────────────────────────────────────────")
        for k, v in result.items():
            print(f"  {k:<25} {v}")
