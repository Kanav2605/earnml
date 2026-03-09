"""
ml/models/price_predictor.py
─────────────────────────────
LSTM neural network that predicts the direction and magnitude
of price movement over the next N trading days.

Architecture:
  Input  → [seq_len=60, n_features] time-series window
  LSTM   → 128 units, dropout 0.2
  LSTM   → 64  units, dropout 0.2
  Dense  → 32  units, relu
  Output → 1   unit   (predicted % return)

Usage:
  python ml/models/price_predictor.py --ticker NVDA --days 5
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.data.pipeline import fetch_stock_data, add_technical_features

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 60  # look-back window (trading days)

LSTM_FEATURES = [
    "Close", "Volume",
    "return_1d", "return_5d",
    "rsi_14", "macd", "macd_hist",
    "bb_pct", "bb_width",
    "atr_pct", "vol_ratio",
    "stoch_k",
    "volatility_10d",
    "price_vs_sma20", "price_vs_sma50",
    "ema_5", "ema_20",
]


# ─────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────

def prepare_sequences(df: pd.DataFrame, forward_days: int = 5):
    """
    Convert a feature DataFrame into (X, y) numpy arrays
    of overlapping windows of length SEQ_LEN.

    y = forward % return (regression target)
    """
    # Keep only available feature columns
    cols = [c for c in LSTM_FEATURES if c in df.columns]
    data = df[cols].values.astype(np.float32)

    # Scale each feature independently
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)

    # Build target: forward return on Close
    close_idx = cols.index("Close")
    raw_close = df["Close"].values

    X_list, y_list = [], []
    for i in range(SEQ_LEN, len(data_scaled) - forward_days):
        X_list.append(data_scaled[i - SEQ_LEN : i])
        future  = raw_close[i + forward_days]
        current = raw_close[i]
        y_list.append((future - current) / current)   # % return

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, scaler, cols


# ─────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────

def build_lstm_model(n_features: int) -> tf.keras.Model:
    """
    Two-layer stacked LSTM with BatchNorm and Dropout.
    Outputs a single scalar (predicted % return).
    """
    model = Sequential([
        Input(shape=(SEQ_LEN, n_features)),

        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.1),

        Dense(1, activation="linear"),   # regression output
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="huber",                     # robust to outliers
        metrics=["mae"]
    )
    return model


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(ticker: str, forward_days: int = 5, epochs: int = 100):
    print(f"\n{'='*55}")
    print(f"  LSTM PRICE PREDICTOR — {ticker} ({forward_days}d forward)")
    print(f"{'='*55}\n")

    raw  = fetch_stock_data(ticker, period="5y")
    feat = add_technical_features(raw)

    X, y, scaler, cols = prepare_sequences(feat, forward_days)
    print(f"Sequences: {X.shape}  |  Targets: {y.shape}")
    print(f"Features: {cols}\n")

    # Chronological train/val/test split (no shuffling!)
    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}\n")

    model = build_lstm_model(n_features=X.shape[2])
    model.summary()

    ckpt_path = MODELS_DIR / f"{ticker}_lstm_best.keras"
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=1),
        ModelCheckpoint(str(ckpt_path), save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ──────────────────────────────────────────
    y_pred = model.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    # Directional accuracy
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_test))

    print(f"\n── Test Metrics ──────────────────────────────────────")
    print(f"  MAE              : {mae:.4f}  ({mae*100:.2f}%)")
    print(f"  RMSE             : {rmse:.4f}  ({rmse*100:.2f}%)")
    print(f"  Directional Acc  : {dir_acc:.3f}  ({dir_acc*100:.1f}%)\n")

    # ── Plots ────────────────────────────────────────────────
    _plot_training(history, ticker, forward_days)
    _plot_predictions(y_test, y_pred, ticker, forward_days)

    # ── Save ─────────────────────────────────────────────────
    meta = {
        "scaler": scaler,
        "feature_cols": cols,
        "forward_days": forward_days,
        "seq_len": SEQ_LEN,
        "metrics": {"mae": mae, "rmse": rmse, "directional_acc": dir_acc},
    }
    model.save(str(MODELS_DIR / f"{ticker}_lstm.keras"))
    joblib.dump(meta, MODELS_DIR / f"{ticker}_lstm_meta.joblib")
    print(f"[lstm] Saved → {MODELS_DIR / f'{ticker}_lstm.keras'}")

    return model, meta


def _plot_training(history, ticker, fwd):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{ticker} LSTM — Training Loss"); axes[0].legend()
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(history.history["mae"],     label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE"); axes[1].legend()
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{ticker}_lstm_training.png", dpi=120)
    plt.close()


def _plot_predictions(y_true, y_pred, ticker, fwd):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true * 100, label="Actual", alpha=0.7, linewidth=1)
    ax.plot(y_pred * 100, label="Predicted", alpha=0.7, linewidth=1, linestyle="--")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(f"{ticker} LSTM — {fwd}d Predicted vs Actual Return (%)")
    ax.set_ylabel("Return (%)"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{ticker}_lstm_predictions.png", dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def predict(ticker: str) -> dict:
    """Load model and predict next N days return for `ticker`."""
    model_path = MODELS_DIR / f"{ticker}_lstm.keras"
    meta_path  = MODELS_DIR / f"{ticker}_lstm_meta.joblib"

    if not model_path.exists():
        print(f"[lstm] No model found for {ticker}, training…")
        model, meta = train(ticker)
    else:
        model = load_model(str(model_path))
        meta  = joblib.load(str(meta_path))

    scaler      = meta["scaler"]
    feat_cols   = meta["feature_cols"]
    fwd         = meta["forward_days"]
    metrics     = meta.get("metrics", {})

    raw  = fetch_stock_data(ticker, period="6mo")
    feat = add_technical_features(raw)

    cols = [c for c in feat_cols if c in feat.columns]
    data = feat[cols].values[-SEQ_LEN:].astype(np.float32)
    data_scaled = scaler.transform(data)
    X = data_scaled[np.newaxis, :, :]   # (1, SEQ_LEN, n_features)

    pred_return = float(model.predict(X, verbose=0)[0][0])
    current_price = float(feat["Close"].iloc[-1])
    predicted_price = current_price * (1 + pred_return)

    return {
        "ticker":              ticker,
        "current_price":       round(current_price, 2),
        "predicted_return":    round(pred_return * 100, 2),
        "predicted_price":     round(predicted_price, 2),
        "forward_days":        fwd,
        "direction":           "UP ↑" if pred_return > 0 else "DOWN ↓",
        "model_directional_acc": round(metrics.get("directional_acc", 0) * 100, 1),
        "model":               "LSTM Price Predictor",
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Price Predictor")
    parser.add_argument("--ticker",  default="AAPL")
    parser.add_argument("--days",    type=int, default=5)
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    if args.train:
        train(args.ticker, forward_days=args.days, epochs=args.epochs)
    if args.predict or not args.train:
        result = predict(args.ticker)
        print("\n── Prediction ────────────────────────────────────────")
        for k, v in result.items():
            print(f"  {k:<30} {v}")
