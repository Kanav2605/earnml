"""
api/db.py
──────────
Lightweight database layer using SQLAlchemy Core.

Defaults to SQLite (zero config for dev/portfolio demo).
Set DATABASE_URL=postgresql://... env var for production.

Tables:
  predictions   — every prediction call stored here
  model_metrics — training metrics per run
"""

from __future__ import annotations

import os, json
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, text, MetaData, Table, Column,
    Integer, String, Float, Boolean, DateTime, JSON
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///earnml.db")
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
        _engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
        _create_tables()
    return _engine


@contextmanager
def get_db():
    engine = get_engine()
    with engine.connect() as conn:
        yield conn
        conn.commit()


def _create_tables():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker              TEXT NOT NULL,
                predicted_at        TEXT NOT NULL,
                signal              TEXT,
                score               REAL,
                beat_probability    REAL,
                clf_prediction      TEXT,
                current_price       REAL,
                predicted_return    REAL,
                sentiment_label     TEXT,
                vol_regime          TEXT,
                vix                 REAL,
                t10y_rate           REAL,
                drift_status        TEXT,
                elapsed_seconds     REAL,
                llm_summary         TEXT,
                actual_beat         INTEGER,
                actual_price_5d     REAL,
                actual_return_5d    REAL,
                outcome_recorded_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at   TEXT NOT NULL,
                ticker        TEXT,
                model_name    TEXT NOT NULL,
                metric_name   TEXT NOT NULL,
                metric_value  REAL NOT NULL,
                split         TEXT,
                mlflow_run_id TEXT
            )
        """))
        conn.commit()


def save_prediction(ticker: str, result: dict) -> int:
    """Insert a prediction row. Returns the new row id."""
    comp   = result.get("composite", {})
    models = result.get("models", {})
    macro  = result.get("macro", {})
    clf    = models.get("classifier", {})
    lstm   = models.get("price_predictor", {})
    sent   = models.get("sentiment", {})
    vol    = models.get("volatility", {})
    comm   = result.get("commentary", {})
    drift  = result.get("drift", {})

    with get_db() as conn:
        r = conn.execute(text("""
            INSERT INTO predictions
              (ticker, predicted_at, signal, score,
               beat_probability, clf_prediction,
               current_price, predicted_return,
               sentiment_label, vol_regime,
               vix, t10y_rate, drift_status,
               elapsed_seconds, llm_summary)
            VALUES
              (:ticker, :predicted_at, :signal, :score,
               :beat_probability, :clf_prediction,
               :current_price, :predicted_return,
               :sentiment_label, :vol_regime,
               :vix, :t10y_rate, :drift_status,
               :elapsed_seconds, :llm_summary)
        """), {
            "ticker":           ticker,
            "predicted_at":     datetime.utcnow().isoformat(),
            "signal":           comp.get("signal"),
            "score":            comp.get("score"),
            "beat_probability": clf.get("beat_probability"),
            "clf_prediction":   clf.get("prediction"),
            "current_price":    lstm.get("current_price"),
            "predicted_return": lstm.get("predicted_return"),
            "sentiment_label":  sent.get("overall_sentiment"),
            "vol_regime":       vol.get("regime"),
            "vix":              macro.get("vix"),
            "t10y_rate":        macro.get("t10y"),
            "drift_status":     drift.get("status"),
            "elapsed_seconds":  result.get("elapsed_seconds"),
            "llm_summary":      comm.get("summary") if isinstance(comm, dict) else None,
        })
        return r.lastrowid


def get_history(ticker: str, limit: int = 20) -> list[dict]:
    """Fetch recent predictions for a ticker."""
    with get_db() as conn:
        rows = conn.execute(text("""
            SELECT id, predicted_at, signal, score,
                   beat_probability, clf_prediction,
                   current_price, predicted_return,
                   sentiment_label, vol_regime,
                   drift_status, elapsed_seconds,
                   actual_beat, actual_return_5d,
                   llm_summary
            FROM predictions
            WHERE ticker = :ticker
            ORDER BY predicted_at DESC
            LIMIT :limit
        """), {"ticker": ticker, "limit": limit})
        cols = rows.keys()
        return [dict(zip(cols, row)) for row in rows]


def update_outcome(prediction_id: int, actual_beat: bool, actual_price: float) -> None:
    """Record actual outcome for a past prediction."""
    with get_db() as conn:
        # Get original predicted price
        row = conn.execute(text(
            "SELECT current_price, predicted_return FROM predictions WHERE id=:id"
        ), {"id": prediction_id}).fetchone()

        actual_return = None
        if row and row[0]:
            actual_return = round((actual_price - row[0]) / row[0] * 100, 3)

        conn.execute(text("""
            UPDATE predictions
            SET actual_beat         = :beat,
                actual_price_5d     = :price,
                actual_return_5d    = :ret,
                outcome_recorded_at = :ts
            WHERE id = :id
        """), {
            "beat":  int(actual_beat),
            "price": actual_price,
            "ret":   actual_return,
            "ts":    datetime.utcnow().isoformat(),
            "id":    prediction_id,
        })


def save_model_metrics(ticker: str, model_name: str, metrics: dict,
                       split: str = "test", mlflow_run_id: str = None) -> None:
    """Log model training metrics to DB (mirrors what goes to MLflow)."""
    with get_db() as conn:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                conn.execute(text("""
                    INSERT INTO model_metrics
                      (recorded_at, ticker, model_name, metric_name, metric_value, split, mlflow_run_id)
                    VALUES (:ts, :ticker, :model, :metric, :value, :split, :run_id)
                """), {
                    "ts":      datetime.utcnow().isoformat(),
                    "ticker":  ticker,
                    "model":   model_name,
                    "metric":  metric_name,
                    "value":   float(metric_value),
                    "split":   split,
                    "run_id":  mlflow_run_id,
                })
