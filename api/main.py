"""
api/main.py  (v3 — fully integrated)
──────────────────────────────────────
FastAPI backend wiring together every component:
  • ML prediction pipeline  (stacked ensemble + LSTM + FinBERT + GARCH)
  • MLflow experiment tracking  (every prediction logged)
  • Drift detection  (PSI check before serving stale models)
  • LLM analyst commentary  (Claude generates written analysis)
  • Backtesting endpoint  (run historical strategy simulation)
  • SQLite persistence  (predictions stored, outcomes tracked)
  • WebSocket progress  (real-time model pipeline updates)
  • Redis-style in-memory cache  (5-min TTL per ticker)
"""

from __future__ import annotations

import os, sys, time, json, asyncio, traceback, logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from cachetools import TTLCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("earnml.api")

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="EarnML API v3",
    description="Production-grade ML earnings analysis — JPMorgan portfolio project",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: TTLCache = TTLCache(maxsize=64, ttl=300)   # 5-min TTL


# ── Schemas ───────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker:        str  = Field(..., min_length=1, max_length=5, example="AAPL")
    use_lstm:      bool = Field(True)
    use_finbert:   bool = Field(False)
    use_ensemble:  bool = Field(True)
    use_llm:       bool = Field(True,  description="Generate LLM analyst commentary")
    run_backtest:  bool = Field(False, description="Include backtest (slower)")
    force_refresh: bool = Field(False, description="Bypass cache")

class OutcomeUpdate(BaseModel):
    prediction_id: int
    actual_beat:   bool
    actual_price:  float


# ── Health ────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":    "ok",
        "version":   "3.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "mlflow":   _check_mlflow(),
            "database": _check_db(),
        },
    }

def _check_mlflow() -> str:
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))
        return "ok"
    except Exception as e:
        return f"error: {e}"

def _check_db() -> str:
    try:
        from api.db import get_db
        with get_db() as db:
            db.execute("SELECT 1")
        return "ok"
    except Exception as e:
        return f"error: {e}"


# ── Live quote ────────────────────────────────────────────────────

@app.get("/api/quote/{ticker}", tags=["Market Data"])
async def quote(ticker: str):
    """Fast live price quote — no ML, just current market data."""
    ticker = ticker.upper().strip()
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        return {
            "ticker":      ticker,
            "price":       round(float(getattr(info, "last_price",  0) or 0), 2),
            "52w_high":    round(float(getattr(info, "year_high",   0) or 0), 2),
            "52w_low":     round(float(getattr(info, "year_low",    0) or 0), 2),
            "market_cap":  getattr(info, "market_cap",    None),
            "volume":      getattr(info, "last_volume",   None),
            "ts":          datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Main prediction endpoint ──────────────────────────────────────

@app.post("/api/predict", tags=["ML Predictions"])
async def predict(req: PredictRequest, background: BackgroundTasks):
    """
    Full ML prediction pipeline.
    Results cached 5 min per ticker; use force_refresh=true to bypass.
    """
    ticker    = req.ticker.upper().strip()
    cache_key = f"{ticker}:{req.use_lstm}:{req.use_finbert}:{req.use_ensemble}"

    if not req.force_refresh and cache_key in _cache:
        result = dict(_cache[cache_key])
        result["cached"] = True
        log.info(f"Cache hit: {ticker}")
        return result

    t0 = time.time()
    log.info(f"Running full pipeline for {ticker}")

    result = await _run_pipeline(ticker, req)
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    result["cached"] = False

    _cache[cache_key] = result

    # Persist prediction + log to MLflow in background (non-blocking)
    background.add_task(_persist_prediction, ticker, result)
    background.add_task(_log_to_mlflow, ticker, result)

    return result


async def _run_pipeline(ticker: str, req: PredictRequest) -> dict:
    output: dict = {
        "ticker":    ticker,
        "generated": datetime.utcnow().isoformat() + "Z",
        "models":    {},
    }

    # 1. Stacked ensemble / single classifier
    try:
        if req.use_ensemble:
            from ml.ensemble.stacked_classifier import predict as ens_predict
            output["models"]["classifier"] = ens_predict(ticker)
        else:
            from ml.models.earnings_classifier import predict as clf_predict
            output["models"]["classifier"] = clf_predict(ticker)
    except Exception as e:
        log.warning(f"Classifier failed for {ticker}: {e}")
        output["models"]["classifier"] = {"error": str(e)}

    # 2. LSTM
    if req.use_lstm:
        try:
            from ml.models.price_predictor import predict as lstm_predict
            output["models"]["price_predictor"] = lstm_predict(ticker)
        except Exception as e:
            output["models"]["price_predictor"] = {"error": str(e)}
    else:
        output["models"]["price_predictor"] = {"skipped": True}

    # 3. Sentiment
    try:
        from ml.models.sentiment_analyzer import analyze
        output["models"]["sentiment"] = analyze(ticker, use_finbert=req.use_finbert)
    except Exception as e:
        output["models"]["sentiment"] = {"error": str(e)}

    # 4. GARCH
    try:
        from ml.models.volatility_model import forecast_volatility
        output["models"]["volatility"] = forecast_volatility(ticker)
    except Exception as e:
        output["models"]["volatility"] = {"error": str(e)}

    # 5. Macro
    try:
        from ml.data.pipeline import fetch_macro
        macro = fetch_macro()
        if not macro.empty:
            last = macro.iloc[-1]
            output["macro"] = {
                "vix":         round(float(last.get("vix",         0) or 0), 2),
                "vix_regime":  "FEAR" if (last.get("vix", 0) or 0) > 25 else "NORMAL",
                "spx_ret_20d": round(float(last.get("spx_ret_20d", 0) or 0) * 100, 2),
                "t10y":        round(float(last.get("t10y",        0) or 0), 2),
                "rate_regime": "HIGH" if (last.get("t10y", 0) or 0) > 4.0 else "NORMAL",
            }
    except Exception as e:
        output["macro"] = {"error": str(e)}

    # 6. Composite signal
    output["composite"] = _composite(output["models"], output.get("macro", {}))

    # 7. Drift check (non-blocking — just adds a flag)
    try:
        from monitoring.drift_detector import DriftDetector
        from ml.data.pipeline import build_master_dataset
        detector = DriftDetector(ticker)
        df = build_master_dataset(ticker)
        drift_report = detector.check(df.tail(60))
        output["drift"] = {
            "status":              drift_report.get("overall_status", "UNKNOWN"),
            "retrain_recommended": drift_report.get("retrain_recommended", False),
            "features_drifted":    drift_report.get("features_drifted", 0),
        }
    except Exception as e:
        output["drift"] = {"status": "CHECK_FAILED", "error": str(e)}

    # 8. LLM commentary
    if req.use_llm:
        try:
            from llm_layer.analyst import generate_commentary
            output["commentary"] = await generate_commentary(ticker, output)
        except Exception as e:
            output["commentary"] = {"error": str(e)}

    # 9. Backtest (optional — slow)
    if req.run_backtest:
        try:
            from backtesting.engine import run_backtest
            output["backtest"] = run_backtest(ticker)
        except Exception as e:
            output["backtest"] = {"error": str(e)}

    return output


def _composite(models: dict, macro: dict) -> dict:
    votes, reasons = [], []

    clf = models.get("classifier", {})
    if "beat_probability" in clf:
        p = clf["beat_probability"]
        v = 1 if p > 57 else (-1 if p < 43 else 0)
        votes.append(v); reasons.append(f"Ensemble beat prob: {p}%")

    lstm = models.get("price_predictor", {})
    if "predicted_return" in lstm:
        r = lstm["predicted_return"]
        v = 1 if r > 1.5 else (-1 if r < -1.5 else 0)
        votes.append(v); reasons.append(f"LSTM 5d return: {r:+.2f}%")

    sent = models.get("sentiment", {})
    if "overall_sentiment" in sent:
        v = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(sent["overall_sentiment"], 0)
        votes.append(v); reasons.append(f"News sentiment: {sent['overall_sentiment']}")

    vol = models.get("volatility", {})
    if "regime" in vol:
        v = {"LOW": 1, "NORMAL": 0, "ELEVATED": -1, "EXTREME": -1}.get(vol["regime"], 0)
        votes.append(v); reasons.append(f"Vol regime: {vol['regime']}")

    if macro.get("vix_regime") == "FEAR":
        votes.append(-1); reasons.append(f"VIX={macro.get('vix')} — market fear")
    if macro.get("rate_regime") == "HIGH":
        votes.append(-1); reasons.append(f"10Y={macro.get('t10y')}% — high rates")

    score  = round(sum(votes) / max(len(votes), 1), 3)
    signal = ("STRONG_BUY"  if score >= 0.6  else
              "BULLISH"     if score >= 0.25 else
              "STRONG_SELL" if score <= -0.6  else
              "BEARISH"     if score <= -0.25 else "NEUTRAL")

    return {"signal": signal, "score": score, "votes": votes,
            "vote_count": len(votes), "reasons": reasons}


# ── Background tasks ──────────────────────────────────────────────

def _persist_prediction(ticker: str, result: dict) -> None:
    """Store prediction in SQLite for accuracy tracking later."""
    try:
        from api.db import save_prediction
        save_prediction(ticker, result)
    except Exception as e:
        log.warning(f"Failed to persist prediction for {ticker}: {e}")


def _log_to_mlflow(ticker: str, result: dict) -> None:
    """Log inference call metrics to MLflow."""
    try:
        from mlflow_tracking.tracker import MLTracker
        tracker = MLTracker("inference")
        with tracker.start_run(ticker=ticker, tags={"type": "inference"}):
            comp = result.get("composite", {})
            tracker.log_metrics({
                "composite_score":   comp.get("score", 0),
                "elapsed_seconds":   result.get("elapsed_seconds", 0),
                "beat_probability":  result.get("models", {}).get("classifier", {}).get("beat_probability", 0) or 0,
            })
            tracker.log_params({
                "signal":  comp.get("signal", ""),
                "ticker":  ticker,
            })
    except Exception as e:
        log.warning(f"MLflow logging failed for {ticker}: {e}")


# ── Backtest endpoint ─────────────────────────────────────────────

@app.post("/api/backtest/{ticker}", tags=["Backtesting"])
async def backtest(ticker: str):
    """Run historical strategy backtest for a ticker."""
    try:
        from backtesting.engine import run_backtest
        return run_backtest(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Drift monitoring endpoint ─────────────────────────────────────

@app.get("/api/drift/{ticker}", tags=["Monitoring"])
async def drift_check(ticker: str):
    """Check for feature drift — should model be retrained?"""
    try:
        from monitoring.drift_detector import DriftDetector
        from ml.data.pipeline import build_master_dataset
        detector = DriftDetector(ticker.upper())
        df       = build_master_dataset(ticker.upper())
        report   = detector.check(df.tail(60))
        # Strip heavy feature_details for API response
        report.pop("feature_details", None)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Prediction history ────────────────────────────────────────────

@app.get("/api/history/{ticker}", tags=["Monitoring"])
async def prediction_history(ticker: str, limit: int = 20):
    """Return recent predictions for a ticker."""
    try:
        from api.db import get_history
        return {"ticker": ticker.upper(), "predictions": get_history(ticker.upper(), limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/outcome", tags=["Monitoring"])
async def record_outcome(body: OutcomeUpdate):
    """Record actual outcome for a past prediction (for accuracy tracking)."""
    try:
        from api.db import update_outcome
        update_outcome(body.prediction_id, body.actual_beat, body.actual_price)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket streaming ───────────────────────────────────────────

@app.websocket("/ws/predict/{ticker}")
async def ws_predict(websocket: WebSocket, ticker: str,
                     use_lstm: bool = True, use_finbert: bool = False):
    """Stream model progress updates in real-time."""
    await websocket.accept()
    ticker = ticker.upper()

    async def send(step: str, status: str, data: dict = {}):
        await websocket.send_text(json.dumps({
            "step": step, "status": status, "data": data,
            "ts": datetime.utcnow().isoformat()
        }))

    try:
        output: dict = {"ticker": ticker, "generated": datetime.utcnow().isoformat()+"Z", "models": {}}

        await send("classifier", "running", {"msg": "Running stacked ensemble…"})
        try:
            from ml.ensemble.stacked_classifier import predict as ep
            output["models"]["classifier"] = ep(ticker)
            await send("classifier", "done", output["models"]["classifier"])
        except Exception as e:
            await send("classifier", "error", {"error": str(e)})

        if use_lstm:
            await send("lstm", "running", {"msg": "Running LSTM…"})
            try:
                from ml.models.price_predictor import predict as lp
                output["models"]["price_predictor"] = lp(ticker)
                await send("lstm", "done", output["models"]["price_predictor"])
            except Exception as e:
                await send("lstm", "error", {"error": str(e)})

        await send("sentiment", "running", {"msg": "Analysing headlines…"})
        try:
            from ml.models.sentiment_analyzer import analyze
            output["models"]["sentiment"] = analyze(ticker, use_finbert=use_finbert)
            await send("sentiment", "done", output["models"]["sentiment"])
        except Exception as e:
            await send("sentiment", "error", {"error": str(e)})

        await send("volatility", "running", {"msg": "Fitting GARCH…"})
        try:
            from ml.models.volatility_model import forecast_volatility
            output["models"]["volatility"] = forecast_volatility(ticker)
            await send("volatility", "done", output["models"]["volatility"])
        except Exception as e:
            await send("volatility", "error", {"error": str(e)})

        await send("drift", "running", {"msg": "Checking feature drift…"})
        try:
            from monitoring.drift_detector import DriftDetector
            from ml.data.pipeline import build_master_dataset
            detector = DriftDetector(ticker)
            df = build_master_dataset(ticker)
            report = detector.check(df.tail(60))
            output["drift"] = {"status": report["overall_status"], "retrain_recommended": report["retrain_recommended"]}
            await send("drift", "done", output["drift"])
        except Exception as e:
            await send("drift", "error", {"error": str(e)})

        output["composite"] = _composite(output["models"], {})
        await send("complete", "done", output)

    except WebSocketDisconnect:
        log.info(f"WS disconnected: {ticker}")
    except Exception as e:
        await send("error", "error", {"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
