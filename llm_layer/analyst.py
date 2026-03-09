"""
llm_layer/analyst.py
─────────────────────
LLM-powered analyst commentary layer.

Takes the raw ML model outputs (probabilities, signals, metrics)
and generates institutional-quality written analysis using
the Anthropic Claude API.

This demonstrates the modern "ML + LLM" pipeline pattern
that JPMorgan's AI/ML teams are actively building —
combining quantitative signals with natural language generation.

Why this impresses JPM interviewers:
  It shows you understand that ML models produce outputs that
  need human-readable interpretation. You're not just building
  models — you're building *products* that decision-makers can use.

Usage:
  from llm_layer.analyst import generate_commentary
  text = generate_commentary("AAPL", ml_results)
"""

import os
import json
from datetime import datetime
from typing import Optional
import httpx


ANTHROPIC_API = "https://api.anthropic.com/v1/messages"


async def generate_commentary(
    ticker: str,
    ml_results: dict,
    style: str = "institutional",   # "institutional" | "brief" | "technical"
) -> dict:
    """
    Generate written analyst commentary from ML model outputs.

    Parameters
    ----------
    ticker      : stock ticker
    ml_results  : output dict from ml/predict.py run_all()
    style       : writing style for the output

    Returns
    -------
    dict with keys: summary, key_risks, key_catalysts, recommendation
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set", "fallback": _rule_based_commentary(ticker, ml_results)}

    composite = ml_results.get("composite", {})
    models    = ml_results.get("models", {})
    macro     = ml_results.get("macro", {})

    clf   = models.get("classifier", {})
    lstm  = models.get("price_predictor", {})
    sent  = models.get("sentiment", {})
    vol   = models.get("volatility", {})

    STYLE_PROMPTS = {
        "institutional": "Write in the style of a senior JPMorgan equity research analyst. Be precise, data-driven, and use professional financial terminology. Use specific numbers from the data provided.",
        "brief":         "Write a concise 3-bullet executive summary for a portfolio manager who has 30 seconds to read this.",
        "technical":     "Write a technical ML model interpretation for a quantitative researcher. Focus on model confidence, feature drivers, and statistical significance.",
    }

    system = f"""You are a senior quantitative analyst at JPMorgan Chase.
{STYLE_PROMPTS.get(style, STYLE_PROMPTS['institutional'])}
Always respond with valid JSON only. No markdown, no preamble."""

    user = f"""Generate analyst commentary for {ticker} based on these ML model outputs:

COMPOSITE SIGNAL: {composite.get('signal')} (score: {composite.get('score')})
VOTE REASONS: {json.dumps(composite.get('reasons', []))}

EARNINGS CLASSIFIER (Stacked Ensemble XGB+LGB+CAT):
  Beat probability: {clf.get('beat_probability')}%
  Prediction: {clf.get('prediction')}
  Confidence: {clf.get('confidence')}%
  Model: {clf.get('model')}

LSTM PRICE PREDICTION:
  Current price: ${lstm.get('current_price')}
  Predicted price ({lstm.get('forward_days', 5)}d): ${lstm.get('predicted_price')}
  Expected return: {lstm.get('predicted_return')}%
  Historical directional accuracy: {lstm.get('model_directional_acc')}%

NEWS SENTIMENT ({sent.get('model')}):
  Overall: {sent.get('overall_sentiment')}
  Score: {sent.get('sentiment_score')}
  Bullish headlines: {sent.get('finbert_bullish_pct') or sent.get('vader_bullish_pct')}%

GARCH VOLATILITY:
  Current vol: {vol.get('current_conditional_vol')}% (annualised)
  Regime: {vol.get('regime')}
  5d forecast: {vol.get('garch_forecast_5d')}%
  VaR 99% 1d: {vol.get('var_99_1d')}%

MACRO CONTEXT:
  VIX: {macro.get('vix')} ({macro.get('vix_regime')})
  10Y rate: {macro.get('t10y')}% ({macro.get('rate_regime')})
  SPX 20d return: {macro.get('spx_ret_20d')}%

Return this exact JSON structure:
{{
  "summary": "2-3 sentence institutional summary of the overall signal and key drivers",
  "earnings_outlook": "1-2 sentence view on the earnings beat/miss probability and what's driving it",
  "price_outlook": "1-2 sentence LSTM prediction interpretation with confidence context",
  "sentiment_read": "1 sentence on news sentiment and what it implies",
  "risk_assessment": "1-2 sentences on volatility regime and key downside risks",
  "macro_context": "1 sentence on macro tailwinds or headwinds",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "key_catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"],
  "recommendation": "BUY | SELL | HOLD",
  "conviction": "HIGH | MEDIUM | LOW",
  "one_liner": "Single sentence suitable for a Bloomberg headline"
}}"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                ANTHROPIC_API,
                headers={
                    "Content-Type":  "application/json",
                    "x-api-key":     api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system":     system,
                    "messages":   [{"role": "user", "content": user}],
                }
            )
        data = resp.json()
        raw  = data["content"][0]["text"].replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["generated_at"] = datetime.utcnow().isoformat()
        result["model"]        = "claude-sonnet-4-20250514"
        return result

    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}", "fallback": _rule_based_commentary(ticker, ml_results)}
    except Exception as e:
        return {"error": str(e), "fallback": _rule_based_commentary(ticker, ml_results)}


def _rule_based_commentary(ticker: str, ml_results: dict) -> dict:
    """
    Fallback rule-based commentary when LLM is unavailable.
    Deterministic — always produces same output for same inputs.
    """
    composite = ml_results.get("composite", {})
    signal    = composite.get("signal", "NEUTRAL")
    score     = composite.get("score", 0)
    clf       = ml_results.get("models", {}).get("classifier", {})
    vol       = ml_results.get("models", {}).get("volatility", {})

    beat_p  = clf.get("beat_probability", 50)
    regime  = vol.get("regime", "NORMAL")

    rec  = "BUY"  if signal in {"BULLISH","STRONG_BUY"} else \
           "SELL" if signal in {"BEARISH","STRONG_SELL"} else "HOLD"
    conv = "HIGH" if abs(score) > 0.5 else ("MEDIUM" if abs(score) > 0.25 else "LOW")

    return {
        "summary": f"ML composite signal for {ticker} is {signal} with a score of {score:+.2f}. "
                   f"The stacked ensemble assigns a {beat_p}% probability of an earnings beat. "
                   f"Volatility regime is {regime}.",
        "recommendation": rec,
        "conviction":     conv,
        "one_liner":      f"{ticker} rates {rec} based on {signal} composite ML signal (score: {score:+.2f})",
        "generated_at":   datetime.utcnow().isoformat(),
        "model":          "rule_based_fallback",
    }


# ─────────────────────────────────────────────────────────────────
# Sync wrapper for non-async contexts
# ─────────────────────────────────────────────────────────────────

def generate_commentary_sync(ticker: str, ml_results: dict, style: str = "institutional") -> dict:
    """Synchronous wrapper — use in scripts, not in FastAPI async routes."""
    import asyncio
    return asyncio.run(generate_commentary(ticker, ml_results, style))


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    # Test with mock data
    mock_results = {
        "composite": {"signal": "BULLISH", "score": 0.5, "reasons": ["Beat prob 68%", "LSTM +2.1%"]},
        "models": {
            "classifier":      {"beat_probability": 68.0, "prediction": "UP", "confidence": 68.0, "model": "Stacked Ensemble"},
            "price_predictor": {"current_price": 185.0, "predicted_price": 189.0, "predicted_return": 2.16, "forward_days": 5, "model_directional_acc": 56.2},
            "sentiment":       {"overall_sentiment": "BULLISH", "sentiment_score": 0.22, "model": "VADER"},
            "volatility":      {"current_conditional_vol": 18.5, "regime": "NORMAL", "garch_forecast_5d": 19.2, "var_99_1d": -2.8},
        },
        "macro": {"vix": 16.2, "vix_regime": "NORMAL", "t10y": 4.3, "rate_regime": "HIGH", "spx_ret_20d": 2.1},
    }

    print(f"[llm] Generating commentary for {ticker}…")
    result = generate_commentary_sync(ticker, mock_results)
    print(json.dumps(result, indent=2))
