"""
ml/models/sentiment_analyzer.py
─────────────────────────────────
Dual-model news sentiment pipeline:

1. VADER  — fast rule-based baseline (no GPU needed)
2. FinBERT — fine-tuned BERT on financial text (ProsusAI/finbert)
             runs on CPU or GPU automatically

Fetches recent headlines from Yahoo Finance RSS feed
and returns aggregated sentiment scores.

Usage:
  python ml/models/sentiment_analyzer.py --ticker AAPL
"""

import sys
import argparse
import warnings
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# VADER — always available
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# FinBERT — lazy-loaded on first use (large download ~440MB)
_finbert_tokenizer = None
_finbert_model     = None


def _load_finbert():
    """Load FinBERT lazily so the module imports instantly."""
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        print("[sentiment] Loading FinBERT (first run downloads ~440MB)…")
        model_name = "ProsusAI/finbert"
        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _finbert_model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        _finbert_model.eval()
        print("[sentiment] FinBERT loaded ✓")
    return _finbert_tokenizer, _finbert_model


# ─────────────────────────────────────────────────────────────
# News fetching
# ─────────────────────────────────────────────────────────────

def fetch_news_headlines(ticker: str, max_items: int = 30) -> list[dict]:
    """
    Pull recent news headlines for `ticker` from Yahoo Finance RSS.
    Returns list of {"title": str, "published": str, "source": str}.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
    ]

    headlines = []
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200:
                continue
            # Simple XML parse without lxml dependency
            items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
            for item in items[:max_items]:
                title = re.search(r"<title>(.*?)</title>", item)
                pubdate = re.search(r"<pubDate>(.*?)</pubDate>", item)
                if title:
                    headlines.append({
                        "title":     _clean(title.group(1)),
                        "published": pubdate.group(1) if pubdate else "",
                        "source":    "Yahoo Finance",
                    })
            if headlines:
                break
        except Exception as e:
            print(f"[sentiment] RSS fetch error: {e}")
            continue

    # Fallback: use yfinance news
    if not headlines:
        try:
            stock = yf.Ticker(ticker)
            news  = stock.news or []
            for n in news[:max_items]:
                headlines.append({
                    "title":     n.get("title", ""),
                    "published": str(n.get("providerPublishTime", "")),
                    "source":    n.get("publisher", ""),
                })
        except Exception as e:
            print(f"[sentiment] yfinance news fallback error: {e}")

    print(f"[sentiment] Fetched {len(headlines)} headlines for {ticker}")
    return headlines


def _clean(text: str) -> str:
    """Strip HTML entities and whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()


# ─────────────────────────────────────────────────────────────
# VADER scoring
# ─────────────────────────────────────────────────────────────

def score_vader(headlines: list[dict]) -> list[dict]:
    """Adds VADER compound score [-1, +1] to each headline."""
    vader = SentimentIntensityAnalyzer()
    for h in headlines:
        scores = vader.polarity_scores(h["title"])
        h["vader_compound"] = scores["compound"]
        h["vader_pos"]      = scores["pos"]
        h["vader_neg"]      = scores["neg"]
        h["vader_neu"]      = scores["neu"]
    return headlines


# ─────────────────────────────────────────────────────────────
# FinBERT scoring
# ─────────────────────────────────────────────────────────────

def score_finbert(headlines: list[dict], batch_size: int = 8) -> list[dict]:
    """
    Adds FinBERT sentiment label + probabilities to each headline.
    Labels: positive / negative / neutral
    """
    import torch

    tokenizer, model = _load_finbert()
    texts = [h["title"] for h in headlines]

    all_probs  = []
    all_labels = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        # FinBERT label order: positive(0), negative(1), neutral(2)
        for p in probs:
            all_probs.append(p)
            all_labels.append(["positive", "negative", "neutral"][np.argmax(p)])

    for h, prob, label in zip(headlines, all_probs, all_labels):
        h["finbert_label"]    = label
        h["finbert_positive"] = round(float(prob[0]), 4)
        h["finbert_negative"] = round(float(prob[1]), 4)
        h["finbert_neutral"]  = round(float(prob[2]), 4)
        # Composite: positive - negative
        h["finbert_score"]    = round(float(prob[0] - prob[1]), 4)

    return headlines


# ─────────────────────────────────────────────────────────────
# Aggregate scores
# ─────────────────────────────────────────────────────────────

def aggregate_sentiment(headlines: list[dict], use_finbert: bool = True) -> dict:
    """
    Compute portfolio-level sentiment metrics from headline scores.
    Returns a summary dict used by the API.
    """
    if not headlines:
        return {
            "error": "No headlines available",
            "overall_sentiment": "NEUTRAL",
            "sentiment_score": 0.0,
        }

    vader_scores   = [h.get("vader_compound", 0) for h in headlines]
    avg_vader      = np.mean(vader_scores)
    vader_pos_pct  = sum(1 for s in vader_scores if s > 0.05) / len(vader_scores)
    vader_neg_pct  = sum(1 for s in vader_scores if s < -0.05) / len(vader_scores)

    result = {
        "ticker":            headlines[0].get("ticker", ""),
        "headline_count":    len(headlines),
        "vader_avg":         round(avg_vader, 4),
        "vader_bullish_pct": round(vader_pos_pct * 100, 1),
        "vader_bearish_pct": round(vader_neg_pct * 100, 1),
    }

    if use_finbert and all("finbert_score" in h for h in headlines):
        fb_scores    = [h["finbert_score"] for h in headlines]
        avg_fb       = np.mean(fb_scores)
        fb_pos_pct   = sum(1 for h in headlines if h["finbert_label"] == "positive") / len(headlines)
        fb_neg_pct   = sum(1 for h in headlines if h["finbert_label"] == "negative") / len(headlines)
        fb_neu_pct   = 1 - fb_pos_pct - fb_neg_pct

        result.update({
            "finbert_avg":          round(avg_fb, 4),
            "finbert_bullish_pct":  round(fb_pos_pct * 100, 1),
            "finbert_bearish_pct":  round(fb_neg_pct * 100, 1),
            "finbert_neutral_pct":  round(fb_neu_pct * 100, 1),
        })

        # Blend VADER (40%) + FinBERT (60%)
        blended = 0.4 * avg_vader + 0.6 * avg_fb
    else:
        blended = avg_vader

    # Map to label
    if blended > 0.10:
        label = "BULLISH"
    elif blended < -0.10:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    result["overall_sentiment"] = label
    result["sentiment_score"]   = round(blended, 4)
    result["top_headlines"]     = [
        {"title": h["title"], "score": round(h.get("finbert_score", h.get("vader_compound", 0)), 3)}
        for h in sorted(headlines, key=lambda x: abs(x.get("finbert_score", x.get("vader_compound", 0))), reverse=True)[:5]
    ]

    return result


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def analyze(ticker: str, use_finbert: bool = True) -> dict:
    """Full sentiment pipeline: fetch → VADER → FinBERT → aggregate."""
    headlines = fetch_news_headlines(ticker)
    for h in headlines:
        h["ticker"] = ticker

    headlines = score_vader(headlines)

    if use_finbert:
        try:
            headlines = score_finbert(headlines)
        except Exception as e:
            print(f"[sentiment] FinBERT failed ({e}), using VADER only")
            use_finbert = False

    summary = aggregate_sentiment(headlines, use_finbert=use_finbert)
    summary["model"] = "FinBERT + VADER Ensemble" if use_finbert else "VADER"
    return summary


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial News Sentiment Analyzer")
    parser.add_argument("--ticker",    default="AAPL")
    parser.add_argument("--no-finbert", action="store_true", help="Use VADER only (faster)")
    args = parser.parse_args()

    result = analyze(args.ticker, use_finbert=not args.no_finbert)
    print(f"\n── Sentiment Analysis: {args.ticker} ─────────────────────")
    for k, v in result.items():
        if k != "top_headlines":
            print(f"  {k:<28} {v}")
    print("\n  Top Headlines:")
    for h in result.get("top_headlines", []):
        score = h['score']
        bar = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
        print(f"  {bar}  [{score:+.3f}]  {h['title'][:80]}")
