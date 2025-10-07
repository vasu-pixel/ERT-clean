"""Auxiliary data fetchers for the orchestration layer.

These helpers provide lightweight integrations for analyst data, holders,
recent headlines, and sentiment scoring. They can be upgraded to use premium
providers without changing the contract expected by the orchestrator.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

import logging

from . import news_scraper

logger = logging.getLogger(__name__)

# Minimal sentiment lexicon. Replace with an ML model when available.
POSITIVE_WORDS = {
    "beat",
    "beats",
    "growth",
    "strong",
    "outperform",
    "outperformed",
    "gain",
    "surge",
    "positive",
    "bullish",
    "optimistic",
    "record",
    "profit",
    "profits",
    "upgrade",
    "upgrades",
    "rally",
    "expands",
}

NEGATIVE_WORDS = {
    "miss",
    "missed",
    "weak",
    "decline",
    "declines",
    "fall",
    "falls",
    "loss",
    "losses",
    "negative",
    "bearish",
    "downgrade",
    "downgrades",
    "lawsuit",
    "regulatory",
    "risk",
    "risks",
    "cut",
    "cuts",
    "warning",
    "warnings",
    "probe",
}


def _score_sentiment(text: str) -> float:
    """Return a rudimentary sentiment score between -1 and 1."""
    if not text:
        return 0.0

    lower = text.lower()
    positive_hits = sum(word in lower for word in POSITIVE_WORDS)
    negative_hits = sum(word in lower for word in NEGATIVE_WORDS)

    if positive_hits == 0 and negative_hits == 0:
        return 0.0

    score = positive_hits - negative_hits
    total = positive_hits + negative_hits
    return max(min(score / total, 1.0), -1.0)


def _label_sentiment(score: float) -> str:
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def fetch_analyst_estimates(ticker: str) -> Dict[str, float | None]:
    """Return basic analyst estimate scaffolding for a ticker."""
    return {
        "ticker": ticker.upper(),
        "as_of": datetime.utcnow().isoformat(),
        "eps_next_quarter": None,
        "eps_next_year": None,
        "revenue_next_year": None,
        "consensus_rating": None,
    }


def fetch_institutional_holders(ticker: str) -> List[Dict[str, str | float | None]]:
    """Placeholder institutional holder list."""
    return []


@lru_cache(maxsize=256)
def fetch_recent_headlines(ticker: str, limit: int = 12) -> List[Dict[str, Optional[str]]]:
    """Fetch recent headlines for a ticker using yfinance with sentiment scores."""
    headlines: List[Dict[str, Optional[str]]] = []

    scraped = news_scraper.scrape_headlines(ticker, limit=limit)
    for item in scraped:
        text = " ".join(filter(None, [item.get("headline"), item.get("summary")]))
        score = _score_sentiment(text)
        item["sentiment_score"] = score
        item["sentiment_label"] = _label_sentiment(score)
        item.setdefault("ticker", ticker.upper())
        item.setdefault("published_at", datetime.utcnow().isoformat())
        headlines.append(item)

    return headlines


def summarize_headline_sentiment(headlines: List[Dict[str, Optional[str]]]) -> Dict[str, float | int]:
    """Aggregate sentiment stats from headline list."""
    if not headlines:
        return {
            "count": 0,
            "average_score": 0.0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        }

    total = 0.0
    positive = neutral = negative = 0
    for item in headlines:
        score = float(item.get("sentiment_score") or 0.0)
        total += score
        label = item.get("sentiment_label") or _label_sentiment(score)
        if label == "positive":
            positive += 1
        elif label == "negative":
            negative += 1
        else:
            neutral += 1

    count = len(headlines)
    return {
        "count": count,
        "average_score": total / count if count else 0.0,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
    }
