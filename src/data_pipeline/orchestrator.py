"""Core data orchestration utilities for ERT."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any

import requests
from src.fetch.financial_data import get_fundamentals
from dataclasses import asdict

from src.retrieval import FileBackedDocumentStore, SECIngestor
from src.analyze.deterministic import run_deterministic_models
from src.analyze.financial_model import load_segment_config, build_segment_forecast, build_three_statement_model

from .models import CompanyDataset, CompanySnapshot, FinancialDataset
from . import sources

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """Fetches and normalizes company data with lightweight caching."""

    def __init__(
        self,
        cache_ttl: timedelta | None = None,
        deterministic_config: Optional[Dict[str, Dict]] = None,
        peer_mapping: Optional[Dict[str, List[str]]] = None,
        config: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        config = config or {}
        self._cache: Dict[str, tuple[datetime, CompanyDataset]] = {}
        self._cache_ttl = cache_ttl or timedelta(minutes=30)
        self._document_store = FileBackedDocumentStore()
        self._sec_ingestor = SECIngestor(self._document_store)
        det_config = deterministic_config or config.get("deterministic_config") or {}
        self._deterministic_config = det_config
        self._peer_mapping = peer_mapping or config.get("peer_mapping") or {}

    def refresh_company_data(self, ticker: str, force: bool = False) -> CompanyDataset:
        ticker = ticker.upper()
        if not force and ticker in self._cache:
            cached_at, dataset = self._cache[ticker]
            if datetime.utcnow() - cached_at <= self._cache_ttl:
                return dataset

        fundamentals_raw = get_fundamentals(ticker)
        if not fundamentals_raw:
            raise ValueError(f"Unable to fetch fundamentals for {ticker}")

        fundamentals: Dict[str, Optional[float]] = dict(fundamentals_raw)
        price_history_raw = fundamentals.pop("price_history", None)
        statements = fundamentals.pop("statements", {})

        # Convert DataFrame to dict for JSON serialization
        price_history = None
        if price_history_raw is not None:
            try:
                if hasattr(price_history_raw, 'to_dict'):
                    price_history = price_history_raw.to_dict('list')
                else:
                    price_history = price_history_raw
            except Exception as e:
                logger.warning(f"Could not serialize price_history for {ticker}: {e}")
                price_history = None

        snapshot = CompanySnapshot(
            ticker=ticker,
            name=fundamentals.get("name", ticker),
            sector=fundamentals.get("sector", "N/A"),
            industry=fundamentals.get("industry", "N/A"),
            market_cap=fundamentals.get("market_cap"),
            current_price=fundamentals.get("current_price"),
            currency=fundamentals.get("currency"),
            as_of=datetime.utcnow(),
        )

        ratios = self._extract_ratios(fundamentals)

        segment_config = load_segment_config(ticker)

        recent_headlines = sources.fetch_recent_headlines(ticker)
        supplemental = {
            "analyst_estimates": sources.fetch_analyst_estimates(ticker),
            "institutional_holders": sources.fetch_institutional_holders(ticker),
            "recent_headlines": recent_headlines,
            "headline_sentiment": sources.summarize_headline_sentiment(recent_headlines),
            "segment_config": segment_config,
        }

        try:
            peer_metrics, peer_candidates = self._collect_peer_metrics(ticker, snapshot.sector)
            supplemental["peer_metrics"] = peer_metrics
            supplemental["peer_candidates"] = peer_candidates
        except ValueError as exc:
            raise ValueError(f"Failed to collect peer metrics for {ticker}: {exc}") from exc

        try:
            sec_chunks = self._sec_ingestor.ingest_latest_10k(ticker)
            supplemental["sec_filing"] = {
                "chunks": len(sec_chunks),
                "available": bool(sec_chunks),
            }
        except Exception as exc:
            logger.warning("SEC ingestion failed for %s: %s", ticker, exc)

        dataset = CompanyDataset(
            snapshot=snapshot,
            financials=FinancialDataset(
                fundamentals=fundamentals,
                ratios=ratios,
                price_history=price_history,
            ),
            metadata={"source": "yfinance", "fetched_at": datetime.utcnow().isoformat()},
            supplemental=supplemental,
            raw_payload={
                "fundamentals": fundamentals,
                "statements": statements,
            },
        )

        deterministic = run_deterministic_models(dataset, self._deterministic_config)
        dataset.supplemental["deterministic"] = asdict(deterministic)

        segment_forecast = build_segment_forecast(dataset, years=len(deterministic.fcf_projection.get("schedule", [])) or 5, config=segment_config)
        three_statement = build_three_statement_model(dataset, deterministic.fcf_projection, segment_forecast, segment_config)

        dataset.supplemental["segment_forecast"] = [s.__dict__ for s in segment_forecast]
        dataset.supplemental["three_statement_model"] = three_statement

        self._cache[ticker] = (datetime.utcnow(), dataset)
        return dataset

    def _extract_ratios(self, fundamentals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        keys = [
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToSales",
            "returnOnEquity",
            "grossMargins",
            "ebitdaMargins",
            "revenueGrowth",
            "earningsGrowth",
            "freeCashflow",
            "dividendYield",
        ]
        return {key: fundamentals.get(key) for key in keys}

    def _collect_peer_metrics(
        self, ticker: str, sector: str, limit: int = 5
    ) -> tuple[List[Dict[str, Optional[float]]], List[str]]:
        peer_candidates: List[str] = []

        configured_peers = (self._peer_mapping or {}).get(sector)
        if configured_peers:
            peer_candidates = [p.upper() for p in configured_peers if isinstance(p, str)]

        if not peer_candidates:
            try:
                peer_candidates = self._fetch_peer_candidates(ticker, sector)
            except ValueError as e:
                logger.warning(f"FMP peer lookup failed: {e}, using fallback peer list")
                # Fallback to common peers by sector
                peer_candidates = self._get_fallback_peers(ticker, sector)

        if not peer_candidates:
            logger.warning(f"No peers found for {ticker}, using empty peer list")
            return [], []

        peers: List[Dict[str, Optional[float]]] = []
        for peer in peer_candidates:
            if peer.upper() == ticker.upper():
                continue
            try:
                fundamentals = get_fundamentals(peer)
                if not fundamentals:
                    continue
                peers.append(
                    {
                        "ticker": peer,
                        "market_cap": fundamentals.get("market_cap"),
                        "enterprise_value": fundamentals.get("enterpriseValue"),
                        "revenue_growth": fundamentals.get("revenueGrowth"),
                        "ebitda_margin": fundamentals.get("ebitdaMargins"),
                        "trailing_pe": fundamentals.get("trailingPE"),
                        "forward_pe": fundamentals.get("forwardPE"),
                        "enterprise_to_ebitda": fundamentals.get("enterpriseToEbitda"),
                        "free_cash_flow": fundamentals.get("freeCashflow"),
                    }
                )
            except Exception as exc:
                logger.debug("Failed to fetch peer metrics for %s: %s", peer, exc)
            if len(peers) >= limit:
                break

        if not peers:
            logger.warning(f"Unable to assemble peer metrics for {ticker}, returning empty peer list")
            return [], peer_candidates

        return peers, peer_candidates

    def _get_fallback_peers(self, ticker: str, sector: str) -> List[str]:
        """Fallback peer list when FMP API is unavailable"""
        # Common peers by sector
        fallback_map = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA", "AMZN"],
            "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "TMO", "CVS", "CI"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT"],
            "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK"],
            "Communication Services": ["META", "GOOGL", "DIS", "NFLX", "T", "VZ"],
            "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "CL"],
            "Industrials": ["BA", "CAT", "HON", "UNP", "GE", "LMT"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX"],
            "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O"],
            "Basic Materials": ["LIN", "APD", "SHW", "NEM", "FCX", "DOW"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"]
        }

        # Get peers for sector, excluding the ticker itself
        peers = fallback_map.get(sector, ["SPY"])  # Default to SPY if sector unknown
        logger.info(f"Using fallback peers for {ticker} in {sector}: {peers[:5]}")
        return [p for p in peers if p.upper() != ticker.upper()][:5]

    def _fetch_peer_candidates(self, ticker: str, sector: str) -> List[str]:
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.error("FMP_API_KEY missing from environment; cannot perform live peer lookup.")
            raise ValueError("Live peer lookup disabled: set FMP_API_KEY in environment.")

        url = f"https://financialmodelingprep.com/api/v3/stock_peers?symbol={ticker}&apikey={api_key}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json() or {}
        except Exception as exc:
            raise ValueError(f"Live peer lookup failed for {ticker}: {exc}") from exc

        peers_raw = []
        if isinstance(payload, dict):
            peers_raw = payload.get("peersList") or payload.get("peers") or []
        elif isinstance(payload, list) and payload:
            # some endpoints return list with dict entry at index 0
            first = payload[0]
            if isinstance(first, dict):
                peers_raw = first.get("peersList") or first.get("peers") or []

        peers = [p.upper() for p in peers_raw if isinstance(p, str) and p.strip()]

        if not peers:
            detail = f" in sector '{sector}'" if sector else ""
            raise ValueError(
                f"No peers returned for {ticker}{detail}. Verify upstream market data inputs."
            )

        return peers
