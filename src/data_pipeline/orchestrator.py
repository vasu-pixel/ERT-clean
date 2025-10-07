"""Core data orchestration utilities for ERT."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List

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

    def __init__(self, cache_ttl: timedelta | None = None, deterministic_config: Optional[Dict[str, Dict]] = None) -> None:
        self._cache: Dict[str, tuple[datetime, CompanyDataset]] = {}
        self._cache_ttl = cache_ttl or timedelta(minutes=30)
        self._document_store = FileBackedDocumentStore()
        self._sec_ingestor = SECIngestor(self._document_store)
        self._deterministic_config = deterministic_config or {}

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
        price_history = fundamentals.pop("price_history", None)
        statements = fundamentals.pop("statements", {})

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

        supplemental["peer_metrics"] = self._collect_peer_metrics(ticker, snapshot.sector)

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

    def _collect_peer_metrics(self, ticker: str, sector: str, limit: int = 5) -> List[Dict[str, Optional[float]]]:
        sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD'],
            'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA'],
        }

        peer_candidates = sector_mapping.get(sector, [])
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

        return peers
