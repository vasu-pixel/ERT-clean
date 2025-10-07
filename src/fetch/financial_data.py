from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

KEY_FIELDS = {
    "name": "longName",
    "summary": "longBusinessSummary",
    "sector": "sector",
    "industry": "industry",
    "market_cap": "marketCap",
    "current_price": "currentPrice",
    "currency": "currency",
    "trailingPE": "trailingPE",
    "forwardPE": "forwardPE",
    "pegRatio": "pegRatio",
    "priceToSales": "priceToSalesTrailing12Months",
    "returnOnEquity": "returnOnEquity",
    "grossMargins": "grossMargins",
    "ebitdaMargins": "ebitdaMargins",
    "revenueGrowth": "revenueGrowth",
    "earningsGrowth": "earningsQuarterlyGrowth",
    "freeCashflow": "freeCashflow",
    "operatingCashflow": "operatingCashflow",
    "totalRevenue": "totalRevenue",
    "trailingEps": "trailingEps",
    "sharesOutstanding": "sharesOutstanding",
    "dividendYield": "dividendYield",
    "beta": "beta",
    "enterpriseValue": "enterpriseValue",
    "enterpriseToEbitda": "enterpriseToEbitda",
}


def _fetch_yf_info(ticker: str) -> Dict[str, Optional[float]]:
    stock = yf.Ticker(ticker)
    info = stock.info

    data: Dict[str, Optional[float]] = {}
    for key, yf_key in KEY_FIELDS.items():
        data[key] = info.get(yf_key)

    try:
        history = stock.history(period="2y")
    except Exception as exc:
        logger.warning("Failed to fetch price history for %s: %s", ticker, exc)
        history = pd.DataFrame()

    data["price_history"] = history
    return data


def _fetch_financial_statements(stock: yf.Ticker) -> Dict[str, pd.DataFrame]:
    statements = {}
    try:
        statements["income_statement"] = stock.financials
        statements["balance_sheet"] = stock.balance_sheet
        statements["cash_flow"] = stock.cashflow
        statements["quarterly_financials"] = stock.quarterly_financials
        statements["quarterly_cash_flow"] = stock.quarterly_cashflow
    except Exception as exc:
        logger.warning("Failed to fetch financial statements: %s", exc)
    return statements


def get_fundamentals(ticker: str) -> Dict[str, object]:
    ticker = ticker.upper()

    try:
        inlined = _fetch_yf_info(ticker)
        stock = yf.Ticker(ticker)
        statements = _fetch_financial_statements(stock)

        fundamentals = {
            **inlined,
            "statements": statements,
        }

        # Basic validation: ensure we have core metrics
        if fundamentals.get("totalRevenue") in (None, 0):
            logger.warning("Missing totalRevenue for %s", ticker)
        if fundamentals.get("trailingEps") in (None, 0):
            logger.warning("Missing EPS for %s", ticker)

        return fundamentals

    except Exception as exc:
        logger.error("Failed to fetch fundamentals for %s: %s", ticker, exc)
        return {}
