from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Try to import Polygon.io integration
try:
    from .polygon_data import get_polygon_fundamentals
    POLYGON_AVAILABLE = bool(os.getenv("POLYGON_API_KEY"))
    if POLYGON_AVAILABLE:
        logger.info("Polygon.io integration enabled")
except ImportError:
    POLYGON_AVAILABLE = False
    logger.warning("Polygon.io integration not available")

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

    # Try Polygon.io first (Starter plan with unlimited requests)
    if POLYGON_AVAILABLE:
        try:
            logger.info(f"Fetching {ticker} from Polygon.io (primary source)")
            polygon_data = get_polygon_fundamentals(ticker)

            # Merge Polygon data with yfinance format for compatibility
            fundamentals = _merge_polygon_data(polygon_data, ticker)

            # Validate we got good data
            if fundamentals.get("totalRevenue") and fundamentals.get("totalRevenue") > 0:
                logger.info(f"Successfully fetched {ticker} from Polygon.io")
                return fundamentals
            else:
                logger.warning(f"Polygon.io data incomplete for {ticker}, falling back to yfinance")
        except Exception as exc:
            logger.warning(f"Polygon.io failed for {ticker}: {exc}, falling back to yfinance")

    # Fallback to yfinance
    try:
        logger.info(f"Fetching {ticker} from yfinance (fallback source)")
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


def _merge_polygon_data(polygon_data: Dict, ticker: str) -> Dict[str, object]:
    """Convert Polygon.io data to ERT format (compatible with yfinance structure)"""
    details = polygon_data.get("ticker_details", {})
    key_metrics = polygon_data.get("key_metrics", {})
    latest_fin = polygon_data.get("latest_financials", {})
    prev_close = polygon_data.get("previous_close", {})
    ratios = polygon_data.get("calculated_ratios", {})

    # Extract income statement data
    inc = latest_fin.get("income_statement", {}) if latest_fin else {}
    bal = latest_fin.get("balance_sheet", {}) if latest_fin else {}
    cf = latest_fin.get("cash_flow", {}) if latest_fin else {}

    # Map to ERT/yfinance compatible format
    merged = {
        # Company info
        "name": details.get("name"),
        "summary": details.get("description"),
        "sector": details.get("sic_description"),  # Approximate - Polygon doesn't have direct sector
        "industry": details.get("sic_description"),
        "currency": details.get("currency_name"),

        # Market data
        "market_cap": details.get("market_cap"),
        "current_price": prev_close.get("close") if prev_close else None,
        "sharesOutstanding": details.get("weighted_shares_outstanding"),

        # Income statement
        "totalRevenue": inc.get("revenues"),
        "grossProfit": inc.get("gross_profit"),
        "operatingIncome": inc.get("operating_income_loss"),
        "netIncome": inc.get("net_income_loss_attributable_to_parent"),
        "ebitda": inc.get("operating_income_loss"),  # Approximation

        # Balance sheet
        "totalAssets": bal.get("assets"),
        "totalLiabilities": bal.get("liabilities"),
        "totalEquity": bal.get("equity_attributable_to_parent"),
        "totalCurrentAssets": bal.get("current_assets"),
        "totalCurrentLiabilities": bal.get("current_liabilities"),

        # Cash flow
        "operatingCashflow": cf.get("net_cash_flow_from_operating_activities"),
        "freeCashflow": cf.get("net_cash_flow_from_operating_activities"),  # Will refine later

        # Ratios (calculated)
        "grossMargins": ratios.get("gross_margin"),
        "operatingMargins": ratios.get("operating_margin"),
        "profitMargins": ratios.get("net_margin"),

        # Price history
        "price_history": polygon_data.get("price_history", pd.DataFrame()),

        # Statements (for compatibility)
        "statements": {
            "polygon_financials": polygon_data.get("financials", []),
            "polygon_source": True,
        },

        # Additional polygon-specific data
        "polygon_data": {
            "cik": details.get("cik"),
            "sic_code": details.get("sic_code"),
            "total_employees": details.get("total_employees"),
            "homepage_url": details.get("homepage_url"),
            "dividends": polygon_data.get("dividends", []),
            "splits": polygon_data.get("splits", []),
        },
    }

    # Calculate additional derived metrics
    revenue = inc.get("revenues")
    shares = details.get("weighted_shares_outstanding")
    net_income = inc.get("net_income_loss_attributable_to_parent")
    price = prev_close.get("close") if prev_close else None

    if revenue and shares and revenue > 0 and shares > 0:
        merged["revenuePerShare"] = revenue / shares

    if net_income and shares and shares > 0:
        merged["trailingEps"] = net_income / shares

        # Calculate PE if we have price
        if price and merged["trailingEps"] and merged["trailingEps"] > 0:
            merged["trailingPE"] = price / merged["trailingEps"]

    # Enterprise value approximation
    if merged.get("market_cap") and bal.get("liabilities"):
        cash = bal.get("current_assets", 0) * 0.3  # Rough estimate of cash
        merged["enterpriseValue"] = merged["market_cap"] + bal.get("liabilities", 0) - cash

        # EV/Revenue
        if revenue and revenue > 0:
            merged["enterpriseToRevenue"] = merged["enterpriseValue"] / revenue

        # EV/EBITDA
        if merged.get("ebitda") and merged["ebitda"] > 0:
            merged["enterpriseToEbitda"] = merged["enterpriseValue"] / merged["ebitda"]

    # Return on metrics
    if net_income and merged.get("totalEquity") and merged["totalEquity"] > 0:
        merged["returnOnEquity"] = net_income / merged["totalEquity"]

    if net_income and merged.get("totalAssets") and merged["totalAssets"] > 0:
        merged["returnOnAssets"] = net_income / merged["totalAssets"]

    # Current ratio
    curr_assets = bal.get("current_assets")
    curr_liab = bal.get("current_liabilities")
    if curr_assets and curr_liab and curr_liab > 0:
        merged["currentRatio"] = curr_assets / curr_liab

    # Debt to equity
    if merged.get("totalLiabilities") and merged.get("totalEquity") and merged["totalEquity"] > 0:
        merged["debtToEquity"] = merged["totalLiabilities"] / merged["totalEquity"]

    logger.info(f"Merged Polygon.io data for {ticker}: {len([k for k, v in merged.items() if v is not None])} fields populated")

    return merged
