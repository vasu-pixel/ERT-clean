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

# Map Polygon SIC descriptions to standard sectors for peer matching
SIC_TO_SECTOR_MAP = {
    "ELECTRONIC COMPUTERS": "Technology",
    "COMPUTER PROGRAMMING SERVICES": "Technology",
    "COMPUTER INTEGRATED SYSTEMS DESIGN": "Technology",
    "SEMICONDUCTORS & RELATED DEVICES": "Technology",
    "TELEPHONE COMMUNICATIONS": "Communication Services",
    "CABLE & OTHER PAY TELEVISION SERVICES": "Communication Services",
    "PHARMACEUTICAL PREPARATIONS": "Healthcare",
    "BIOLOGICAL PRODUCTS": "Healthcare",
    "MEDICAL DEVICES": "Healthcare",
    "RETAIL-DRUG STORES AND PROPRIETARY STORES": "Healthcare",
    "HOSPITAL & MEDICAL SERVICE PLANS": "Healthcare",
    "CRUDE PETROLEUM & NATURAL GAS": "Energy",
    "PETROLEUM REFINING": "Energy",
    "NATIONAL COMMERCIAL BANKS": "Financial Services",
    "SAVINGS INSTITUTION": "Financial Services",
    "SECURITY BROKERS & DEALERS": "Financial Services",
    "INSURANCE CARRIERS": "Financial Services",
    "MOTOR VEHICLES & PASSENGER CAR BODIES": "Consumer Cyclical",
    "RETAIL-EATING PLACES": "Consumer Cyclical",
    "RETAIL-APPAREL & ACCESSORY STORES": "Consumer Cyclical",
    "RETAIL-VARIETY STORES": "Consumer Defensive",
    "RETAIL-GROCERY STORES": "Consumer Defensive",
    "BEVERAGES": "Consumer Defensive",
    "AIRCRAFT": "Industrials",
    "INDUSTRIAL MACHINERY & EQUIPMENT": "Industrials",
    "ELECTRIC SERVICES": "Utilities",
    "NATURAL GAS TRANSMISSION": "Utilities",
    "REAL ESTATE INVESTMENT TRUSTS": "Real Estate",
}

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
    "operatingMargins": "operatingMargins",
    "profitMargins": "profitMargins",
    "revenueGrowth": "revenueGrowth",
    "earningsGrowth": "earningsQuarterlyGrowth",
    "freeCashflow": "freeCashflow",
    "operatingCashflow": "operatingCashflow",
    "totalRevenue": "totalRevenue",
    "netIncome": "netIncome",
    "ebitda": "ebitda",
    "trailingEps": "trailingEps",
    "sharesOutstanding": "sharesOutstanding",
    "dividendYield": "dividendYield",
    "beta": "beta",
    "enterpriseValue": "enterpriseValue",
    "enterpriseToEbitda": "enterpriseToEbitda",
    "totalAssets": "totalAssets",
    "totalDebt": "totalDebt",
    "totalCash": "totalCash",
    "totalLiabilities": "totalLiabilities",
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

            # Validate we got critical fields for accurate analysis
            required_fields = {
                "totalRevenue": fundamentals.get("totalRevenue"),
                "market_cap": fundamentals.get("market_cap"),
                "current_price": fundamentals.get("current_price"),
                "name": fundamentals.get("name")
            }

            missing_fields = [field for field, value in required_fields.items()
                            if value is None or (isinstance(value, (int, float)) and value <= 0)]

            if not missing_fields:
                logger.info(f"Successfully fetched {ticker} from Polygon.io with all critical fields")
                return fundamentals
            else:
                logger.warning(f"Polygon.io data incomplete for {ticker} (missing: {missing_fields}), falling back to yfinance")
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

        # Extract missing fields from financial statements if available
        if statements.get("income_statement") is not None and not statements["income_statement"].empty:
            income_stmt = statements["income_statement"]

            # Try to get Net Income from statement if missing from info
            if not fundamentals.get("netIncome"):
                for net_income_field in ["Net Income", "Net Income From Continuing Operation Net Minority Interest",
                                        "Normalized Income", "Net Income From Continuing And Discontinued Operation"]:
                    if net_income_field in income_stmt.index:
                        fundamentals["netIncome"] = income_stmt.loc[net_income_field].iloc[0]
                        logger.info(f"Extracted Net Income from statement for {ticker}: ${fundamentals['netIncome']:,.0f}")
                        break

        if statements.get("balance_sheet") is not None and not statements["balance_sheet"].empty:
            balance_sheet = statements["balance_sheet"]

            # Try to get Total Assets from balance sheet if missing
            if not fundamentals.get("totalAssets"):
                for assets_field in ["Total Assets", "Assets"]:
                    if assets_field in balance_sheet.index:
                        fundamentals["totalAssets"] = balance_sheet.loc[assets_field].iloc[0]
                        logger.info(f"Extracted Total Assets from balance sheet for {ticker}: ${fundamentals['totalAssets']:,.0f}")
                        break

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

    # Map SIC description to standard sector for peer matching
    sic_desc = details.get("sic_description", "")
    mapped_sector = SIC_TO_SECTOR_MAP.get(sic_desc, sic_desc)  # Use mapped sector or original

    # Map to ERT/yfinance compatible format
    merged = {
        # Company info
        "name": details.get("name"),
        "summary": details.get("description"),
        "sector": mapped_sector,  # Use mapped sector for peer matching
        "industry": sic_desc,  # Keep SIC as industry
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

    # Enterprise value calculation with actual cash data
    if merged.get("market_cap") and bal.get("liabilities"):
        # Try to get actual cash from yfinance (more reliable than Polygon)
        actual_cash = None
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker)
            balance_sheet = ticker_obj.balance_sheet

            if balance_sheet is not None and not balance_sheet.empty:
                # Try multiple field names for cash
                for cash_field in ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments',
                                  'cashAndCashEquivalents', 'cash_and_cash_equivalents']:
                    if cash_field in balance_sheet.index:
                        actual_cash = balance_sheet.loc[cash_field].iloc[0]
                        if actual_cash and actual_cash > 0:
                            logger.info(f"Using actual cash from yfinance for {ticker}: ${actual_cash:,.0f}")
                            break
        except Exception as e:
            logger.debug(f"Could not fetch cash from yfinance for {ticker}: {e}")

        # Fallback to estimation if yfinance fails
        if actual_cash is None or actual_cash <= 0:
            actual_cash = bal.get("current_assets", 0) * 0.3
            logger.debug(f"Using estimated cash (30% of current assets) for {ticker}: ${actual_cash:,.0f}")

        # Calculate total debt (long-term debt + current liabilities)
        total_debt = bal.get("liabilities", 0)

        merged["enterpriseValue"] = merged["market_cap"] + total_debt - actual_cash

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
