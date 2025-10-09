"""
Simplified Polygon.io integration - focuses on what actually works reliably
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

import pandas as pd
from polygon import RESTClient

logger = logging.getLogger(__name__)


def get_polygon_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch financial data from Polygon.io - simplified robust version
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.warning("POLYGON_API_KEY not set")
        return {}

    client = RESTClient(api_key)
    ticker = ticker.upper()
    result = {
        "ticker": ticker,
        "source": "polygon.io",
        "fetched_at": datetime.utcnow().isoformat(),
    }

    # 1. Ticker Details (company info)
    try:
        details = client.get_ticker_details(ticker)
        result["ticker_details"] = {
            "name": getattr(details, 'name', None),
            "market_cap": getattr(details, 'market_cap', None),
            "description": getattr(details, 'description', None),
            "sic_description": getattr(details, 'sic_description', None),
            "total_employees": getattr(details, 'total_employees', None),
            "homepage_url": getattr(details, 'homepage_url', None),
            "currency_name": getattr(details, 'currency_name', None),
            "weighted_shares_outstanding": getattr(details, 'weighted_shares_outstanding', None),
        }
    except Exception as e:
        logger.warning(f"Ticker details failed for {ticker}: {e}")
        result["ticker_details"] = {}

    # 2. Previous Close (latest price)
    try:
        agg_list = client.get_previous_close_agg(ticker)
        if agg_list and len(agg_list) > 0:
            bar = agg_list[0]
            result["previous_close"] = {
                "close": getattr(bar, 'close', None),
                "open": getattr(bar, 'open', None),
                "high": getattr(bar, 'high', None),
                "low": getattr(bar, 'low', None),
                "volume": getattr(bar, 'volume', None),
            }
    except Exception as e:
        logger.warning(f"Previous close failed for {ticker}: {e}")
        result["previous_close"] = {}

    # 3. Price History (2 years)
    try:
        from_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        prices = []
        for agg in client.list_aggs(ticker=ticker, multiplier=1, timespan="day",
                                    from_=from_date, to=to_date, limit=1000):
            prices.append({
                "timestamp": pd.to_datetime(getattr(agg, 'timestamp', 0), unit='ms'),
                "close": getattr(agg, 'close', None),
                "open": getattr(agg, 'open', None),
                "high": getattr(agg, 'high', None),
                "low": getattr(agg, 'low', None),
                "volume": getattr(agg, 'volume', None),
            })

        if prices:
            df = pd.DataFrame(prices)
            df.set_index('timestamp', inplace=True)
            result["price_history"] = df
            logger.info(f"Fetched {len(df)} days of price history")
        else:
            result["price_history"] = pd.DataFrame()

    except Exception as e:
        logger.warning(f"Price history failed for {ticker}: {e}")
        result["price_history"] = pd.DataFrame()

    # 4. Financials (income/balance/cash flow statements)
    try:
        financials_list = []
        for fin in client.vx.list_stock_financials(ticker=ticker, limit=8, sort="filing_date", order="desc"):
            fin_dict = {
                "fiscal_year": getattr(fin, 'fiscal_year', None),
                "fiscal_period": getattr(fin, 'fiscal_period', None),
                "end_date": getattr(fin, 'end_date', None),
                "filing_date": getattr(fin, 'filing_date', None),
            }

            # Try to extract financials
            if hasattr(fin, 'financials'):
                findata = fin.financials

                # Income statement
                if hasattr(findata, 'income_statement'):
                    inc = findata.income_statement
                    fin_dict["income_statement"] = {
                        "revenues": _safe_get_value(inc, 'revenues'),
                        "cost_of_revenue": _safe_get_value(inc, 'cost_of_revenue'),
                        "gross_profit": _safe_get_value(inc, 'gross_profit'),
                        "operating_expenses": _safe_get_value(inc, 'operating_expenses'),
                        "operating_income_loss": _safe_get_value(inc, 'operating_income_loss'),
                        "net_income_loss_attributable_to_parent": _safe_get_value(inc, 'net_income_loss_attributable_to_parent'),
                    }

                # Balance sheet
                if hasattr(findata, 'balance_sheet'):
                    bal = findata.balance_sheet
                    fin_dict["balance_sheet"] = {
                        "assets": _safe_get_value(bal, 'assets'),
                        "current_assets": _safe_get_value(bal, 'current_assets'),
                        "liabilities": _safe_get_value(bal, 'liabilities'),
                        "current_liabilities": _safe_get_value(bal, 'current_liabilities'),
                        "equity_attributable_to_parent": _safe_get_value(bal, 'equity_attributable_to_parent'),
                    }

                # Cash flow
                if hasattr(findata, 'cash_flow_statement'):
                    cf = findata.cash_flow_statement
                    fin_dict["cash_flow"] = {
                        "net_cash_flow_from_operating_activities": _safe_get_value(cf, 'net_cash_flow_from_operating_activities'),
                    }

            financials_list.append(fin_dict)

        result["financials"] = financials_list
        if financials_list:
            result["latest_financials"] = financials_list[0]
            logger.info(f"Fetched {len(financials_list)} financial reports")

    except Exception as e:
        logger.warning(f"Financials failed for {ticker}: {e}")
        result["financials"] = []

    return result


def _safe_get_value(obj, attr_name: str) -> Optional[float]:
    """Safely extract value from Polygon financial object"""
    try:
        attr = getattr(obj, attr_name, None)
        if attr and hasattr(attr, 'value'):
            return attr.value
        elif isinstance(attr, dict):
            return attr.get('value')
        return None
    except:
        return None
