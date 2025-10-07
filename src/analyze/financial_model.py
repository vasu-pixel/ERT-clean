"""Advanced financial modeling utilities."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.data_pipeline.models import CompanyDataset
from src.analyze.deterministic import _safe_divide

logger = logging.getLogger(__name__)

CONFIG_CANDIDATES = [
    Path("segment_assumptions.json"),
    Path("config/segment_assumptions.json"),
]


def load_segment_config(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    config: Dict[str, Any] = {}
    for path in CONFIG_CANDIDATES:
        if path.exists():
            try:
                config = json.loads(path.read_text())
            except Exception as exc:
                logger.warning("Unable to read %s: %s", path, exc)
            break
    default_config = config.get("default", {})
    ticker_config = config.get(ticker, {})
    merged = {**default_config, **ticker_config}
    segments = merged.get("segments") or default_config.get("segments") or []
    merged["segments"] = segments
    return merged


@dataclass
class SegmentForecast:
    name: str
    revenue: List[float]
    ebit: List[float]


def build_segment_forecast(dataset: CompanyDataset, years: int, config: Dict[str, Any]) -> List[SegmentForecast]:
    det = dataset.supplemental.get("deterministic", {}) if hasattr(dataset, "supplemental") else {}
    history = det.get("history", {}) if isinstance(det, dict) else {}
    history_rows = history.get("history") if isinstance(history, dict) else None
    ratios = history.get("ratios") if isinstance(history, dict) else None
    latest_revenue = None
    latest_operating_margin = None
    if history_rows:
        latest_row = history_rows[0]
        latest_revenue = latest_row.get("revenue")
        latest_operating_margin = _safe_divide(latest_row.get("operating_income"), latest_row.get("revenue"))
    if latest_revenue is None:
        latest_revenue = dataset.supplemental.get("deterministic", {}).get("forecast", {}).get("revenue", {}).get("ttm")
    if latest_operating_margin is None and ratios:
        latest_operating_margin = ratios[0].get("operating_margin")
    if latest_operating_margin is None:
        latest_operating_margin = _safe_divide(dataset.financials.fundamentals.get("operatingIncome"), latest_revenue)
    if latest_operating_margin is None:
        latest_operating_margin = 0.1
    revenue_growth = dataset.supplemental.get("deterministic", {}).get("forecast", {}).get("revenue_growth", None)
    if revenue_growth is None:
        revenue_growth = dataset.financials.fundamentals.get("revenueGrowth") or 0.03
    segments_cfg = config.get("segments", [])
    if not segments_cfg:
        segments_cfg = [{"name": "Consolidated", "revenue_share": 1.0, "revenue_growth_adjustments": [], "ebit_margin": None}]
    segment_forecasts: List[SegmentForecast] = []
    for segment in segments_cfg:
        share = segment.get("revenue_share", 0)
        if share <= 0:
            continue
        base_revenue = latest_revenue * share if latest_revenue else None
        if not base_revenue:
            continue
        adj_list = segment.get("revenue_growth_adjustments", [])
        ebit_margin = segment.get("ebit_margin")
        if ebit_margin is None:
            ebit_margin = latest_operating_margin
        series_revenue: List[float] = []
        series_ebit: List[float] = []
        current_revenue = base_revenue
        for year in range(years):
            growth_adj = adj_list[year] if year < len(adj_list) else 0
            growth = revenue_growth + growth_adj
            current_revenue = current_revenue * (1 + growth)
            series_revenue.append(current_revenue)
            series_ebit.append(current_revenue * ebit_margin)
        segment_forecasts.append(SegmentForecast(name=segment.get("name", f"Segment {len(segment_forecasts)+1}"), revenue=series_revenue, ebit=series_ebit))
    return segment_forecasts


def build_three_statement_model(dataset: CompanyDataset, fcf_schedule: Dict[str, Any], segment_forecasts: List[SegmentForecast], config: Dict[str, Any]) -> Dict[str, Any]:
    years = len(fcf_schedule.get("schedule", []))
    if years == 0:
        return {}
    assumptions = fcf_schedule.get("assumptions", {})
    history = dataset.supplemental.get("deterministic", {}).get("history", {}) if hasattr(dataset, "supplemental") else {}
    history_rows = history.get("history") if isinstance(history, dict) else None
    latest = history_rows[0] if history_rows else {}
    cash = latest.get("cash") or dataset.financials.fundamentals.get("cash")
    if cash is None:
        cash = dataset.financials.fundamentals.get("totalCash") or 0.0
    debt = latest.get("total_debt") or dataset.financials.fundamentals.get("totalDebt") or 0.0
    equity = latest.get("total_equity") or dataset.financials.fundamentals.get("totalStockholderEquity")
    if equity is None:
        equity = dataset.financials.fundamentals.get("market_cap")
    payout_ratio = config.get("payout_ratio")
    if payout_ratio is None and history_rows:
        dividends = latest.get("dividends")
        if dividends and latest.get("net_income"):
            payout_ratio = _safe_divide(dividends, latest.get("net_income"))
    if payout_ratio is None:
        payout_ratio = 0.25
    revenue_total = np.sum([sf.revenue for sf in segment_forecasts], axis=0) if segment_forecasts else [entry.get("revenue") for entry in fcf_schedule["schedule"]]
    ebit_total = np.sum([sf.ebit for sf in segment_forecasts], axis=0) if segment_forecasts else [entry.get("ebit") for entry in fcf_schedule["schedule"]]
    income_statement: List[Dict[str, Any]] = []
    balance_sheet: List[Dict[str, Any]] = []
    cash_flow: List[Dict[str, Any]] = []
    previous_equity = equity
    previous_cash = cash
    previous_debt = debt
    for idx, entry in enumerate(fcf_schedule["schedule"]):
        revenue = revenue_total[idx]
        ebit = ebit_total[idx]
        tax_rate = assumptions.get("tax_rate", 0.24)
        nopat = ebit * (1 - tax_rate)
        depreciation = entry.get("depreciation")
        capex = entry.get("capex")
        wc_change = entry.get("change_working_capital")
        fcf = entry.get("free_cash_flow")
        dividends = nopat * payout_ratio
        retained_earnings = nopat - dividends
        previous_equity = (previous_equity or 0) + retained_earnings
        previous_cash = (previous_cash or 0) + fcf - dividends
        if previous_cash < 0 and previous_debt is not None:
            previous_debt += abs(previous_cash)
            previous_cash = 0
        income_statement.append({
            "year": entry.get("year"),
            "revenue": revenue,
            "ebit": ebit,
            "net_income": nopat,
            "dividends": dividends,
        })
        balance_sheet.append({
            "year": entry.get("year"),
            "cash": previous_cash,
            "total_debt": previous_debt,
            "total_equity": previous_equity,
        })
        cash_flow.append({
            "year": entry.get("year"),
            "operating_cash_flow": entry.get("nopat") + depreciation - wc_change if entry.get("nopat") is not None else None,
            "capex": capex,
            "free_cash_flow": fcf,
            "dividends": dividends,
        })
    return {
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "assumptions": {
            "payout_ratio": payout_ratio,
            "tax_rate": assumptions.get("tax_rate"),
        },
    }
