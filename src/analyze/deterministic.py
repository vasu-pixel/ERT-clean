"""Deterministic analytics layer for valuation, forecasting, and risk."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.data_pipeline.models import CompanyDataset

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    revenue: Dict[str, Optional[float]]
    eps: Dict[str, Optional[float]]


@dataclass
class ValuationResult:
    dcf_value: Optional[float]
    multiples_summary: Dict[str, Optional[float]]
    assumptions: Dict[str, float]
    scenarios: Dict[str, Optional[float]]
    scenario_parameters: Dict[str, Dict[str, float]]
    sensitivity: Dict[str, Any]


@dataclass
class RiskResult:
    liquidity: Dict[str, Optional[float]]
    leverage: Dict[str, Optional[float]]
    market: Dict[str, Optional[float]]
    notes: Dict[str, str]


@dataclass
class DeterministicOutputs:
    forecast: ForecastResult
    valuation: ValuationResult
    risk: RiskResult
    trends: Dict[str, Optional[float]]
    peer_metrics: List[Dict[str, Optional[float]]]
    history: Dict[str, Any]
    fcf_projection: Dict[str, Any]


def _safe_float(value: Optional[float], fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, floor: float, ceiling: float) -> float:
    return max(floor, min(ceiling, value))

def _get_risk_free_rate(config: Dict[str, Any]) -> float:
    fallback = float(config.get("risk_free_rate", 0.045))
    proxy_symbol = config.get("risk_free_proxy", "^TNX")
    try:
        proxy = yf.Ticker(proxy_symbol)
        history = proxy.history(period="5d", interval="1d")
        if not history.empty and "Close" in history.columns:
            latest = float(history["Close"].dropna().iloc[-1])
            if latest > 0:
                # Treasury indices like ^TNX are quoted in percent
                rate = latest / (100.0 if latest > 1 else 1.0)
                return rate
    except Exception as exc:
        logger.debug("Risk-free rate fetch failed (%s): %s", proxy_symbol, exc)
    return fallback


def _estimate_beta(ticker: str, config: Dict[str, Any], fallback: float) -> float:
    market_symbol = config.get("market_index", "^GSPC")
    lookback = config.get("beta_lookback", "2y")
    interval = config.get("beta_interval", "1d")
    try:
        asset_hist = yf.Ticker(ticker).history(period=lookback, interval=interval, auto_adjust=True)
        market_hist = yf.Ticker(market_symbol).history(period=lookback, interval=interval, auto_adjust=True)

        if asset_hist.empty or market_hist.empty:
            raise ValueError("insufficient history")

        df = pd.DataFrame({
            "asset": asset_hist["Close"],
            "market": market_hist["Close"],
        }).dropna()
        returns = df.pct_change().dropna()
        if returns.empty or len(returns) < 30:
            raise ValueError("not enough returns to compute beta")

        cov_matrix = np.cov(returns["asset"], returns["market"])
        market_var = returns["market"].var()
        if market_var <= 0:
            raise ValueError("non-positive market variance")

        beta = cov_matrix[0][1] / market_var
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("invalid beta result")
        return float(beta)
    except Exception as exc:
        logger.debug("Beta estimation failed for %s: %s", ticker, exc)
        return fallback


def _extract_statement_value(statement: Optional[pd.DataFrame], candidate_labels: List[str]) -> Optional[float]:
    if statement is None or statement.empty:
        return None
    for label in candidate_labels:
        if label in statement.index:
            series = statement.loc[label]
            if isinstance(series, pd.Series):
                values = series.dropna().astype(float)
                if not values.empty:
                    return float(values.iloc[0])
            elif isinstance(series, (int, float)):
                return float(series)
    return None


def _compute_wacc(dataset: CompanyDataset, statements: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, float]:
    fundamentals = dataset.financials.fundamentals
    valuation_cfg = config or {}

    risk_free = _get_risk_free_rate(valuation_cfg)
    equity_risk_premium = float(valuation_cfg.get("equity_risk_premium", 0.055))
    default_beta = float(valuation_cfg.get("default_beta", 1.0))
    sector_overrides = valuation_cfg.get("sector_beta_overrides", {}) or {}

    fallback_beta = float(sector_overrides.get(dataset.snapshot.sector, default_beta))
    raw_beta = fundamentals.get("beta")
    try:
        provided_beta = float(raw_beta) if raw_beta is not None else None
        if provided_beta and provided_beta > 0:
            fallback_beta = provided_beta
    except (TypeError, ValueError):
        pass

    beta = _estimate_beta(dataset.snapshot.ticker, valuation_cfg, fallback_beta)
    cost_of_equity = risk_free + beta * equity_risk_premium

    income_statement = statements.get("income_statement") if statements else None
    balance_sheet = statements.get("balance_sheet") if statements else None

    interest_expense_stmt = _extract_statement_value(
        income_statement,
        ["Interest Expense", "Interest Expense Non Operating", "Interest Expense"]
    )
    interest_expense = abs(_safe_float(interest_expense_stmt, abs(_safe_float(fundamentals.get("interestExpense")))))

    reported_total_debt = _safe_float(fundamentals.get("totalDebt"))
    total_debt_stmt = _extract_statement_value(
        balance_sheet,
        ["Total Debt", "Long Term Debt", "Short Long Term Debt"]
    )
    total_debt = abs(_safe_float(total_debt_stmt, reported_total_debt))

    base_cost_of_debt = float(valuation_cfg.get("base_cost_of_debt", risk_free + 0.02))
    cost_of_debt = base_cost_of_debt
    if total_debt > 0 and interest_expense:
        derived_cost = interest_expense / total_debt
        if derived_cost > 0:
            cost_of_debt = derived_cost

    interest_coverage = fundamentals.get("interestCoverage")
    try:
        coverage = float(interest_coverage)
        if coverage < 2:
            cost_of_debt += 0.02
        elif coverage > 6:
            cost_of_debt = max(cost_of_debt - 0.01, 0.01)
    except (TypeError, ValueError):
        pass

    tax_rate = float(valuation_cfg.get("tax_rate", _safe_float(fundamentals.get("effectiveTaxRate"), 0.25)))

    equity_value = _safe_float(fundamentals.get("marketCap"), dataset.snapshot.market_cap)
    total_capital = equity_value + total_debt

    if total_capital <= 0:
        equity_weight = 0.8
        debt_weight = 0.2
    else:
        equity_weight = equity_value / total_capital
        debt_weight = total_debt / total_capital

    wacc = cost_of_equity * equity_weight + cost_of_debt * (1 - tax_rate) * debt_weight
    wacc_floor = float(valuation_cfg.get("wacc_floor", risk_free + 0.01))
    wacc_ceiling = float(valuation_cfg.get("wacc_ceiling", risk_free + 0.10))
    wacc = _clamp(wacc, wacc_floor, wacc_ceiling)

    return {
        "wacc": wacc,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "beta": beta,
        "risk_free_rate": risk_free,
        "equity_risk_premium": equity_risk_premium,
        "tax_rate": tax_rate,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "total_debt": total_debt,
        "interest_expense": interest_expense,
    }


def build_forecast(dataset: CompanyDataset, config: Optional[Dict] = None) -> ForecastResult:
    fundamentals = dataset.financials.fundamentals
    config = config or {}

    revenue_growth = _safe_float(fundamentals.get("revenueGrowth"), config.get("default_revenue_growth", 0.05))
    earnings_growth = _safe_float(fundamentals.get("earningsGrowth"), config.get("default_eps_growth", 0.05))

    statements = dataset.raw_payload.get("statements", {}) if isinstance(dataset.raw_payload, dict) else {}
    revenue_ttm = _safe_float(fundamentals.get("totalRevenue"), _compute_revenue_ttm(statements))
    eps_ttm = _safe_float(fundamentals.get("trailingEps"), _compute_eps_ttm(fundamentals, statements))

    if revenue_ttm <= 0:
        raise ValueError("Revenue TTM unavailable; cannot build forecast")
    if eps_ttm == 0:
        raise ValueError("EPS TTM unavailable; cannot build forecast")

    forecasts_revenue = {
        "ttm": revenue_ttm,
        "next_year": revenue_ttm * (1 + revenue_growth),
        "year_two": revenue_ttm * (1 + revenue_growth) ** 2,
    }

    forecasts_eps = {
        "ttm": eps_ttm,
        "next_year": eps_ttm * (1 + earnings_growth),
        "year_two": eps_ttm * (1 + earnings_growth) ** 2,
    }

    return ForecastResult(revenue=forecasts_revenue, eps=forecasts_eps)


def run_dcf(dataset: CompanyDataset, config: Optional[Dict] = None) -> ValuationResult:
    fundamentals = dataset.financials.fundamentals
    valuation_cfg = config or {}

    statements = dataset.raw_payload.get("statements", {}) if isinstance(dataset.raw_payload, dict) else {}

    free_cash_flow = _safe_float(fundamentals.get("freeCashflow"), _compute_free_cash_flow(statements))
    shares_outstanding = _safe_float(fundamentals.get("sharesOutstanding"), 0.0)
    wacc_inputs = _compute_wacc(dataset, statements, valuation_cfg)
    wacc = wacc_inputs["wacc"]
    terminal_growth = float(valuation_cfg.get("terminal_growth", 0.025))
    if terminal_growth >= wacc:
        terminal_growth = max(wacc - 0.005, 0.0)
    projection_growth = _safe_float(fundamentals.get("revenueGrowth"), valuation_cfg.get("projection_growth", 0.04))
    horizon_years = int(valuation_cfg.get("horizon_years", 5))

    def _run_single_dcf(growth: float, scenario_wacc: float) -> Optional[float]:
        if free_cash_flow <= 0 or shares_outstanding <= 0 or scenario_wacc <= terminal_growth:
            return None
        cash_flows = []
        current_fcf = free_cash_flow
        for _ in range(horizon_years):
            current_fcf *= (1 + growth)
            cash_flows.append(current_fcf)

        discounted = [cf / ((1 + scenario_wacc) ** (i + 1)) for i, cf in enumerate(cash_flows)]
        terminal_value = cash_flows[-1] * (1 + terminal_growth) / (scenario_wacc - terminal_growth)
        terminal_discounted = terminal_value / ((1 + scenario_wacc) ** horizon_years)
        equity_value = sum(discounted) + terminal_discounted
        return equity_value / shares_outstanding

    intrinsic_value = _run_single_dcf(projection_growth, wacc)

    default_scenarios = {
        "bear": {"growth_delta": -0.03, "wacc_delta": 0.015},
        "base": {"growth_delta": 0.0, "wacc_delta": 0.0},
        "bull": {"growth_delta": 0.03, "wacc_delta": -0.015},
    }
    user_scenarios = valuation_cfg.get("scenarios", {}) or {}

    scenarios: Dict[str, Optional[float]] = {}
    scenario_parameters: Dict[str, Dict[str, float]] = {}
    for name, defaults in default_scenarios.items():
        override = user_scenarios.get(name, {})
        growth_delta = float(override.get("growth_delta", defaults["growth_delta"]))
        wacc_delta = float(override.get("wacc_delta", defaults["wacc_delta"]))
        growth = projection_growth + growth_delta
        floor = float(valuation_cfg.get("wacc_floor", wacc_inputs.get("risk_free_rate", 0.0)))
        ceiling = float(valuation_cfg.get("wacc_ceiling", wacc + 0.05))
        scenario_wacc = _clamp(wacc + wacc_delta, floor, ceiling)
        scenarios[name] = _run_single_dcf(growth, scenario_wacc)
        scenario_parameters[name] = {
            "growth": growth,
            "growth_delta": growth_delta,
            "wacc": scenario_wacc,
            "wacc_delta": wacc_delta,
        }

    multiples_summary = {
        "trailing_pe": fundamentals.get("trailingPE"),
        "forward_pe": fundamentals.get("forwardPE"),
        "price_to_sales": fundamentals.get("priceToSales"),
        "ev_to_ebitda": fundamentals.get("enterpriseToEbitda"),
        "price_to_book": fundamentals.get("priceToBook"),
    }

    assumptions = {
        "wacc": wacc,
        "risk_free_rate": wacc_inputs.get("risk_free_rate", 0.0),
        "equity_risk_premium": wacc_inputs.get("equity_risk_premium", 0.0),
        "beta": wacc_inputs.get("beta", 0.0),
        "cost_of_equity": wacc_inputs.get("cost_of_equity", 0.0),
        "cost_of_debt": wacc_inputs.get("cost_of_debt", 0.0),
        "tax_rate": wacc_inputs.get("tax_rate", 0.0),
        "equity_weight": wacc_inputs.get("equity_weight", 0.0),
        "debt_weight": wacc_inputs.get("debt_weight", 0.0),
        "total_debt": wacc_inputs.get("total_debt", 0.0),
        "interest_expense": wacc_inputs.get("interest_expense", 0.0),
        "terminal_growth": terminal_growth,
        "projection_growth": projection_growth,
        "free_cash_flow": free_cash_flow,
        "shares_outstanding": shares_outstanding,
    }

    sensitivity = _build_dcf_sensitivity(free_cash_flow, shares_outstanding, projection_growth, terminal_growth, wacc, horizon_years)

    return ValuationResult(
        dcf_value=intrinsic_value,
        multiples_summary=multiples_summary,
        assumptions=assumptions,
        scenarios=scenarios,
        scenario_parameters=scenario_parameters,
        sensitivity=sensitivity,
    )


def _build_dcf_sensitivity(fcf: float, shares_outstanding: float, growth: float, terminal_growth: float,
                           base_wacc: float, horizon_years: int) -> Dict[str, Any]:
    if fcf <= 0 or shares_outstanding <= 0:
        return {}

    wacc_values = [base_wacc - 0.01, base_wacc, base_wacc + 0.01]
    tg_values = [terminal_growth - 0.01, terminal_growth, terminal_growth + 0.01]

    def dcf_value(wacc: float, tg: float) -> Optional[float]:
        if wacc <= tg or wacc <= 0:
            return None
        cash_flows = []
        current_fcf = fcf
        for _ in range(horizon_years):
            current_fcf *= (1 + growth)
            cash_flows.append(current_fcf)
        discounted = [cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(cash_flows)]
        terminal_value = cash_flows[-1] * (1 + tg) / (wacc - tg)
        terminal_discounted = terminal_value / ((1 + wacc) ** horizon_years)
        equity_value = sum(discounted) + terminal_discounted
        return equity_value / shares_outstanding

    matrix: List[List[Optional[float]]] = []
    for tg in tg_values:
        row = []
        for wacc in wacc_values:
            row.append(dcf_value(wacc, tg))
        matrix.append(row)

    return {
        "wacc_values": wacc_values,
        "terminal_growth_values": tg_values,
        "dcf_matrix": matrix,
    }


def compute_risk_metrics(dataset: CompanyDataset) -> RiskResult:
    fundamentals = dataset.financials.fundamentals

    liquidity = {
        "current_ratio": fundamentals.get("currentRatio"),
        "quick_ratio": fundamentals.get("quickRatio"),
        "cash_per_share": fundamentals.get("totalCashPerShare"),
    }

    ebitda = _safe_float(fundamentals.get("ebitda"))
    interest_expense = abs(_safe_float(fundamentals.get("interestExpense")))
    interest_coverage = None
    if ebitda and interest_expense:
        interest_coverage = ebitda / interest_expense if interest_expense else None

    leverage = {
        "debt_to_equity": fundamentals.get("debtToEquity"),
        "net_debt": fundamentals.get("netDebt"),
        "interest_coverage": interest_coverage,
    }

    market = {
        "beta": fundamentals.get("beta"),
        "dividend_yield": fundamentals.get("dividendYield"),
        "short_ratio": fundamentals.get("shortRatio"),
    }

    notes = {}
    current_ratio = _safe_float(liquidity.get("current_ratio"))
    if current_ratio and current_ratio < 1:
        notes["liquidity"] = "Current ratio below 1 indicates potential short-term liquidity pressure."

    debt_to_equity = _safe_float(leverage.get("debt_to_equity"))
    if debt_to_equity and debt_to_equity > 200:
        notes["leverage"] = "Debt-to-equity above 200% suggests elevated leverage risk."

    if interest_coverage and interest_coverage < 2:
        notes["interest"] = "Interest coverage below 2x indicates potential debt servicing risk."

    beta = _safe_float(market.get("beta"))
    if beta and beta > 1.3:
        notes["market"] = "Beta above 1.3 implies high sensitivity to market movements."

    return RiskResult(liquidity=liquidity, leverage=leverage, market=market, notes=notes)


def compute_price_trends(dataset: CompanyDataset) -> Dict[str, Optional[float]]:
    history = dataset.financials.price_history
    if history is None or getattr(history, "empty", True):
        return {}

    close_series = None
    for column in ["Close", "close", "Adj Close", "adjclose"]:
        if column in history:
            close_series = history[column]
            break
    if close_series is None:
        return {}

    close_series = close_series.dropna()
    if close_series.empty:
        return {}

    def pct_change(period: int) -> Optional[float]:
        if len(close_series) <= period:
            return None
        try:
            return float(close_series.iloc[-1] / close_series.iloc[-period - 1] - 1)
        except (ZeroDivisionError, IndexError):
            return None

    daily_returns = close_series.pct_change().dropna()
    volatility = float(daily_returns.std() * (252 ** 0.5)) if not daily_returns.empty else None

    trends = {
        "return_1m": pct_change(21),
        "return_3m": pct_change(63),
        "return_6m": pct_change(126),
        "return_1y": pct_change(252),
        "annualized_volatility": volatility,
    }

    if "Volume" in history:
        volume = history["Volume"].dropna()
        if not volume.empty:
            trends["average_volume_3m"] = float(volume.tail(63).mean())

    return trends


def compute_historical_trends(dataset: CompanyDataset) -> Dict[str, Any]:
    statements = dataset.raw_payload.get("statements") if isinstance(dataset.raw_payload, dict) else {}
    if not isinstance(statements, dict):
        return {}

    income_annual = statements.get("income_statement")
    cash_annual = statements.get("cash_flow")
    balance_annual = statements.get("balance_sheet")

    if not isinstance(income_annual, pd.DataFrame) or income_annual.empty:
        return {}

    def extract_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
        if not isinstance(df, pd.DataFrame):
            return pd.Series(dtype=float)
        for candidate in candidates:
            if candidate in df.index:
                series = df.loc[candidate].dropna()
                if not series.empty:
                    return series
        return pd.Series(dtype=float)

    revenue_series = extract_series(income_annual, ["Total Revenue", "Revenue", "Total Revenues"])
    gross_profit_series = extract_series(income_annual, ["Gross Profit", "Gross Profit Services", "Gross Profit Healthcare"])
    operating_income_series = extract_series(income_annual, ["Operating Income", "Operating Income or Loss", "Income Before Tax"])
    net_income_series = extract_series(income_annual, [
        "Net Income", "Net Income Applicable To Common Shares", "Net Income Common Stockholders",
        "Net Income Attributable to Common Stockholders"
    ])
    eps_series = extract_series(income_annual, ["Diluted EPS", "EPS (Diluted)"])
    cfo_series = extract_series(cash_annual, [
        "Net Cash Provided By Operating Activities", "Net Cash Provided by Operating Activities"
    ])
    capex_series = extract_series(cash_annual, ["Capital Expenditures", "Purchase Of Property Plant And Equipment"])
    total_assets_series = extract_series(balance_annual, ["Total Assets"])
    total_equity_series = extract_series(balance_annual, [
        "Total Stockholder Equity", "Total Equity", "Total Shareholder Equity"
    ])
    total_debt_series = extract_series(balance_annual, ["Total Debt", "Long Term Debt", "Long-term Debt"])
    current_assets_series = extract_series(balance_annual, ["Total Current Assets"])
    current_liabilities_series = extract_series(balance_annual, ["Total Current Liabilities"])

    periods = sorted(set(revenue_series.index) | set(cfo_series.index), reverse=True)[:5]

    history_rows: List[Dict[str, Any]] = []
    ratio_rows: List[Dict[str, Any]] = []
    previous_revenue: Optional[float] = None

    for period in periods:
        period_key = str(period)[:10]

        revenue = _get_from_series(revenue_series, period)
        gross_profit = _get_from_series(gross_profit_series, period)
        operating_income = _get_from_series(operating_income_series, period)
        net_income = _get_from_series(net_income_series, period)
        eps = _get_from_series(eps_series, period)
        cfo = _get_from_series(cfo_series, period)
        capex = _get_from_series(capex_series, period)
        fcf = cfo - capex if (cfo is not None and capex is not None) else None
        assets = _get_from_series(total_assets_series, period)
        equity = _get_from_series(total_equity_series, period)
        debt = _get_from_series(total_debt_series, period)
        current_assets = _get_from_series(current_assets_series, period)
        current_liabilities = _get_from_series(current_liabilities_series, period)
        working_capital = None
        if current_assets is not None and current_liabilities is not None:
            working_capital = current_assets - current_liabilities

        history_rows.append({
            "period": period_key,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "net_income": net_income,
            "diluted_eps": eps,
            "operating_cash_flow": cfo,
            "capex": capex,
            "free_cash_flow": fcf,
            "total_assets": assets,
            "total_equity": equity,
            "total_debt": debt,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "working_capital": working_capital,
        })

        gross_margin = _safe_divide(gross_profit, revenue)
        operating_margin = _safe_divide(operating_income, revenue)
        net_margin = _safe_divide(net_income, revenue)
        revenue_growth = _safe_divide((revenue - previous_revenue) if (revenue is not None and previous_revenue) else None, previous_revenue) if previous_revenue else None
        roe = _safe_divide(net_income, equity)
        invested_capital = (assets - equity) if (assets is not None and equity is not None) else None
        roic = _safe_divide(operating_income, invested_capital)

        ratio_rows.append({
            "period": period_key,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "net_margin": net_margin,
            "revenue_growth": revenue_growth,
            "return_on_equity": roe,
            "roic": roic,
        })

        previous_revenue = revenue if revenue is not None else previous_revenue

    return {
        "history": history_rows,
        "ratios": ratio_rows,
    }


def build_fcf_schedule(dataset: CompanyDataset, history: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
    config = config or {}
    statements = dataset.raw_payload.get("statements") if isinstance(dataset.raw_payload, dict) else {}
    if not history:
        return {}

    history_rows = history.get("history", [])
    ratio_rows = history.get("ratios", [])
    if not history_rows:
        return {}

    latest = history_rows[0]
    revenue_base = latest.get("revenue")
    if not revenue_base:
        return {}

    revenue_growth = config.get("detailed_growth")
    if revenue_growth is None and ratio_rows:
        revenue_growth = ratio_rows[0].get("revenue_growth")
    if revenue_growth is None:
        revenue_growth = 0.03

    ebit_margin = ratio_rows[0].get("operating_margin") if ratio_rows else None
    if ebit_margin is None:
        ebit_margin = _safe_divide(latest.get("operating_income"), latest.get("revenue"))
    if ebit_margin is None:
        ebit_margin = 0.1

    tax_rate = _compute_effective_tax_rate(statements)
    if tax_rate is None:
        tax_rate = config.get("tax_rate", 0.24)

    depreciation_pct = config.get("depreciation_pct")
    if depreciation_pct is None:
        depreciation_pct = _compute_depreciation_ratio(statements, revenue_base)
    if depreciation_pct is None:
        depreciation_pct = 0.025

    capex_pct = config.get("capex_pct")
    if capex_pct is None:
        capex = latest.get("capex")
        capex_pct = _safe_divide(capex, revenue_base)
    if capex_pct is None:
        capex_pct = 0.03

    working_capital_ratio = config.get("working_capital_ratio")
    if working_capital_ratio is None:
        working_capital_ratio = _compute_working_capital_ratio(history_rows)
    if working_capital_ratio is None:
        working_capital_ratio = 0.02

    projection_years = config.get("fcf_years", 5)
    previous_revenue = revenue_base
    schedule: List[Dict[str, Any]] = []

    for year in range(1, projection_years + 1):
        revenue = previous_revenue * (1 + revenue_growth)
        ebit = revenue * ebit_margin
        nopat = ebit * (1 - tax_rate)
        depreciation = revenue * depreciation_pct
        capex = revenue * capex_pct
        wc_change = working_capital_ratio * (revenue - previous_revenue)
        free_cash_flow = nopat + depreciation - capex - wc_change

        schedule.append({
            "year": year,
            "revenue": revenue,
            "ebit": ebit,
            "nopat": nopat,
            "depreciation": depreciation,
            "capex": capex,
            "change_working_capital": wc_change,
            "free_cash_flow": free_cash_flow,
        })

        previous_revenue = revenue

    return {
        "assumptions": {
            "revenue_growth": revenue_growth,
            "ebit_margin": ebit_margin,
            "tax_rate": tax_rate,
            "depreciation_pct": depreciation_pct,
            "capex_pct": capex_pct,
            "working_capital_ratio": working_capital_ratio,
            "base_revenue": revenue_base,
        },
        "schedule": schedule,
    }


def _compute_revenue_ttm(statements: Dict[str, Any]) -> Optional[float]:
    for key in ("quarterly_financials", "income_statement"):
        df = statements.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "Total Revenue" in df.index:
                series = df.loc["Total Revenue"].dropna()
            elif "Revenue" in df.index:
                series = df.loc["Revenue"].dropna()
            else:
                continue

            if series.empty:
                continue
            if key.startswith("quarterly"):
                return float(series.head(4).sum())
            return float(series.iloc[0])
    return None


def _compute_eps_ttm(fundamentals: Dict[str, Any], statements: Dict[str, Any]) -> Optional[float]:
    eps = fundamentals.get("trailingEps")
    if eps and eps not in (0, "N/A"):
        return float(eps)

    shares = _safe_float(fundamentals.get("sharesOutstanding"))
    if shares <= 0:
        return None

    for key in ("quarterly_financials", "income_statement"):
        df = statements.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            line = None
            for candidate in ("Net Income", "Net Income Applicable To Common Shares", "Net Income Common Stockholders", "Net Income Attributable to Common Stockholders"):
                if candidate in df.index:
                    line = candidate
                    break
            if not line:
                continue
            series = df.loc[line].dropna()
            if series.empty:
                continue
            net_income = float(series.head(4).sum()) if key.startswith("quarterly") else float(series.iloc[0])
            if net_income:
                return net_income / shares
    return None


def _compute_free_cash_flow(statements: Dict[str, Any]) -> Optional[float]:
    cashflow = statements.get("cash_flow")
    if isinstance(cashflow, pd.DataFrame) and not cashflow.empty:
        for candidate in ("Free Cash Flow", "Free Cash Flow (Operating Activities - Capital Expenditures)", "Net Cash Provided By Operating Activities"):
            if candidate in cashflow.index:
                values = cashflow.loc[candidate].dropna()
                if not values.empty:
                    return float(values.iloc[0])
    return None


def _get_from_series(series: pd.Series, key) -> Optional[float]:
    if series is None or not isinstance(series, pd.Series):
        return None
    if key in series.index:
        value = series.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    try:
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _compute_effective_tax_rate(statements: Dict[str, Any]) -> Optional[float]:
    income = statements.get("income_statement")
    if not isinstance(income, pd.DataFrame) or income.empty:
        return None

    tax_series = None
    for candidate in ("Income Tax Expense", "Provision for Income Taxes", "Income Taxes", "Income Tax Provision"):
        if candidate in income.index:
            tax_series = income.loc[candidate].dropna()
            break
    pretax_series = None
    for candidate in ("Income Before Tax", "Earnings Before Tax", "Ebt"):  # fallback
        if candidate in income.index:
            pretax_series = income.loc[candidate].dropna()
            break

    if tax_series is None or pretax_series is None or tax_series.empty or pretax_series.empty:
        return None

    latest_tax = float(tax_series.iloc[0])
    latest_pretax = float(pretax_series.iloc[0])
    return _safe_divide(latest_tax, latest_pretax)


def _compute_depreciation_ratio(statements: Dict[str, Any], revenue_base: Optional[float]) -> Optional[float]:
    if not revenue_base:
        return None
    cashflow = statements.get("cash_flow")
    if isinstance(cashflow, pd.DataFrame) and not cashflow.empty:
        for candidate in ("Depreciation", "Depreciation And Amortization", "Depreciation & Amortization"):
            if candidate in cashflow.index:
                series = cashflow.loc[candidate].dropna()
                if not series.empty:
                    return _safe_divide(float(series.iloc[0]), revenue_base)
    income = statements.get("income_statement")
    if isinstance(income, pd.DataFrame) and not income.empty:
        for candidate in ("Depreciation", "Depreciation And Amortization"):
            if candidate in income.index:
                series = income.loc[candidate].dropna()
                if not series.empty:
                    return _safe_divide(float(series.iloc[0]), revenue_base)
    return None


def _compute_working_capital_ratio(history_rows: List[Dict[str, Any]]) -> Optional[float]:
    if not history_rows:
        return None
    latest = history_rows[0]
    revenue = latest.get("revenue")
    working_capital = latest.get("working_capital")
    if working_capital is None:
        # attempt to derive from components if available
        current_assets = latest.get("current_assets")
        current_liabilities = latest.get("current_liabilities")
        if current_assets is not None and current_liabilities is not None:
            working_capital = current_assets - current_liabilities
    return _safe_divide(working_capital, revenue)


def run_deterministic_models(dataset: CompanyDataset, config: Optional[Dict] = None) -> DeterministicOutputs:
    config = config or {}

    forecast = build_forecast(dataset, config.get("forecast"))
    valuation = run_dcf(dataset, config.get("valuation"))
    risk = compute_risk_metrics(dataset)
    trends = compute_price_trends(dataset)
    peer_metrics = dataset.supplemental.get("peer_metrics", []) if hasattr(dataset, "supplemental") else []
    history = compute_historical_trends(dataset)
    fcf_projection = build_fcf_schedule(dataset, history, config.get("forecast"))

    return DeterministicOutputs(
        forecast=forecast,
        valuation=valuation,
        risk=risk,
        trends=trends,
        peer_metrics=peer_metrics,
        history=history,
        fcf_projection=fcf_projection,
    )
