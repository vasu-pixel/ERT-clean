"""Datamodels for normalized company data."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class CompanySnapshot:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: Optional[float]
    current_price: Optional[float]
    currency: Optional[str]
    as_of: datetime
    shares_outstanding: Optional[float] = None
    enterprise_value: Optional[float] = None


@dataclass
class FinancialDataset:
    fundamentals: Dict[str, Optional[float]]
    ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    price_history: Optional[pd.DataFrame] = None
    balance_sheet: Optional[pd.DataFrame] = None
    income_statement: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None


@dataclass
class CompanyDataset:
    snapshot: CompanySnapshot
    financials: FinancialDataset
    metadata: Dict[str, Any] = field(default_factory=dict)
    supplemental: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def ticker(self) -> str:
        return self.snapshot.ticker
