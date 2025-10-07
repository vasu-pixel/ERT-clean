"""Valuation model."""
from dataclasses import dataclass

import yfinance as yf


@dataclass
class Valuation:
    """Valuation model."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.stock = yf.Ticker(self.ticker)

    def get_historical_data(self) -> dict:
        """Get historical financial data."""
        return {
            "income_statement": self.stock.income_stmt,
            "balance_sheet": self.stock.balance_sheet,
            "cash_flow": self.stock.cashflow,
        }
