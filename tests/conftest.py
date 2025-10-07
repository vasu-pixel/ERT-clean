"""
Test configuration and fixtures for ERT testing suite
"""

import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logging_config import ERTLogger, ert_logger


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "valuation": {
            "risk_free_rate": 0.045,
            "equity_risk_premium": 0.055,
            "wacc_floor": 0.06,
            "wacc_ceiling": 0.14,
            "scenarios": {
                "bear": {"growth_delta": -0.02, "wacc_delta": 0.01},
                "base": {"growth_delta": 0.0, "wacc_delta": 0.0},
                "bull": {"growth_delta": 0.02, "wacc_delta": -0.01}
            },
            "sector_beta_overrides": {
                "Technology": 1.2,
                "Healthcare": 0.9,
                "Energy": 1.4
            }
        },
        "forecast_defaults": {
            "revenue_growth_years": 5,
            "terminal_growth_rate": 0.025,
            "forecast_periods": 10
        }
    }


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data for testing"""
    mock_info = {
        'longName': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'marketCap': 3000000000000,
        'beta': 1.2,
        'trailingPE': 25.5,
        'forwardPE': 22.0,
        'priceToBook': 8.5,
        'debtToEquity': 1.73,
        'returnOnEquity': 0.147,
        'currentRatio': 1.04,
        'totalRevenue': 394328000000,
        'totalDebt': 122797000000,
        'totalCash': 29965000000,
        'freeCashflow': 99584000000,
        'operatingCashflow': 122151000000
    }

    mock_financials = {
        'Total Revenue': [394328000000, 365817000000, 274515000000],
        'Operating Income': [119437000000, 108949000000, 83344000000],
        'Net Income': [99803000000, 94680000000, 64687000000]
    }

    return {
        'info': mock_info,
        'financials': mock_financials
    }


@pytest.fixture
def mock_company_data():
    """Mock company dataset for testing"""
    from data_pipeline.models import CompanyDataset

    return CompanyDataset(
        ticker="AAPL",
        info={
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 3000000000000,
            'beta': 1.2
        },
        financial_statements={
            'income_statement': {
                'Total Revenue': [394328000000, 365817000000, 274515000000],
                'Net Income': [99803000000, 94680000000, 64687000000]
            },
            'balance_sheet': {
                'Total Debt': [122797000000, 108047000000, 91807000000],
                'Total Cash': [29965000000, 35929000000, 38016000000]
            },
            'cash_flow': {
                'Operating Cash Flow': [122151000000, 104038000000, 80674000000],
                'Free Cash Flow': [99584000000, 92953000000, 73365000000]
            }
        },
        market_data={
            'current_price': 175.50,
            'shares_outstanding': 15728231616
        }
    )


@pytest.fixture
def test_logger():
    """Test logger instance with temporary directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_logger_instance = ERTLogger(log_dir=temp_dir)
        yield test_logger_instance


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment with mocked external dependencies"""
    with patch('yfinance.Ticker') as mock_ticker, \
         patch('requests.get') as mock_requests, \
         patch('time.sleep'):  # Skip sleep calls in tests

        # Mock successful yfinance responses
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance

        # Mock successful requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_requests.return_value = mock_response

        yield {
            'mock_ticker': mock_ticker,
            'mock_requests': mock_requests
        }


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    import pandas as pd

    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    prices = 150 + (dates.dayofyear / 365) * 25  # Trending upward

    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': 50000000 + (dates.dayofyear % 10) * 1000000
    }).set_index('Date')


@pytest.fixture
def performance_test_config():
    """Configuration for performance testing"""
    return {
        'max_execution_time': 30.0,  # seconds
        'memory_limit_mb': 500,
        'acceptable_cache_hit_rate': 0.8
    }


class TestMetrics:
    """Helper class for collecting test metrics"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.errors = []
        self.performance_logs = []

    def start_timing(self):
        self.start_time = datetime.now(timezone.utc)

    def end_timing(self):
        self.end_time = datetime.now(timezone.utc)

    def get_duration(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def log_error(self, error):
        self.errors.append({
            'timestamp': datetime.now(timezone.utc),
            'error': str(error),
            'type': type(error).__name__
        })

    def log_performance(self, operation, duration, success):
        self.performance_logs.append({
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now(timezone.utc)
        })


@pytest.fixture
def test_metrics():
    """Test metrics collection fixture"""
    return TestMetrics()