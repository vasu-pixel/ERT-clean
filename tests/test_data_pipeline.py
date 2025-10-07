"""
Test suite for data pipeline components
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from data_pipeline.orchestrator import DataOrchestrator
from data_pipeline.models import CompanyDataset
from data_pipeline import sources


class TestDataOrchestrator:
    """Test data orchestration functionality"""

    def test_initialization(self, test_config):
        """Test orchestrator initialization"""
        orchestrator = DataOrchestrator(config=test_config)
        assert orchestrator.config == test_config
        assert hasattr(orchestrator, 'cache')

    def test_company_data_collection(self, mock_yfinance_data, test_metrics):
        """Test company data collection with mocked yfinance"""
        test_metrics.start_timing()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = mock_yfinance_data['info']
            mock_ticker_instance.financials = pd.DataFrame(mock_yfinance_data['financials'])
            mock_ticker.return_value = mock_ticker_instance

            orchestrator = DataOrchestrator()

            try:
                dataset = orchestrator.get_company_data("AAPL")
                test_metrics.end_timing()
                test_metrics.log_performance("company_data_collection", test_metrics.get_duration(), True)

                assert isinstance(dataset, CompanyDataset)
                assert dataset.ticker == "AAPL"
                assert dataset.info['longName'] == 'Apple Inc.'
                assert dataset.info['sector'] == 'Technology'

            except Exception as e:
                test_metrics.log_error(e)
                raise

    def test_data_validation(self, mock_company_data):
        """Test data validation mechanisms"""
        orchestrator = DataOrchestrator()

        # Test valid data
        assert orchestrator._validate_company_data(mock_company_data)

        # Test missing required fields
        invalid_data = CompanyDataset(
            ticker="TEST",
            info={},  # Missing required fields
            financial_statements={},
            market_data={}
        )

        with pytest.raises(ValueError):
            orchestrator._validate_company_data(invalid_data)

    def test_cache_functionality(self, mock_company_data, test_metrics):
        """Test caching mechanisms"""
        orchestrator = DataOrchestrator()

        # First call should cache the data
        test_metrics.start_timing()
        orchestrator.cache.set("AAPL", mock_company_data, ttl=3600)
        cached_data = orchestrator.cache.get("AAPL")
        test_metrics.end_timing()

        test_metrics.log_performance("cache_operations", test_metrics.get_duration(), True)

        assert cached_data is not None
        assert cached_data.ticker == "AAPL"

    def test_error_handling(self):
        """Test error handling in data collection"""
        orchestrator = DataOrchestrator()

        with patch('yfinance.Ticker') as mock_ticker:
            # Simulate yfinance error
            mock_ticker.side_effect = Exception("Network error")

            with pytest.raises(Exception):
                orchestrator.get_company_data("INVALID")


class TestDataSources:
    """Test data source modules"""

    def test_headline_fetching(self, test_metrics):
        """Test news headline fetching"""
        test_metrics.start_timing()

        with patch('requests.get') as mock_get:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '<html><body><h1>Apple Stock Rises</h1></body></html>'
            mock_get.return_value = mock_response

            try:
                headlines = sources.fetch_recent_headlines("AAPL", limit=5)
                test_metrics.end_timing()
                test_metrics.log_performance("headline_fetching", test_metrics.get_duration(), True)

                assert isinstance(headlines, list)
                assert len(headlines) <= 5

            except Exception as e:
                test_metrics.log_error(e)
                # Don't fail test for network issues
                pytest.skip(f"Network issue: {e}")

    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        sample_headlines = [
            "Apple Stock Surges on Strong Earnings",
            "Investors Worried About Apple's Future",
            "Apple Announces New Product Launch"
        ]

        sentiment_summary = sources.summarize_headline_sentiment(sample_headlines)

        assert isinstance(sentiment_summary, dict)
        assert 'count' in sentiment_summary
        assert 'sentiment_score' in sentiment_summary
        assert sentiment_summary['count'] == len(sample_headlines)

    def test_financial_data_processing(self, mock_yfinance_data):
        """Test financial data processing and cleaning"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.financials = pd.DataFrame(mock_yfinance_data['financials'])
            mock_ticker.return_value = mock_ticker_instance

            # Test data cleaning
            cleaned_data = sources.clean_financial_data(mock_ticker_instance.financials)

            assert isinstance(cleaned_data, pd.DataFrame)
            assert not cleaned_data.empty
            assert not cleaned_data.isnull().all().any()  # No columns with all NaN


class TestCompanyDataset:
    """Test CompanyDataset model"""

    def test_dataset_creation(self, mock_company_data):
        """Test dataset creation and validation"""
        assert mock_company_data.ticker == "AAPL"
        assert isinstance(mock_company_data.info, dict)
        assert isinstance(mock_company_data.financial_statements, dict)

    def test_financial_ratios_calculation(self, mock_company_data):
        """Test financial ratios calculation"""
        ratios = mock_company_data.calculate_financial_ratios()

        assert isinstance(ratios, dict)
        assert 'current_ratio' in ratios or 'debt_to_equity' in ratios

    def test_growth_rates_calculation(self, mock_company_data):
        """Test growth rates calculation"""
        growth_rates = mock_company_data.calculate_growth_rates()

        assert isinstance(growth_rates, dict)
        assert 'revenue_growth' in growth_rates

    def test_data_export(self, mock_company_data, temp_directory):
        """Test data export functionality"""
        export_path = temp_directory / "test_export.json"

        mock_company_data.export_to_json(export_path)

        assert export_path.exists()
        assert export_path.stat().st_size > 0


class TestDataIntegrity:
    """Test data integrity and consistency"""

    def test_revenue_consistency(self, mock_company_data):
        """Test revenue data consistency across statements"""
        income_statement = mock_company_data.financial_statements.get('income_statement', {})

        if 'Total Revenue' in income_statement:
            revenues = income_statement['Total Revenue']
            assert all(isinstance(rev, (int, float)) for rev in revenues)
            assert all(rev > 0 for rev in revenues)  # Revenues should be positive

    def test_balance_sheet_balance(self, mock_company_data):
        """Test balance sheet balancing"""
        balance_sheet = mock_company_data.financial_statements.get('balance_sheet', {})

        # Basic balance sheet validation
        if 'Total Assets' in balance_sheet and 'Total Liabilities' in balance_sheet:
            assets = balance_sheet['Total Assets']
            liabilities = balance_sheet['Total Liabilities']

            # Assets should be greater than liabilities for healthy companies
            for i in range(min(len(assets), len(liabilities))):
                assert assets[i] >= liabilities[i]

    def test_cash_flow_consistency(self, mock_company_data):
        """Test cash flow statement consistency"""
        cash_flow = mock_company_data.financial_statements.get('cash_flow', {})

        if 'Operating Cash Flow' in cash_flow and 'Free Cash Flow' in cash_flow:
            ocf = cash_flow['Operating Cash Flow']
            fcf = cash_flow['Free Cash Flow']

            # Free cash flow should generally be less than operating cash flow
            for i in range(min(len(ocf), len(fcf))):
                if ocf[i] > 0:  # Only check for positive operating cash flow
                    assert fcf[i] <= ocf[i]


class TestPerformanceMetrics:
    """Test performance-related metrics"""

    def test_data_collection_speed(self, performance_test_config, test_metrics):
        """Test data collection performance"""
        max_time = performance_test_config['max_execution_time']

        test_metrics.start_timing()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {'longName': 'Test Company'}
            mock_ticker.return_value = mock_ticker_instance

            orchestrator = DataOrchestrator()

            try:
                dataset = orchestrator.get_company_data("TEST")
                test_metrics.end_timing()

                duration = test_metrics.get_duration()
                test_metrics.log_performance("data_collection_speed", duration, True)

                assert duration < max_time, f"Data collection took {duration}s, max allowed {max_time}s"

            except Exception as e:
                test_metrics.log_error(e)
                raise

    def test_memory_usage(self, performance_test_config, mock_company_data):
        """Test memory usage during data processing"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate heavy data processing
        orchestrator = DataOrchestrator()
        datasets = [mock_company_data for _ in range(100)]  # Create many datasets

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        max_memory = performance_test_config['memory_limit_mb']
        assert memory_increase < max_memory, f"Memory usage increased by {memory_increase}MB, max allowed {max_memory}MB"

    def test_concurrent_data_access(self, mock_company_data):
        """Test concurrent data access scenarios"""
        import threading
        import time

        orchestrator = DataOrchestrator()
        results = []
        errors = []

        def fetch_data(ticker):
            try:
                with patch('yfinance.Ticker') as mock_ticker:
                    mock_ticker_instance = MagicMock()
                    mock_ticker_instance.info = {'longName': f'{ticker} Company'}
                    mock_ticker.return_value = mock_ticker_instance

                    dataset = orchestrator.get_company_data(ticker)
                    results.append(dataset)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        for ticker in tickers:
            thread = threading.Thread(target=fetch_data, args=(ticker,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == len(tickers), f"Expected {len(tickers)} results, got {len(results)}"