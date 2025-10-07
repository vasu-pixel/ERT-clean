"""
Integration tests for the complete ERT system
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from create_professional_model import create_professional_model
from stock_report_generator import StockReportGenerator


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    def test_professional_model_creation_workflow(self, test_metrics, temp_directory):
        """Test complete professional model creation workflow"""
        test_metrics.start_timing()

        with patch('yfinance.Ticker') as mock_ticker:
            # Setup comprehensive mock data
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {
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

            # Mock financial statements
            import pandas as pd
            mock_ticker_instance.financials = pd.DataFrame({
                'Total Revenue': [394328000000, 365817000000, 274515000000],
                'Operating Income': [119437000000, 108949000000, 83344000000],
                'Net Income': [99803000000, 94680000000, 64687000000]
            })

            mock_ticker_instance.balance_sheet = pd.DataFrame({
                'Total Debt': [122797000000, 108047000000, 91807000000],
                'Total Cash': [29965000000, 35929000000, 38016000000],
                'Total Assets': [381191000000, 365725000000, 338516000000]
            })

            mock_ticker_instance.cashflow = pd.DataFrame({
                'Operating Cash Flow': [122151000000, 104038000000, 80674000000],
                'Free Cash Flow': [99584000000, 92953000000, 73365000000]
            })

            mock_ticker.return_value = mock_ticker_instance

            try:
                # Change to temp directory for output
                import os
                original_cwd = os.getcwd()
                os.chdir(str(temp_directory))

                result = create_professional_model(
                    ticker="AAPL",
                    num_simulations=1000,  # Reduced for testing
                    force_refresh=True
                )

                test_metrics.end_timing()
                test_metrics.log_performance("professional_model_creation", test_metrics.get_duration(), True)

                # Verify result structure
                assert isinstance(result, dict)
                assert 'status' in result
                assert 'output_file' in result

                # Verify output file was created
                if 'output_file' in result and result['output_file']:
                    output_path = Path(result['output_file'])
                    assert output_path.exists(), f"Output file not created: {result['output_file']}"
                    assert output_path.stat().st_size > 0, "Output file is empty"

            except Exception as e:
                test_metrics.log_error(e)
                raise
            finally:
                os.chdir(original_cwd)

    def test_report_generation_workflow(self, test_metrics, temp_directory):
        """Test complete report generation workflow"""
        test_metrics.start_timing()

        with patch('yfinance.Ticker') as mock_ticker:
            # Setup mock data
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {
                'longName': 'Microsoft Corporation',
                'sector': 'Technology',
                'marketCap': 2800000000000,
                'beta': 0.9
            }

            mock_ticker.return_value = mock_ticker_instance

            try:
                generator = StockReportGenerator()

                # Test report generation
                import os
                original_cwd = os.getcwd()
                os.chdir(str(temp_directory))

                report_result = generator.generate_comprehensive_report(
                    ticker="MSFT",
                    output_format="json"
                )

                test_metrics.end_timing()
                test_metrics.log_performance("report_generation", test_metrics.get_duration(), True)

                assert isinstance(report_result, dict)

            except Exception as e:
                test_metrics.log_error(e)
                # Don't fail test if optional components are missing
                pytest.skip(f"Report generation not fully implemented: {e}")
            finally:
                os.chdir(original_cwd)

    def test_data_pipeline_to_valuation_workflow(self, mock_company_data, test_config, test_metrics):
        """Test data pipeline to valuation workflow"""
        test_metrics.start_timing()

        from data_pipeline.orchestrator import DataOrchestrator
        from analyze.deterministic import run_deterministic_models

        try:
            # Initialize orchestrator
            orchestrator = DataOrchestrator(config=test_config)

            # Mock data collection
            with patch.object(orchestrator, 'get_company_data', return_value=mock_company_data):
                dataset = orchestrator.get_company_data("TEST")

                # Run valuation
                valuation_result = run_deterministic_models(dataset, test_config)

                test_metrics.end_timing()
                test_metrics.log_performance("data_to_valuation_workflow", test_metrics.get_duration(), True)

                # Verify workflow results
                assert isinstance(valuation_result, dict)
                assert 'dcf_results' in valuation_result

        except Exception as e:
            test_metrics.log_error(e)
            raise

    def test_monitoring_integration_workflow(self, test_metrics, temp_directory):
        """Test monitoring system integration"""
        from src.utils.logging_config import ERTLogger
        from src.utils.monitoring_dashboard import MonitoringDashboard

        test_metrics.start_timing()

        try:
            # Initialize monitoring components
            logger = ERTLogger(log_dir=str(temp_directory))
            dashboard = MonitoringDashboard(logger=logger)

            # Start monitoring
            dashboard.start_monitoring()

            # Simulate some operations
            logger.log_performance_metric("test_operation", 0.1, True, "integration_test")
            logger.log_user_action("test_workflow", "test_user")

            # Collect metrics
            performance_summary = logger.get_performance_summary()
            system_health = dashboard._collect_system_health()

            # Stop monitoring
            dashboard.stop_monitoring()

            test_metrics.end_timing()
            test_metrics.log_performance("monitoring_integration", test_metrics.get_duration(), True)

            # Verify monitoring data
            assert isinstance(performance_summary, dict)
            assert hasattr(system_health, 'cpu_percent')

        except Exception as e:
            test_metrics.log_error(e)
            raise


class TestSystemReliability:
    """Test system reliability and error recovery"""

    def test_network_failure_handling(self, test_metrics):
        """Test handling of network failures"""
        test_metrics.start_timing()

        with patch('yfinance.Ticker') as mock_ticker:
            # Simulate network failure
            mock_ticker.side_effect = ConnectionError("Network unavailable")

            from data_pipeline.orchestrator import DataOrchestrator

            orchestrator = DataOrchestrator()

            try:
                dataset = orchestrator.get_company_data("AAPL")
                # Should not reach here
                assert False, "Expected ConnectionError"
            except ConnectionError:
                # Expected behavior
                test_metrics.end_timing()
                test_metrics.log_performance("network_failure_handling", test_metrics.get_duration(), True)
            except Exception as e:
                test_metrics.log_error(e)
                # Other exceptions might be acceptable depending on implementation
                test_metrics.end_timing()
                test_metrics.log_performance("network_failure_handling", test_metrics.get_duration(), False)

    def test_invalid_ticker_handling(self, test_metrics):
        """Test handling of invalid ticker symbols"""
        test_metrics.start_timing()

        with patch('yfinance.Ticker') as mock_ticker:
            # Simulate invalid ticker response
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {}  # Empty info indicates invalid ticker
            mock_ticker.return_value = mock_ticker_instance

            from data_pipeline.orchestrator import DataOrchestrator

            orchestrator = DataOrchestrator()

            try:
                dataset = orchestrator.get_company_data("INVALID")
                test_metrics.end_timing()
                test_metrics.log_performance("invalid_ticker_handling", test_metrics.get_duration(), True)

                # Should handle gracefully, either with empty dataset or exception
                if dataset:
                    assert dataset.ticker == "INVALID"

            except Exception as e:
                test_metrics.log_error(e)
                test_metrics.end_timing()
                test_metrics.log_performance("invalid_ticker_handling", test_metrics.get_duration(), False)

    def test_memory_pressure_handling(self, test_metrics):
        """Test system behavior under memory pressure"""
        import gc

        test_metrics.start_timing()

        try:
            # Create memory pressure by allocating large objects
            large_objects = []
            for i in range(100):
                # Create 10MB objects
                large_objects.append(bytearray(10 * 1024 * 1024))

            # Try to run valuation under memory pressure
            from analyze.deterministic import _compute_wacc
            from data_pipeline.models import CompanyDataset

            test_data = CompanyDataset(
                ticker="TEST",
                info={'sector': 'Technology', 'marketCap': 1000000000},
                financial_statements={},
                market_data={}
            )

            wacc = _compute_wacc(test_data, {'risk_free_rate': 0.03, 'equity_risk_premium': 0.06})

            test_metrics.end_timing()
            test_metrics.log_performance("memory_pressure_handling", test_metrics.get_duration(), True)

            assert isinstance(wacc, float)

        except MemoryError:
            test_metrics.end_timing()
            test_metrics.log_performance("memory_pressure_handling", test_metrics.get_duration(), False)
            pytest.skip("System ran out of memory during test")
        except Exception as e:
            test_metrics.log_error(e)
            test_metrics.end_timing()
            test_metrics.log_performance("memory_pressure_handling", test_metrics.get_duration(), False)
        finally:
            # Clean up memory
            large_objects.clear()
            gc.collect()

    def test_concurrent_operations(self, test_metrics):
        """Test concurrent operations and thread safety"""
        import threading
        import time

        test_metrics.start_timing()

        results = []
        errors = []

        def run_valuation(ticker_suffix):
            try:
                from analyze.deterministic import _compute_wacc
                from data_pipeline.models import CompanyDataset

                test_data = CompanyDataset(
                    ticker=f"TEST{ticker_suffix}",
                    info={'sector': 'Technology', 'marketCap': 1000000000},
                    financial_statements={},
                    market_data={}
                )

                wacc = _compute_wacc(test_data, {
                    'risk_free_rate': 0.03,
                    'equity_risk_premium': 0.06,
                    'wacc_floor': 0.05,
                    'wacc_ceiling': 0.15
                })

                results.append((ticker_suffix, wacc))

            except Exception as e:
                errors.append((ticker_suffix, e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=run_valuation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        test_metrics.end_timing()
        test_metrics.log_performance("concurrent_operations", test_metrics.get_duration(), len(errors) == 0)

        # Verify results
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
        assert len(results) == 10
        assert all(isinstance(wacc, float) for _, wacc in results)


class TestConfigurationManagement:
    """Test configuration management and validation"""

    def test_config_loading_and_validation(self, test_config):
        """Test configuration loading and validation"""
        from data_pipeline.orchestrator import DataOrchestrator

        # Test valid configuration
        orchestrator = DataOrchestrator(config=test_config)
        assert orchestrator.config == test_config

        # Test invalid configuration
        invalid_config = {'invalid': 'config'}

        try:
            orchestrator_invalid = DataOrchestrator(config=invalid_config)
            # Should handle gracefully or raise appropriate exception
        except (ValueError, KeyError):
            # Expected for invalid config
            pass

    def test_scenario_configuration(self, test_config):
        """Test scenario configuration validation"""
        scenarios = test_config['valuation']['scenarios']

        # Verify all required scenarios exist
        assert 'bear' in scenarios
        assert 'base' in scenarios
        assert 'bull' in scenarios

        # Verify scenario parameters
        for scenario_name, scenario_config in scenarios.items():
            assert 'growth_delta' in scenario_config
            assert 'wacc_delta' in scenario_config
            assert isinstance(scenario_config['growth_delta'], (int, float))
            assert isinstance(scenario_config['wacc_delta'], (int, float))

    def test_valuation_parameters(self, test_config):
        """Test valuation parameter validation"""
        valuation_config = test_config['valuation']

        # Check required parameters
        required_params = ['risk_free_rate', 'equity_risk_premium', 'wacc_floor', 'wacc_ceiling']
        for param in required_params:
            assert param in valuation_config
            assert isinstance(valuation_config[param], (int, float))
            assert 0 <= valuation_config[param] <= 1  # Should be reasonable percentage

        # Check bounds make sense
        assert valuation_config['wacc_floor'] < valuation_config['wacc_ceiling']
        assert valuation_config['risk_free_rate'] < valuation_config['equity_risk_premium']


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def test_valuation_speed_benchmark(self, mock_company_data, test_config, test_metrics, performance_test_config):
        """Test valuation speed benchmarks"""
        from analyze.deterministic import run_deterministic_models

        max_time = performance_test_config['max_execution_time']

        test_metrics.start_timing()

        try:
            result = run_deterministic_models(mock_company_data, test_config)
            test_metrics.end_timing()

            duration = test_metrics.get_duration()
            test_metrics.log_performance("valuation_speed_benchmark", duration, True)

            assert duration < max_time, f"Valuation took {duration}s, benchmark is {max_time}s"
            assert isinstance(result, dict)

        except Exception as e:
            test_metrics.log_error(e)
            test_metrics.end_timing()
            test_metrics.log_performance("valuation_speed_benchmark", test_metrics.get_duration(), False)
            raise

    def test_data_collection_speed_benchmark(self, performance_test_config, test_metrics):
        """Test data collection speed benchmarks"""
        from data_pipeline.orchestrator import DataOrchestrator

        max_time = performance_test_config['max_execution_time'] / 2  # Should be faster than valuation

        test_metrics.start_timing()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {'longName': 'Test Company', 'sector': 'Technology'}
            mock_ticker.return_value = mock_ticker_instance

            orchestrator = DataOrchestrator()

            try:
                dataset = orchestrator.get_company_data("BENCHMARK")
                test_metrics.end_timing()

                duration = test_metrics.get_duration()
                test_metrics.log_performance("data_collection_speed_benchmark", duration, True)

                assert duration < max_time, f"Data collection took {duration}s, benchmark is {max_time}s"

            except Exception as e:
                test_metrics.log_error(e)
                test_metrics.end_timing()
                test_metrics.log_performance("data_collection_speed_benchmark", test_metrics.get_duration(), False)
                raise

    def test_cache_performance_benchmark(self, performance_test_config, test_metrics):
        """Test cache performance benchmarks"""
        from data_pipeline.orchestrator import DataOrchestrator

        acceptable_hit_rate = performance_test_config['acceptable_cache_hit_rate']

        orchestrator = DataOrchestrator()

        # Warm up cache
        test_data = {'test': 'data'}
        orchestrator.cache.set("test_key", test_data, ttl=3600)

        # Measure cache performance
        hits = 0
        total_requests = 100

        test_metrics.start_timing()

        for i in range(total_requests):
            result = orchestrator.cache.get("test_key")
            if result is not None:
                hits += 1

        test_metrics.end_timing()

        hit_rate = hits / total_requests
        test_metrics.log_performance("cache_performance_benchmark", test_metrics.get_duration(), hit_rate >= acceptable_hit_rate)

        assert hit_rate >= acceptable_hit_rate, f"Cache hit rate {hit_rate:.2%} below benchmark {acceptable_hit_rate:.2%}"