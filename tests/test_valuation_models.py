"""
Test suite for valuation models and deterministic analysis
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from analyze.deterministic import (
    run_deterministic_models,
    _compute_wacc,
    _compute_dcf_valuation,
    _compute_comparables_valuation
)


class TestWACCCalculation:
    """Test WACC calculation functionality"""

    def test_wacc_basic_calculation(self, test_config, mock_company_data):
        """Test basic WACC calculation"""
        wacc = _compute_wacc(mock_company_data, test_config['valuation'])

        assert isinstance(wacc, float)
        assert 0.04 <= wacc <= 0.20  # Reasonable WACC range
        assert wacc >= test_config['valuation']['wacc_floor']
        assert wacc <= test_config['valuation']['wacc_ceiling']

    def test_wacc_sector_overrides(self, test_config, mock_company_data):
        """Test WACC calculation with sector beta overrides"""
        # Test technology sector override
        tech_company = mock_company_data
        tech_company.info['sector'] = 'Technology'

        wacc_tech = _compute_wacc(tech_company, test_config['valuation'])

        # Test healthcare sector override
        healthcare_company = mock_company_data
        healthcare_company.info['sector'] = 'Healthcare'

        wacc_healthcare = _compute_wacc(healthcare_company, test_config['valuation'])

        # Healthcare should have lower WACC than technology (lower beta)
        assert wacc_healthcare < wacc_tech

    def test_wacc_bounds_enforcement(self, test_config, mock_company_data):
        """Test WACC bounds enforcement"""
        # Modify config to test bounds
        extreme_config = test_config['valuation'].copy()
        extreme_config['wacc_floor'] = 0.15
        extreme_config['wacc_ceiling'] = 0.16

        wacc = _compute_wacc(mock_company_data, extreme_config)

        assert extreme_config['wacc_floor'] <= wacc <= extreme_config['wacc_ceiling']

    def test_wacc_missing_data_handling(self, test_config):
        """Test WACC calculation with missing data"""
        from data_pipeline.models import CompanyDataset

        # Create dataset with minimal data
        minimal_data = CompanyDataset(
            ticker="TEST",
            info={'sector': 'Technology'},
            financial_statements={},
            market_data={}
        )

        wacc = _compute_wacc(minimal_data, test_config['valuation'])

        # Should still return a valid WACC using defaults
        assert isinstance(wacc, float)
        assert wacc > 0


class TestDCFValuation:
    """Test DCF valuation model"""

    def test_dcf_basic_calculation(self, mock_company_data, test_config):
        """Test basic DCF calculation"""
        valuation_result = _compute_dcf_valuation(mock_company_data, test_config)

        assert isinstance(valuation_result, dict)
        assert 'enterprise_value' in valuation_result
        assert 'equity_value' in valuation_result
        assert 'price_per_share' in valuation_result
        assert 'wacc' in valuation_result

        # Sanity checks
        assert valuation_result['enterprise_value'] > 0
        assert valuation_result['equity_value'] > 0
        assert valuation_result['price_per_share'] > 0

    def test_dcf_scenario_analysis(self, mock_company_data, test_config):
        """Test DCF scenario analysis"""
        scenarios = ['bear', 'base', 'bull']
        results = {}

        for scenario in scenarios:
            result = _compute_dcf_valuation(mock_company_data, test_config, scenario=scenario)
            results[scenario] = result

        # Bull case should have higher valuation than bear case
        assert results['bull']['price_per_share'] > results['bear']['price_per_share']
        assert results['bull']['enterprise_value'] > results['bear']['enterprise_value']

    def test_dcf_sensitivity_analysis(self, mock_company_data, test_config):
        """Test DCF sensitivity to key parameters"""
        base_result = _compute_dcf_valuation(mock_company_data, test_config)

        # Test sensitivity to terminal growth rate
        high_growth_config = test_config.copy()
        high_growth_config['forecast_defaults']['terminal_growth_rate'] = 0.035

        high_growth_result = _compute_dcf_valuation(mock_company_data, high_growth_config)

        # Higher terminal growth should result in higher valuation
        assert high_growth_result['price_per_share'] > base_result['price_per_share']

    def test_dcf_cash_flow_projections(self, mock_company_data, test_config):
        """Test cash flow projection logic"""
        result = _compute_dcf_valuation(mock_company_data, test_config)

        if 'cash_flow_projections' in result:
            projections = result['cash_flow_projections']
            assert len(projections) > 0
            assert all(cf > 0 for cf in projections)  # All cash flows should be positive

    def test_dcf_terminal_value_calculation(self, mock_company_data, test_config):
        """Test terminal value calculation"""
        result = _compute_dcf_valuation(mock_company_data, test_config)

        if 'terminal_value' in result:
            terminal_value = result['terminal_value']
            assert terminal_value > 0
            assert terminal_value < result['enterprise_value']  # Terminal value should be less than total EV after discounting


class TestComparablesValuation:
    """Test comparables/relative valuation"""

    def test_comparables_basic_calculation(self, mock_company_data):
        """Test basic comparables calculation"""
        # Mock peer data
        mock_peers = [
            {'ticker': 'PEER1', 'pe_ratio': 25.0, 'pb_ratio': 8.0, 'ev_ebitda': 15.0},
            {'ticker': 'PEER2', 'pe_ratio': 22.0, 'pb_ratio': 7.5, 'ev_ebitda': 14.0},
            {'ticker': 'PEER3', 'pe_ratio': 28.0, 'pb_ratio': 9.0, 'ev_ebitda': 16.0}
        ]

        with patch('analyze.deterministic._get_peer_companies') as mock_peers_func:
            mock_peers_func.return_value = mock_peers

            result = _compute_comparables_valuation(mock_company_data)

            assert isinstance(result, dict)
            assert 'pe_valuation' in result
            assert 'pb_valuation' in result
            assert 'ev_ebitda_valuation' in result
            assert 'average_valuation' in result

    def test_comparables_outlier_handling(self, mock_company_data):
        """Test handling of outlier peer multiples"""
        # Include some outlier peers
        mock_peers = [
            {'ticker': 'NORMAL1', 'pe_ratio': 25.0, 'pb_ratio': 8.0},
            {'ticker': 'NORMAL2', 'pe_ratio': 22.0, 'pb_ratio': 7.5},
            {'ticker': 'OUTLIER', 'pe_ratio': 100.0, 'pb_ratio': 50.0}  # Extreme outlier
        ]

        with patch('analyze.deterministic._get_peer_companies') as mock_peers_func:
            mock_peers_func.return_value = mock_peers

            result = _compute_comparables_valuation(mock_company_data)

            # Average should not be heavily skewed by outlier
            avg_pe = (25.0 + 22.0) / 2  # Outlier should be excluded
            assert abs(result.get('peer_avg_pe', 0) - avg_pe) < 5.0

    def test_comparables_insufficient_peers(self, mock_company_data):
        """Test handling when insufficient peer data is available"""
        mock_peers = []  # No peers

        with patch('analyze.deterministic._get_peer_companies') as mock_peers_func:
            mock_peers_func.return_value = mock_peers

            result = _compute_comparables_valuation(mock_company_data)

            # Should handle gracefully
            assert isinstance(result, dict)
            assert result.get('status') == 'insufficient_data' or len(result) == 0


class TestIntegratedValuation:
    """Test integrated valuation combining multiple methods"""

    def test_full_deterministic_analysis(self, mock_company_data, test_config, test_metrics):
        """Test full deterministic analysis pipeline"""
        test_metrics.start_timing()

        try:
            result = run_deterministic_models(mock_company_data, test_config)
            test_metrics.end_timing()
            test_metrics.log_performance("full_deterministic_analysis", test_metrics.get_duration(), True)

            assert isinstance(result, dict)
            assert 'dcf_results' in result
            assert 'comparables_results' in result
            assert 'summary' in result

            # Verify all scenarios are included
            dcf_results = result['dcf_results']
            assert 'bear' in dcf_results
            assert 'base' in dcf_results
            assert 'bull' in dcf_results

        except Exception as e:
            test_metrics.log_error(e)
            raise

    def test_valuation_reasonableness_checks(self, mock_company_data, test_config):
        """Test valuation reasonableness and sanity checks"""
        result = run_deterministic_models(mock_company_data, test_config)

        # Extract key metrics
        base_price = result['dcf_results']['base']['price_per_share']
        market_cap = mock_company_data.info.get('marketCap', 0)
        current_price = mock_company_data.market_data.get('current_price', 0)

        if current_price > 0:
            # Valuation should be within reasonable range of current price
            price_ratio = base_price / current_price
            assert 0.5 <= price_ratio <= 2.0, f"Valuation seems unreasonable: {price_ratio:.2f}x current price"

    def test_consistency_across_scenarios(self, mock_company_data, test_config):
        """Test consistency across valuation scenarios"""
        result = run_deterministic_models(mock_company_data, test_config)
        dcf_results = result['dcf_results']

        bear_price = dcf_results['bear']['price_per_share']
        base_price = dcf_results['base']['price_per_share']
        bull_price = dcf_results['bull']['price_per_share']

        # Verify logical ordering
        assert bear_price <= base_price <= bull_price

        # Verify reasonable spreads (bull shouldn't be more than 3x bear)
        assert bull_price / bear_price <= 3.0

    def test_error_handling_invalid_data(self, test_config):
        """Test error handling with invalid or corrupted data"""
        from data_pipeline.models import CompanyDataset

        # Create dataset with problematic data
        bad_data = CompanyDataset(
            ticker="BAD",
            info={'marketCap': -1000000},  # Negative market cap
            financial_statements={
                'income_statement': {
                    'Total Revenue': [0, 0, 0],  # Zero revenues
                    'Net Income': [-1000000, -2000000, -3000000]  # Increasing losses
                }
            },
            market_data={'current_price': 0}  # Zero price
        )

        # Should handle gracefully without crashing
        try:
            result = run_deterministic_models(bad_data, test_config)
            # If it completes, result should indicate issues
            assert 'warnings' in result or 'errors' in result
        except ValueError as e:
            # Expected for severely bad data
            assert "invalid" in str(e).lower() or "data" in str(e).lower()


class TestPerformanceOptimization:
    """Test performance optimization features"""

    def test_calculation_caching(self, mock_company_data, test_config, test_metrics):
        """Test caching of expensive calculations"""
        # First calculation
        test_metrics.start_timing()
        result1 = _compute_dcf_valuation(mock_company_data, test_config)
        duration1 = test_metrics.get_duration()
        test_metrics.end_timing()

        # Second calculation (should use cache)
        test_metrics.start_timing()
        result2 = _compute_dcf_valuation(mock_company_data, test_config)
        duration2 = test_metrics.get_duration()
        test_metrics.end_timing()

        # Results should be identical
        assert result1['price_per_share'] == result2['price_per_share']

        # Second calculation should be faster (if caching is implemented)
        test_metrics.log_performance("cached_calculation", duration2, True)

    def test_parallel_scenario_processing(self, mock_company_data, test_config, test_metrics):
        """Test parallel processing of scenarios"""
        test_metrics.start_timing()

        result = run_deterministic_models(mock_company_data, test_config)

        test_metrics.end_timing()
        duration = test_metrics.get_duration()
        test_metrics.log_performance("parallel_scenarios", duration, True)

        # Verify all scenarios completed
        assert len(result['dcf_results']) == 3  # bear, base, bull

        # Performance should be reasonable for parallel processing
        assert duration < 10.0  # Should complete within 10 seconds

    def test_memory_efficiency(self, test_config):
        """Test memory efficiency during large calculations"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple valuations
        for i in range(10):
            from data_pipeline.models import CompanyDataset

            # Create mock dataset
            mock_data = CompanyDataset(
                ticker=f"TEST{i}",
                info={'marketCap': 1000000000, 'sector': 'Technology'},
                financial_statements={
                    'income_statement': {
                        'Total Revenue': [1000000000, 1100000000, 1200000000],
                        'Net Income': [100000000, 110000000, 120000000]
                    }
                },
                market_data={'current_price': 100.0}
            )

            run_deterministic_models(mock_data, test_config)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for 10 valuations)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase}MB"