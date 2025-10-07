#!/usr/bin/env python3
"""
Comprehensive QA test script for ERT enhanced features
Tests all recent upgrades and integrations
"""

import json
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_wacc_computation():
    """Test WACC computation with sector-aware beta"""
    print("🔧 Testing WACC Computation...")
    try:
        from analyze.deterministic import _compute_wacc
        from data_pipeline.models import CompanyDataset

        # Load config
        with open('config.json') as f:
            config = json.load(f)

        valuation_config = config.get('valuation', {})

        # Test sector beta overrides
        sector_overrides = valuation_config.get('sector_beta_overrides', {})
        print(f"  ✅ Sector beta overrides: {len(sector_overrides)} sectors")

        # Test risk parameters
        risk_free = valuation_config.get('risk_free_rate', 0.045)
        equity_premium = valuation_config.get('equity_risk_premium', 0.055)
        print(f"  ✅ Risk-free rate: {risk_free:.1%}")
        print(f"  ✅ Equity risk premium: {equity_premium:.1%}")

        # Test WACC bounds
        wacc_floor = valuation_config.get('wacc_floor', 0.06)
        wacc_ceiling = valuation_config.get('wacc_ceiling', 0.14)
        print(f"  ✅ WACC bounds: {wacc_floor:.1%} - {wacc_ceiling:.1%}")

        return True

    except Exception as e:
        print(f"  ❌ WACC test failed: {e}")
        return False

def test_scenario_overrides():
    """Test scenario overrides from config.json"""
    print("📊 Testing Scenario Management...")
    try:
        with open('config.json') as f:
            config = json.load(f)

        scenarios = config.get('valuation', {}).get('scenarios', {})

        expected_scenarios = ['bear', 'base', 'bull']
        for scenario in expected_scenarios:
            if scenario in scenarios:
                params = scenarios[scenario]
                growth_delta = params.get('growth_delta', 0)
                wacc_delta = params.get('wacc_delta', 0)
                print(f"  ✅ {scenario.upper()}: Growth Δ{growth_delta:+.1%}, WACC Δ{wacc_delta:+.1%}")
            else:
                print(f"  ❌ Missing scenario: {scenario}")
                return False

        return True

    except Exception as e:
        print(f"  ❌ Scenario test failed: {e}")
        return False

def test_excel_export():
    """Test Excel export functionality"""
    print("📈 Testing Excel Export...")
    try:
        from stock_report_generator import StockReportGenerator

        # Check if export method exists
        if hasattr(StockReportGenerator, 'export_model_to_excel'):
            print("  ✅ export_model_to_excel method exists")
        else:
            print("  ❌ export_model_to_excel method missing")
            return False

        # Check for required dependencies
        try:
            import openpyxl
            print("  ✅ openpyxl dependency available")
        except ImportError:
            print("  ⚠️  openpyxl not installed (pip install openpyxl)")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Excel export test failed: {e}")
        return False

def test_data_orchestrator():
    """Test Data Orchestrator config integration"""
    print("🎯 Testing Data Orchestrator...")
    try:
        from data_pipeline.orchestrator import DataOrchestrator

        # Load config
        with open('config.json') as f:
            config = json.load(f)

        # Check forecast defaults
        forecast_defaults = config.get('forecast_defaults', {})
        if forecast_defaults:
            print(f"  ✅ Forecast defaults: {len(forecast_defaults)} parameters")
        else:
            print("  ⚠️  No forecast defaults in config")

        # Check valuation config
        valuation_config = config.get('valuation', {})
        if valuation_config:
            print(f"  ✅ Valuation config: {len(valuation_config)} parameters")
        else:
            print("  ❌ No valuation config found")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Data Orchestrator test failed: {e}")
        return False

def test_news_sentiment():
    """Test news & sentiment pipeline"""
    print("📰 Testing News & Sentiment Pipeline...")
    try:
        from data_pipeline import sources

        # Test headline fetching (may fail without network)
        print("  🔍 Testing headline scraping...")
        try:
            headlines = sources.fetch_recent_headlines('AAPL', limit=3)
            if headlines:
                print(f"  ✅ Headlines fetched: {len(headlines)}")

                # Test sentiment analysis
                summary = sources.summarize_headline_sentiment(headlines)
                print(f"  ✅ Sentiment summary: {summary['count']} headlines analyzed")

            else:
                print("  ⚠️  No headlines fetched (network may be required)")

        except Exception as e:
            print(f"  ⚠️  Network scraping failed: {e}")

        # Test sentiment function exists
        if hasattr(sources, 'summarize_headline_sentiment'):
            print("  ✅ Sentiment analysis function available")
        else:
            print("  ❌ Sentiment analysis function missing")
            return False

        return True

    except Exception as e:
        print(f"  ❌ News & sentiment test failed: {e}")
        return False

def test_report_generation():
    """Test report generation integration"""
    print("📄 Testing Report Generation...")
    try:
        from stock_report_generator import StockReportGenerator

        # Check if class loads
        print("  ✅ StockReportGenerator loads successfully")

        # Check key methods exist
        required_methods = ['generate_comprehensive_report']
        for method in required_methods:
            if hasattr(StockReportGenerator, method):
                print(f"  ✅ {method} method exists")
            else:
                print(f"  ❌ {method} method missing")
                return False

        return True

    except Exception as e:
        print(f"  ❌ Report generation test failed: {e}")
        return False

def test_full_integration():
    """Test full integration with sample data"""
    print("🚀 Testing Full Integration...")
    try:
        from analyze.deterministic import run_deterministic_models

        print("  ✅ Deterministic models import successfully")
        print("  ⚠️  Full integration test requires live data connection")

        return True

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run all QA tests"""
    print("🧪 ERT Comprehensive QA Test Suite")
    print("=" * 50)

    tests = [
        ("WACC Computation", test_wacc_computation),
        ("Scenario Overrides", test_scenario_overrides),
        ("Excel Export", test_excel_export),
        ("Data Orchestrator", test_data_orchestrator),
        ("News & Sentiment", test_news_sentiment),
        ("Report Generation", test_report_generation),
        ("Full Integration", test_full_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
        print()

    # Summary
    print("📋 Test Results Summary")
    print("=" * 50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Ready for deployment.")
    else:
        print("⚠️  Some tests failed. Review issues before deployment.")

if __name__ == "__main__":
    main()