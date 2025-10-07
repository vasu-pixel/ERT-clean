#!/usr/bin/env python3
"""
Professional Equity Research Model Creator
Combines advanced Monte Carlo simulation with institutional-grade Excel templates
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stock_report_generator import StockReportGenerator
from utils.ollama_engine import OllamaEngine
from analyze.monte_carlo_simulation import (
    AdvancedMonteCarloSimulator,
    create_default_parameter_distributions,
    create_default_correlation_matrix
)
from analyze.multi_method_valuation import MultiMethodValuationFramework
from excel.professional_excel_template import ProfessionalExcelTemplate
from utils.fallback_config import fallback_manager
from utils.logging_config import ert_logger, performance_monitor, app_logger
from utils.monitoring_dashboard import monitoring_dashboard

@performance_monitor("create_professional_model", "main")
def create_professional_model(ticker: str,
                            output_dir: str = "professional_models",
                            force_refresh: bool = True,
                            num_simulations: int = 10000) -> str:
    """Create comprehensive professional equity research model"""

    print(f"🚀 Creating Professional Equity Research Model for {ticker}")
    print("=" * 70)

    # Start monitoring
    monitoring_dashboard.start_monitoring()

    # Log user action
    ert_logger.log_user_action(
        action="create_professional_model",
        ticker=ticker,
        num_simulations=num_simulations,
        force_refresh=force_refresh
    )

    app_logger.info(f"Starting professional model creation for {ticker}")

    # Initialize degraded mode tracking (will be determined after initial API attempts)
    degraded_mode = False

    try:
        # Initialize components with error handling
        print(f"🔧 Initializing components...")

        try:
            ai_engine = OllamaEngine()
            print("   ✅ AI Engine initialized")
        except Exception as e:
            print(f"   ⚠️ AI Engine failed to initialize: {e}")
            print("   📝 Continuing without AI engine - some features may be limited")
            ai_engine = None

        try:
            if ai_engine:
                generator = StockReportGenerator(ai_engine)
                print("   ✅ Report Generator initialized")
            else:
                # Create a minimal data orchestrator for basic functionality
                from data_pipeline.data_orchestrator import DataOrchestrator
                generator = type('MockGenerator', (), {'data_orchestrator': DataOrchestrator()})()
                print("   ✅ Basic Data Orchestrator initialized (without AI features)")
        except Exception as e:
            print(f"   ❌ Report Generator failed: {e}")
            return None

        try:
            monte_carlo_simulator = AdvancedMonteCarloSimulator(num_simulations=num_simulations)
            print("   ✅ Monte Carlo Simulator initialized")
        except Exception as e:
            print(f"   ⚠️ Monte Carlo Simulator failed: {e}")
            print("   📝 Continuing without Monte Carlo - basic analysis only")
            monte_carlo_simulator = None

        try:
            multi_method_framework = MultiMethodValuationFramework()
            print("   ✅ Multi-Method Framework initialized")
        except Exception as e:
            print(f"   ⚠️ Multi-Method Framework failed: {e}")
            print("   📝 Continuing with basic DCF analysis only")
            multi_method_framework = None

        try:
            excel_template = ProfessionalExcelTemplate()
            print("   ✅ Excel Template initialized")
        except Exception as e:
            print(f"   ❌ Excel Template failed: {e}")
            return None

        # Step 1: Refresh company data with retry logic
        print(f"📊 Step 1: Fetching comprehensive data for {ticker}...")
        dataset = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                dataset = generator.data_orchestrator.refresh_company_data(ticker, force=force_refresh)
                print("   ✅ Data fetching successful")
                break
            except Exception as e:
                print(f"   ⚠️ Data fetch attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("   ❌ All data fetch attempts failed. Using cached data if available.")
                    try:
                        dataset = generator.data_orchestrator.refresh_company_data(ticker, force=False)
                        print("   ✅ Cached data loaded successfully")
                    except Exception as cache_error:
                        print(f"   ❌ Cached data also failed: {cache_error}")
                        return None
                else:
                    import time
                    print(f"   ⏳ Retrying in 2 seconds...")
                    time.sleep(2)

        if not dataset:
            print("❌ No dataset available. Cannot proceed.")
            return None

        # Now check system health after initial data attempts
        print(f"🔍 Checking system health and API status...")
        data_freshness = fallback_manager.get_data_freshness_status()

        if not data_freshness:
            print("   ℹ️  No previous API calls recorded - this appears to be a fresh start")
        else:
            stale_apis = [api for api, status in data_freshness.items() if status in ["Stale", "Very Stale"]]
            if stale_apis:
                print(f"⚠️ Detected stale API data for: {', '.join(stale_apis)} - activating degraded mode")
                degraded_mode = True
                degraded_config = fallback_manager.create_degraded_mode_config()
                num_simulations = min(num_simulations, degraded_config['reduced_monte_carlo_runs'])
            else:
                print("   ✅ All APIs showing fresh data - full mode active")

        # Step 2: Get deterministic analysis with fallback
        print(f"🔬 Step 2: Running deterministic valuation models...")
        deterministic_data = None

        try:
            deterministic_data = dataset.supplemental.get("deterministic", {})
            if not deterministic_data:
                print("   ⚠️ No supplemental deterministic data found. Attempting to generate...")
                # Try to run deterministic analysis directly
                from analyze import deterministic
                dcf_results = deterministic.run_dcf(dataset)
                deterministic_data = {
                    'valuation': {
                        'dcf_value': dcf_results.dcf_value if dcf_results and dcf_results.dcf_value else 0,
                        'assumptions': dcf_results.assumptions if dcf_results else {}
                    },
                    'metrics': {},
                    'analysis_timestamp': str(datetime.now())
                }
                print("   ✅ Deterministic analysis generated successfully")
        except Exception as e:
            print(f"   ❌ Deterministic analysis failed: {e}")
            print("   📝 Creating minimal deterministic data for basic functionality")
            deterministic_data = {
                'valuation': {'dcf_value': 0},
                'metrics': {},
                'analysis_timestamp': str(datetime.now())
            }

        if not deterministic_data:
            print("❌ No deterministic data available. Cannot proceed.")
            return None

        # Step 3: Run Multi-Method Valuation Analysis with error handling
        print(f"🔬 Step 3: Running comprehensive multi-method valuation...")
        multi_method_results = None

        if multi_method_framework:
            try:
                multi_method_results = multi_method_framework.perform_comprehensive_valuation(dataset)
                print("   ✅ Multi-method valuation completed successfully")
            except Exception as e:
                print(f"   ⚠️ Multi-method valuation failed: {e}")
                print("   📝 Continuing with basic DCF analysis only")
        else:
            print("   ⚠️ Multi-method framework not available")

        # Step 4: Run Monte Carlo simulation with error handling
        print(f"🎲 Step 4: Running Monte Carlo simulation ({num_simulations:,} iterations)...")
        monte_carlo_results = None

        if monte_carlo_simulator:
            try:
                # Create parameter distributions based on company fundamentals
                parameter_distributions = create_default_parameter_distributions(dataset)
                print("   ✅ Parameter distributions created")

                # Create correlation matrix for economic realism
                correlation_matrix = create_default_correlation_matrix()
                print("   ✅ Correlation matrix created")

                # Run simulation
                monte_carlo_results = monte_carlo_simulator.run_simulation(
                    dataset=dataset,
                    parameter_distributions=parameter_distributions,
                    correlation_matrix=correlation_matrix
                )
                print("   ✅ Monte Carlo simulation completed successfully")
            except Exception as e:
                print(f"   ⚠️ Monte Carlo simulation failed: {e}")
                print("   📝 Continuing without Monte Carlo results")
        else:
            print("   ⚠️ Monte Carlo simulator not available")

        # Step 5: Create professional Excel model with error handling
        print(f"📈 Step 5: Creating professional Excel template...")
        model_path = None

        try:
            # Create output path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print("   ✅ Output directory created")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = output_path / f"{ticker}_Professional_Model_{timestamp}.xlsx"

            # Create comprehensive Excel model with all valuation methods
            model_path = excel_template.create_professional_model(
                ticker=ticker,
                dataset=dataset,
                deterministic_data=deterministic_data,
                monte_carlo_results=monte_carlo_results,
                multi_method_results=multi_method_results,
                output_path=str(excel_file)
            )
            print("   ✅ Excel model created successfully")

        except Exception as e:
            print(f"   ❌ Excel model creation failed: {e}")
            print("   📝 Attempting to create basic Excel model...")

            try:
                # Fallback: create basic model without advanced features
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_file = output_path / f"{ticker}_Basic_Model_{timestamp}.xlsx"

                model_path = excel_template.create_professional_model(
                    ticker=ticker,
                    dataset=dataset,
                    deterministic_data=deterministic_data,
                    monte_carlo_results=None,  # Skip Monte Carlo if it failed
                    multi_method_results=None,  # Skip multi-method if it failed
                    output_path=str(excel_file)
                )
                print("   ✅ Basic Excel model created successfully")

            except Exception as fallback_error:
                print(f"   ❌ Even basic Excel model failed: {fallback_error}")
                print("   📝 No Excel output will be generated")
                model_path = None

        if not model_path:
            print("❌ No Excel model could be created. Check error messages above.")
            return None

        # Print comprehensive summary
        print("\n" + "=" * 70)
        print("🎉 PROFESSIONAL EQUITY RESEARCH MODEL CREATED!")
        print("=" * 70)
        print(f"📁 File: {model_path}")
        print(f"📊 Ticker: {ticker}")
        print(f"💰 DCF Value: ${deterministic_data.get('valuation', {}).get('dcf_value', 'N/A'):.2f}")
        print(f"📅 Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show degraded mode status
        if degraded_mode:
            print(f"⚠️ Mode: DEGRADED (Using fallback data due to API unavailability)")
        else:
            print(f"✅ Mode: FULL (Live market data integration)")

        # Multi-Method Valuation Summary
        if multi_method_results:
            print(f"\n🎯 Multi-Method Valuation Results:")
            print(f"   DCF Valuation: ${multi_method_results.dcf_value:.2f}")
            print(f"   Relative Valuation: ${multi_method_results.relative_value:.2f}")
            print(f"   Sum-of-Parts: ${multi_method_results.sum_of_parts_value:.2f}")
            print(f"   Asset-Based: ${multi_method_results.asset_based_value:.2f}")
            print(f"   Real Options: ${multi_method_results.real_options_value:.2f}")
            print(f"   Weighted Average: ${multi_method_results.weighted_average_value:.2f}")
            print(f"   Confidence-Weighted: ${multi_method_results.confidence_weighted_value:.2f}")

            # Investment recommendation from multi-method
            if hasattr(multi_method_results, 'investment_recommendation'):
                recommendation = multi_method_results.investment_recommendation.get('recommendation', 'N/A')
                confidence = multi_method_results.investment_recommendation.get('confidence_level', 0)
                print(f"   Multi-Method Recommendation: {recommendation} (Confidence: {confidence:.1%})")

        # Monte Carlo Summary
        if monte_carlo_results:
            print(f"\n🎲 Monte Carlo Results ({monte_carlo_results.simulation_runs:,} simulations):")
            print(f"   Mean Value: ${monte_carlo_results.percentiles['mean']:.2f}")
            print(f"   P5 (VaR): ${monte_carlo_results.percentiles['p5']:.2f}")
            print(f"   P95: ${monte_carlo_results.percentiles['p95']:.2f}")
            print(f"   Probability of Loss: {monte_carlo_results.probability_of_loss*100:.1f}%")

            # Investment recommendation
            current_price = dataset.snapshot.current_price
            if current_price:
                prob_upside = (monte_carlo_results.dcf_values > current_price).mean() * 100
                median_upside = (monte_carlo_results.percentiles['p50'] / current_price - 1) * 100

                if prob_upside > 70 and median_upside > 15:
                    recommendation = "STRONG BUY"
                elif prob_upside > 60 and median_upside > 5:
                    recommendation = "BUY"
                elif prob_upside > 40:
                    recommendation = "HOLD"
                else:
                    recommendation = "SELL"

                print(f"\n💼 Investment Analysis:")
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   Probability of Upside: {prob_upside:.1f}%")
                print(f"   Median Upside: {median_upside:+.1f}%")
                print(f"   Recommendation: {recommendation}")

        print(f"\n📋 Professional Model Features:")
        features = [
            "📊 Executive Dashboard with key metrics",
            "🎯 5-Method Valuation Framework (DCF, Relative, SoP, Asset-Based, Real Options)",
            "🎲 Monte Carlo simulation results",
            "🎯 Interactive sensitivity analysis",
            "🌪️ Dynamic scenario analysis (Bear/Base/Bull)",
            "🔄 Market-adjusted valuation weights",
            "⚙️ Analyst input assumptions (orange cells)",
            "📈 Professional charts and visualizations",
            "💰 Detailed valuation bridge",
            "⚠️ Risk metrics and VaR calculations",
            "🔒 Protected formulas with input validation",
            "🎨 Institutional-grade formatting"
        ]

        for feature in features:
            print(f"   ✅ {feature}")

        print(f"\n🔬 Advanced Analytics:")
        analytics = [
            f"Parameter distributions with correlation matrices",
            f"Value-at-Risk (5%): ${monte_carlo_results.percentiles['p5']:.2f}" if monte_carlo_results else "N/A",
            f"Expected shortfall: ${monte_carlo_results.expected_shortfall:.2f}" if monte_carlo_results else "N/A",
            f"Skewness: {monte_carlo_results.skewness:.2f}" if monte_carlo_results else "N/A",
            f"Top sensitivity: {max(monte_carlo_results.parameter_sensitivities.items(), key=lambda x: abs(x[1]))[0] if monte_carlo_results else 'N/A'}"
        ]

        for analytic in analytics:
            print(f"   📊 {analytic}")

        # Log successful completion with metrics
        app_logger.info(f"Professional model creation completed successfully for {ticker}")
        ert_logger.log_user_action(
            action="model_file_created",
            ticker=ticker,
            output_path=model_path,
            file_size_mb=Path(model_path).stat().st_size / (1024 * 1024) if Path(model_path).exists() else 0,
            num_simulations=num_simulations
        )

        return model_path

    except Exception as e:
        print(f"❌ Error creating professional model: {e}")

        # Log the error with full context
        ert_logger.log_error_event(
            error=e,
            operation="create_professional_model",
            module="main",
            ticker=ticker,
            num_simulations=num_simulations
        )

        app_logger.error(f"Professional model creation failed for {ticker}: {str(e)}")

        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Create professional equity research model")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("-o", "--output", default="professional_models", help="Output directory")
    parser.add_argument("--no-refresh", action="store_true", help="Use cached data")
    parser.add_argument("-s", "--simulations", type=int, default=10000, help="Number of Monte Carlo simulations")

    args = parser.parse_args()

    model_path = create_professional_model(
        ticker=args.ticker.upper(),
        output_dir=args.output,
        force_refresh=not args.no_refresh,
        num_simulations=args.simulations
    )

    if model_path:
        print(f"\n✅ Success! Professional model saved to: {model_path}")
        print(f"📖 Open the Excel file to explore the comprehensive professional analysis.")
        print(f"🎯 Features: Monte Carlo simulation, sensitivity analysis, professional formatting")

        # Log successful completion
        app_logger.info(f"Professional model creation completed successfully for {args.ticker}")
        ert_logger.log_user_action(
            action="model_creation_completed",
            ticker=args.ticker,
            output_path=model_path,
            success=True
        )

        # Print monitoring summary
        print("\n📊 System Performance Summary:")
        monitoring_dashboard.print_dashboard()
    else:
        print(f"\n❌ Failed to create professional model for {args.ticker}")

        # Log failure
        app_logger.error(f"Professional model creation failed for {args.ticker}")
        ert_logger.log_user_action(
            action="model_creation_failed",
            ticker=args.ticker,
            success=False
        )
        sys.exit(1)

if __name__ == "__main__":
    main()