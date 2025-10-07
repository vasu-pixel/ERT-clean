#!/usr/bin/env python3
"""
Standalone Excel Model Creator
Creates comprehensive institutional-grade Excel valuation models
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stock_report_generator import StockReportGenerator
from utils.ollama_engine import OllamaEngine
from excel.advanced_valuation_model import AdvancedValuationExcel

def create_excel_model(ticker: str, output_dir: str = "models", force_refresh: bool = True):
    """Create comprehensive Excel valuation model for a ticker"""

    print(f"🚀 Creating Excel valuation model for {ticker}")
    print("=" * 60)

    try:
        # Initialize components
        ai_engine = OllamaEngine()
        generator = StockReportGenerator(ai_engine)
        excel_creator = AdvancedValuationExcel()

        # Refresh company data
        print(f"📊 Fetching comprehensive data for {ticker}...")
        dataset = generator.data_orchestrator.refresh_company_data(ticker, force=force_refresh)

        # Get deterministic analysis
        print(f"🔬 Running enhanced valuation models...")
        deterministic_data = dataset.supplemental.get("deterministic", {})

        if not deterministic_data:
            print("❌ No deterministic data available. Ensure data pipeline is working.")
            return None

        # Create output path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = output_path / f"{ticker}_Valuation_Model_{timestamp}.xlsx"

        # Create Excel model
        print(f"📈 Creating advanced Excel model...")
        model_path = excel_creator.create_comprehensive_model(
            ticker,
            dataset,
            deterministic_data,
            str(excel_file)
        )

        # Print model summary
        print("\n" + "=" * 60)
        print("🎉 EXCEL MODEL CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 File: {model_path}")
        print(f"📊 Ticker: {ticker}")
        print(f"💰 DCF Value: ${deterministic_data.get('valuation', {}).get('dcf_value', 'N/A')}")
        print(f"📅 Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n📋 Model Contents:")
        sheets = [
            "🎯 Dashboard - Executive summary",
            "📊 Live Market Data - Real-time parameters",
            "💰 DCF Model - Cash flow valuation",
            "🌪️ Scenario Analysis - Bear/Base/Bull cases",
            "🔥 Sensitivity Analysis - Heat maps",
            "👥 Comparable Analysis - Peer comparison",
            "📈 Historical Analysis - 5-year trends",
            "🎲 Monte Carlo - Simulation analysis",
            "⚙️ Assumptions - Interactive inputs",
            "📋 Raw Data - Complete reference"
        ]

        for sheet in sheets:
            print(f"   {sheet}")

        print(f"\n💡 Features:")
        print(f"   ✅ Live market data integration")
        print(f"   ✅ Dynamic WACC calculation")
        print(f"   ✅ Multi-scenario analysis")
        print(f"   ✅ Interactive sensitivity tables")
        print(f"   ✅ Monte Carlo simulation")
        print(f"   ✅ Institutional formatting")

        return model_path

    except Exception as e:
        print(f"❌ Error creating Excel model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Create comprehensive Excel valuation model")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("-o", "--output", default="models", help="Output directory")
    parser.add_argument("--no-refresh", action="store_true", help="Use cached data")

    args = parser.parse_args()

    model_path = create_excel_model(
        ticker=args.ticker.upper(),
        output_dir=args.output,
        force_refresh=not args.no_refresh
    )

    if model_path:
        print(f"\n✅ Success! Excel model saved to: {model_path}")
        print(f"📖 Open the file in Excel to explore the comprehensive valuation model.")
    else:
        print(f"\n❌ Failed to create Excel model for {args.ticker}")
        sys.exit(1)

if __name__ == "__main__":
    main()