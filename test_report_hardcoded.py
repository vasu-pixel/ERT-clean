#!/usr/bin/env python3
"""
Hardcoded test report generator for quality checking
Bypasses all API calls and uses mock data
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.ai_engine import OllamaEngine
from src.stock_report_generator import StockReportGenerator, CompanyProfile

# Hardcoded company data for AAPL
HARDCODED_AAPL_PROFILE = CompanyProfile(
    ticker="AAPL",
    company_name="Apple Inc.",
    sector="Technology",
    industry="Consumer Electronics",
    market_cap=2800000000000,  # $2.8T
    current_price=178.50,
    target_price=195.00,
    recommendation="BUY",
    financial_metrics={
        # Valuation Metrics
        'pe_ratio': 29.5,
        'forward_pe': 27.2,
        'peg_ratio': 2.1,
        'price_to_book': 45.8,
        'price_to_sales': 7.5,
        'ev_to_revenue': 7.2,
        'ev_to_ebitda': 22.5,

        # Profitability Metrics
        'gross_margin': 45.2,
        'operating_margin': 30.1,
        'profit_margin': 25.3,
        'roe': 147.5,
        'roa': 28.5,
        'roic': 55.2,

        # Growth Metrics
        'revenue_growth': 8.5,
        'earnings_growth': 12.3,

        # Financial Health
        'debt_to_equity': 181.5,
        'current_ratio': 1.05,
        'quick_ratio': 0.95,

        # Cash Flow
        'operating_cash_flow': 110000000000,
        'free_cash_flow': 99000000000,
        'fcf_yield': 3.5,

        # Other
        'beta': 1.25,
        'dividend_yield': 0.45,
        'payout_ratio': 15.2
    },
    competitors=["MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    esg_data={
        'esg_score': 72,
        'environmental_score': 68,
        'social_score': 75,
        'governance_score': 73,
        'controversy_level': 'Low',
        'sustainability_initiatives': [
            'Carbon neutral by 2030',
            '100% renewable energy in facilities',
            'Recycling program for old devices'
        ]
    },
    risk_factors=[
        'Market volatility and economic uncertainty',
        'Competitive pressure from industry peers',
        'Regulatory changes affecting operations',
        'Interest rate and inflation risks',
        'Supply chain disruptions',
        'China market dependency',
        'Product innovation cycle risks'
    ],
    deterministic={
        "forecast": {
            "revenue": {
                "ttm": 383000000000,
                "next_year": 415000000000,
                "year_two": 445000000000
            },
            "eps": {
                "ttm": 6.42,
                "next_year": 7.15,
                "year_two": 7.95
            }
        },
        "valuation": {
            "dcf_value": 192.50,
            "scenarios": {
                "base": 192.50,
                "bull": 215.75,
                "bear": 165.25
            },
            "multiples_summary": {
                "pe_multiple": 29.5,
                "ev_ebitda": 22.5,
                "price_to_sales": 7.5
            },
            "assumptions": {
                "wacc": 0.085,
                "projection_growth": 0.08,
                "terminal_growth": 0.03
            },
            "sensitivity": {
                "wacc_values": [0.075, 0.085, 0.095],
                "terminal_growth_values": [0.02, 0.03, 0.04],
                "dcf_matrix": [
                    [205.30, 192.50, 181.20],
                    [198.50, 187.30, 177.10],
                    [192.10, 182.50, 173.60]
                ]
            }
        },
        "fcf_projection": {
            "assumptions": {
                "base_revenue": 383000000000,
                "revenue_growth_rate": 0.08,
                "ebit_margin": 0.301,
                "tax_rate": 0.15,
                "depreciation_pct": 0.03,
                "capex_pct": 0.04,
                "working_capital_change_pct": 0.01
            },
            "schedule": [
                {
                    "year": 1,
                    "revenue": 413640000000,
                    "ebit": 124505400000,
                    "nopat": 105829590000,
                    "depreciation": 12409200000,
                    "capex": 16545600000,
                    "change_working_capital": 4136400000,
                    "free_cash_flow": 97556790000
                },
                {
                    "year": 2,
                    "revenue": 446731200000,
                    "ebit": 134466091200,
                    "nopat": 114296177520,
                    "depreciation": 13401936000,
                    "capex": 17869248000,
                    "change_working_capital": 4467312000,
                    "free_cash_flow": 105361553520
                },
                {
                    "year": 3,
                    "revenue": 482469696000,
                    "ebit": 145223378496,
                    "nopat": 123439871721,
                    "depreciation": 14474090880,
                    "capex": 19298787840,
                    "change_working_capital": 4824696960,
                    "free_cash_flow": 113790477801
                },
                {
                    "year": 4,
                    "revenue": 521067231680,
                    "ebit": 156861248736,
                    "nopat": 133332061425,
                    "depreciation": 15632016950,
                    "capex": 20842689267,
                    "change_working_capital": 5210672316,
                    "free_cash_flow": 122910716792
                },
                {
                    "year": 5,
                    "revenue": 562752610214,
                    "ebit": 169450147794,
                    "nopat": 144032625625,
                    "depreciation": 16882578306,
                    "capex": 22510104408,
                    "change_working_capital": 5627526102,
                    "free_cash_flow": 132777573421
                }
            ]
        },
        "risk": {
            "notes": {
                "volatility": "Moderate volatility within sector norms",
                "concentration": "High revenue concentration in iPhone (52%)",
                "geographic": "China represents 19% of revenue - geopolitical risk"
            }
        },
        "trends": {
            "return_1m": 0.052,
            "return_3m": 0.089,
            "return_6m": 0.145,
            "return_1y": 0.325,
            "annualized_volatility": 0.225,
            "avg_volume_30d": 58500000
        },
        "peer_metrics": [
            {
                "ticker": "AAPL",
                "market_cap": 2800000000000,
                "revenue_growth": 0.085,
                "ebitda_margin": 0.325,
                "pe_ratio": 29.5
            },
            {
                "ticker": "MSFT",
                "market_cap": 2650000000000,
                "revenue_growth": 0.112,
                "ebitda_margin": 0.415,
                "pe_ratio": 33.2
            },
            {
                "ticker": "GOOGL",
                "market_cap": 1750000000000,
                "revenue_growth": 0.095,
                "ebitda_margin": 0.295,
                "pe_ratio": 24.5
            },
            {
                "ticker": "AMZN",
                "market_cap": 1450000000000,
                "revenue_growth": 0.121,
                "ebitda_margin": 0.145,
                "pe_ratio": 65.2
            },
            {
                "ticker": "META",
                "market_cap": 825000000000,
                "revenue_growth": 0.165,
                "ebitda_margin": 0.385,
                "pe_ratio": 28.5
            }
        ]
    }
)


def generate_hardcoded_report():
    """Generate a test report using hardcoded data"""

    print("="*80)
    print("HARDCODED TEST REPORT GENERATOR")
    print("="*80)
    print(f"\nTicker: {HARDCODED_AAPL_PROFILE.ticker}")
    print(f"Company: {HARDCODED_AAPL_PROFILE.company_name}")
    print(f"Current Price: ${HARDCODED_AAPL_PROFILE.current_price}")
    print(f"Target Price: ${HARDCODED_AAPL_PROFILE.target_price}")
    print(f"Recommendation: {HARDCODED_AAPL_PROFILE.recommendation}")
    print("\n" + "="*80)

    # Initialize AI engine (use mock if Ollama not available)
    try:
        remote_url = os.getenv('ERT_LLM_BACKEND_URL')
        if remote_url:
            print(f"\n✅ Using Remote LLM: {remote_url}")
            from src.utils.remote_llm_client import RemoteLLMEngine
            ai_engine = RemoteLLMEngine(
                base_url=remote_url,
                api_key=os.getenv('ERT_LLM_API_KEY', 'test-key')
            )
        else:
            print("\n✅ Using Ollama (local)")
            ai_engine = OllamaEngine(
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                model=os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
            )
    except Exception as e:
        print(f"\n⚠️  AI engine error: {e}")
        print("Continuing with fallback content generation...")
        ai_engine = OllamaEngine()  # Will use fallback templates

    # Initialize report generator
    print("\n📊 Initializing report generator...")
    generator = StockReportGenerator(ai_engine)

    # Override the fetch method to return our hardcoded data
    original_fetch = generator.fetch_comprehensive_data
    def mock_fetch(ticker):
        print(f"\n🔄 Using hardcoded data for {ticker}")
        return HARDCODED_AAPL_PROFILE
    generator.fetch_comprehensive_data = mock_fetch

    # Generate report
    print("\n📝 Generating comprehensive report...\n")
    report = generator.generate_comprehensive_report("AAPL")

    # Save report
    output_dir = project_root / 'reports' / 'test_hardcoded'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"AAPL_HARDCODED_TEST_{timestamp}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ Report saved to: {report_path}")

    # Print stats
    word_count = len(report.split())
    estimated_pages = max(1, word_count // 250)

    print(f"\n📈 Report Statistics:")
    print(f"   - Word Count: {word_count:,}")
    print(f"   - Estimated Pages: {estimated_pages}")
    print(f"   - Sections: {report.count('##')}")
    print(f"   - Characters: {len(report):,}")

    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)

    return report_path


if __name__ == "__main__":
    generate_hardcoded_report()
