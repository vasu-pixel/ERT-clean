#!/usr/bin/env python3
"""
Test CVS report generation with all fixes applied
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_cvs_report_generation():
    """Test CVS report generation with fixes"""
    try:
        print("🧪 Testing CVS Report Generation with Fixes")
        print("=" * 50)

        # Import required modules
        from src.stock_report_generator import StockReportGenerator
        from src.utils.openai_engine import OpenAIEngine

        # Initialize AI engine
        print("🤖 Initializing AI engine...")
        ai_engine = OpenAIEngine()

        if not ai_engine.test_connection():
            print("❌ AI engine connection failed")
            return False

        # Initialize report generator
        print("📊 Initializing report generator...")
        generator = StockReportGenerator(ai_engine)

        # Test with CVS ticker
        ticker = "CVS"
        print(f"🏥 Generating report for {ticker}...")

        # Generate comprehensive report
        start_time = datetime.now()
        report_content = generator.generate_comprehensive_report(ticker)
        end_time = datetime.now()

        # Basic validation
        if not report_content or len(report_content.strip()) < 1000:
            print("❌ Report generation failed - content too short")
            return False

        # Check for error indicators
        error_indicators = [
            "temporarily unavailable",
            "Section generation",
            "$0.00",
            "technical difficulties"
        ]

        found_errors = []
        for indicator in error_indicators:
            if indicator in report_content:
                found_errors.append(indicator)

        if found_errors:
            print(f"⚠️ Found potential issues: {found_errors}")
        else:
            print("✅ No critical errors detected in report")

        # Report statistics
        word_count = len(report_content.split())
        generation_time = (end_time - start_time).total_seconds()

        print(f"\n📈 Report Statistics:")
        print(f"  - Word Count: {word_count:,}")
        print(f"  - Generation Time: {generation_time:.1f} seconds")
        print(f"  - Content Length: {len(report_content):,} characters")

        # Check for key sections
        required_sections = [
            "EXECUTIVE SUMMARY",
            "FINANCIAL ANALYSIS",
            "VALUATION",
            "INVESTMENT THESIS",
            "RISK ANALYSIS"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in report_content.upper():
                missing_sections.append(section)

        if missing_sections:
            print(f"⚠️ Missing sections: {missing_sections}")
        else:
            print("✅ All required sections present")

        # Save test report
        test_filename = f"reports/CVS_test_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(test_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n💾 Test report saved: {test_filename}")

        if word_count > 3000 and not found_errors and not missing_sections:
            print("\n🎉 CVS Report Generation Test: PASSED")
            return True
        else:
            print("\n⚠️ CVS Report Generation Test: NEEDS ATTENTION")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cvs_report_generation()
    exit(0 if success else 1)