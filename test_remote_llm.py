#!/usr/bin/env python3
"""
Test script for Remote LLM integration
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.remote_llm_client import RemoteLLMClient, RemoteLLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_remote_llm():
    """Test remote LLM backend connectivity and functionality"""

    # Configuration
    config = RemoteLLMConfig(
        base_url=os.getenv("ERT_LLM_BACKEND_URL", "http://localhost:8000"),
        api_key=os.getenv("ERT_LLM_API_KEY", "ert-vast-api-key-2024"),
        timeout=30
    )

    print("=" * 60)
    print("ERT Remote LLM Integration Test")
    print("=" * 60)
    print(f"Backend URL: {config.base_url}")
    print(f"API Key: {config.api_key[:20]}...")
    print(f"Timeout: {config.timeout}s")
    print()

    async with RemoteLLMClient(config) as client:

        # Test 1: Health Check
        print("🔍 Testing health check...")
        try:
            healthy = await client.health_check()
            if healthy:
                print("✅ Health check: PASSED")
            else:
                print("❌ Health check: FAILED")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

        # Test 2: Status Check
        print("\n📊 Testing status endpoint...")
        try:
            status = await client.get_status()
            print(f"✅ Status: {status.get('status', 'unknown')}")
            print(f"   Model: {status.get('model', 'unknown')}")
            print(f"   Uptime: {status.get('uptime', 0):.2f}s")
        except Exception as e:
            print(f"❌ Status error: {e}")

        # Test 3: Report Generation
        print("\n📝 Testing report generation...")
        try:
            # Sample data
            sample_data = {
                "ticker": "AAPL",
                "company_data": {
                    "longName": "Apple Inc.",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "longBusinessSummary": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
                },
                "financial_data": {
                    "marketCap": 3000000000000,
                    "totalRevenue": 394000000000,
                    "grossProfits": 170000000000,
                    "profitMargins": 0.25,
                    "trailingPE": 29.5,
                    "priceToBook": 39.4,
                    "returnOnEquity": 1.56,
                    "currentPrice": 185.0
                },
                "market_data": {
                    "beta": 1.2,
                    "fiftyTwoWeekHigh": 198.23,
                    "fiftyTwoWeekLow": 164.08
                }
            }

            print("   Generating sample report for AAPL...")
            result = await client.generate_report(
                ticker=sample_data["ticker"],
                company_data=sample_data["company_data"],
                financial_data=sample_data["financial_data"],
                market_data=sample_data["market_data"],
                sections=["executive_summary"]  # Just test one section
            )

            if result.get("success"):
                print("✅ Report generation: PASSED")
                print(f"   Report ID: {result.get('report_id')}")
                print(f"   Generation time: {result.get('generation_time', 0):.2f}s")
                print(f"   Sections: {len(result.get('sections', {}))}")

                # Show first 200 chars of executive summary
                sections = result.get('sections', {})
                if 'executive_summary' in sections:
                    summary = sections['executive_summary'][:200] + "..."
                    print(f"   Preview: {summary}")
            else:
                print("❌ Report generation: FAILED")
                print(f"   Error: {result}")

        except Exception as e:
            print(f"❌ Report generation error: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

def main():
    """Main test function"""
    # Check environment variables
    backend_url = os.getenv("ERT_LLM_BACKEND_URL")
    api_key = os.getenv("ERT_LLM_API_KEY")

    if not backend_url:
        print("❌ ERT_LLM_BACKEND_URL environment variable not set")
        print("   Set it to your Vast.ai instance URL")
        print("   Example: export ERT_LLM_BACKEND_URL=https://12345.vast.ai")
        return False

    if not api_key:
        print("❌ ERT_LLM_API_KEY environment variable not set")
        print("   Generate one using: python vast_ai/generate_api_key.py")
        return False

    # Run async test
    asyncio.run(test_remote_llm())

if __name__ == "__main__":
    main()