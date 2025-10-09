"""
Remote LLM Client for connecting to Vast.ai backend
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx
import time

logger = logging.getLogger(__name__)

@dataclass
class RemoteLLMConfig:
    """Configuration for remote LLM backend"""
    base_url: str
    api_key: str
    timeout: int = 3600  # 1 hour timeout for long-running LLM generation
    max_retries: int = 3
    retry_delay: int = 2

class RemoteLLMClient:
    """Client for connecting to remote LLM backend on Vast.ai"""

    def __init__(self, config: RemoteLLMConfig):
        self.config = config
        self.session = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def health_check(self) -> bool:
        """Check if remote LLM backend is healthy"""
        try:
            if not self.session:
                raise ValueError("Client not initialized. Use async context manager.")

            response = await self.session.get(f"{self.config.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                return data.get("status") in ["healthy", "online"]
            return False

        except Exception as e:
            logger.error(f"LLM backend health check failed: {e}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get backend status information"""
        try:
            if not self.session:
                raise ValueError("Client not initialized. Use async context manager.")

            response = await self.session.get(f"{self.config.base_url}/status")
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Error getting backend status: {e}")
            return {"status": "error", "message": str(e)}

    async def generate_report(
        self,
        ticker: str,
        company_data: Dict[str, Any],
        financial_data: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate equity research report via remote backend"""

        payload = {
            "ticker": ticker,
            "company_data": company_data,
            "financial_data": financial_data,
            "market_data": market_data or {},
            "sections": sections
        }

        for attempt in range(self.config.max_retries):
            try:
                if not self.session:
                    raise ValueError("Client not initialized. Use async context manager.")

                logger.info(f"Generating report for {ticker} (attempt {attempt + 1})")

                response = await self.session.post(
                    f"{self.config.base_url}/generate_report",
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully generated report for {ticker}")
                    return result

                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited, waiting before retry...")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue

                else:
                    response.raise_for_status()

            except httpx.TimeoutException:
                logger.warning(f"Timeout generating report for {ticker} (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise

            except Exception as e:
                logger.error(f"Error generating report for {ticker}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise

        raise Exception(f"Failed to generate report after {self.config.max_retries} attempts")

class RemoteEquityResearchGenerator:
    """
    Remote equity research generator that uses Vast.ai backend
    Drop-in replacement for local EnhancedEquityResearchGenerator
    """

    def __init__(self):
        # Load configuration from environment
        self.base_url = os.getenv("ERT_LLM_BACKEND_URL", "http://localhost:8000")
        self.api_key = os.getenv("ERT_LLM_API_KEY", "ert-vast-api-key-2024")
        self.timeout = int(os.getenv("ERT_LLM_TIMEOUT", "3600"))  # 1 hour default

        self.config = RemoteLLMConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

        # Store data for report generation
        self.ticker = None
        self.company_data = {}
        self.financial_data = {}
        self.market_data = {}

        logger.info(f"Initialized remote LLM client for {self.base_url}")

    def fetch_comprehensive_data(self, ticker: str):
        """
        Fetch comprehensive data for the ticker
        This method maintains compatibility with the existing interface
        """
        self.ticker = ticker
        logger.info(f"Preparing to fetch data for {ticker}")

        # In the remote setup, data fetching is handled by the main application
        # This method just stores the ticker for later use
        return True

    def set_company_data(self, data: Dict[str, Any]):
        """Set company data for report generation"""
        self.company_data = data

    def set_financial_data(self, data: Dict[str, Any]):
        """Set financial data for report generation"""
        self.financial_data = data

    def set_market_data(self, data: Dict[str, Any]):
        """Set market data for report generation"""
        self.market_data = data

    def generate_comprehensive_report(self, ticker: str = None) -> str:
        """
        Generate comprehensive report using remote backend
        Returns markdown formatted report
        """
        if ticker:
            self.ticker = ticker

        if not self.ticker:
            raise ValueError("No ticker specified for report generation")

        try:
            # Run async report generation in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._generate_report_async())
                return result
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error generating report for {self.ticker}: {e}")
            return f"# Error Generating Report\n\nFailed to generate report for {self.ticker}: {str(e)}"

    async def _generate_report_async(self) -> str:
        """Async report generation implementation"""
        async with RemoteLLMClient(self.config) as client:
            # Check backend health
            if not await client.health_check():
                raise Exception("Remote LLM backend is not healthy")

            # Generate report
            result = await client.generate_report(
                ticker=self.ticker,
                company_data=self.company_data,
                financial_data=self.financial_data,
                market_data=self.market_data
            )

            if not result.get("success"):
                raise Exception("Report generation failed")

            # Convert sections to markdown
            sections = result.get("sections", {})
            metadata = result.get("metadata", {})

            # Build markdown report
            markdown_report = self._build_markdown_report(sections, metadata)

            return markdown_report

    def _build_markdown_report(self, sections: Dict[str, str], metadata: Dict[str, Any]) -> str:
        """Build markdown formatted report from sections"""
        report_lines = []

        # Title and header
        report_lines.append(f"# Equity Research Report: {self.ticker}")
        report_lines.append(f"**Generated:** {metadata.get('timestamp', 'N/A')}")
        report_lines.append(f"**Model:** {metadata.get('model', 'N/A')}")
        report_lines.append(f"**Generation Time:** {metadata.get('generation_time', 0):.2f}s")
        report_lines.append("")

        # Add sections
        section_titles = {
            "executive_summary": "Executive Summary",
            "financial_analysis": "Financial Analysis",
            "investment_thesis": "Investment Thesis",
            "valuation_analysis": "Valuation Analysis",
            "risk_analysis": "Risk Analysis"
        }

        for section_key, content in sections.items():
            title = section_titles.get(section_key, section_key.replace("_", " ").title())
            report_lines.append(f"## {title}")
            report_lines.append("")
            report_lines.append(content)
            report_lines.append("")

        # Footer
        report_lines.append("---")
        report_lines.append("*This report was generated using AI and should be used for informational purposes only.*")

        return "\n".join(report_lines)

    async def health_check(self) -> bool:
        """Check if remote backend is available"""
        try:
            async with RemoteLLMClient(self.config) as client:
                return await client.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Factory function for creating the appropriate generator
def create_equity_research_generator() -> RemoteEquityResearchGenerator:
    """Create equity research generator based on configuration"""
    return RemoteEquityResearchGenerator()