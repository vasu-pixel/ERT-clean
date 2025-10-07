#!/usr/bin/env python3
"""
LLM Backend Service for Vast.ai deployment
Serves Mistral:7b or Llama models for equity research report generation
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("ERT_API_KEY", "ert-vast-api-key-2024")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")  # Default to gpt-oss:20b for high-quality analysis

# Security
security = HTTPBearer()

@dataclass
class ReportRequest:
    ticker: str
    company_data: Dict[str, Any]
    financial_data: Dict[str, Any]
    market_data: Dict[str, Any] = None
    sections: List[str] = None

class ReportRequestModel(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    company_data: Dict[str, Any] = Field(..., description="Company information")
    financial_data: Dict[str, Any] = Field(..., description="Financial statements and metrics")
    market_data: Optional[Dict[str, Any]] = Field(None, description="Market and industry data")
    sections: Optional[List[str]] = Field(None, description="Specific sections to generate")

class ReportResponse(BaseModel):
    success: bool
    report_id: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]
    generation_time: float

class StatusResponse(BaseModel):
    status: str
    model: str
    uptime: float
    total_reports: int
    active_requests: int

# Global state
app_start_time = time.time()
total_reports_generated = 0
active_requests = 0

class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, host: str = OLLAMA_HOST, port: int = OLLAMA_PORT):
        self.base_url = f"http://{host}:{port}"
        self.model = MODEL_NAME

    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                # Check if our model is available
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]

                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                    # Try to pull the model
                    await self.pull_model()

                return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def pull_model(self) -> bool:
        """Pull the required model if not available"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                logger.info(f"Pulling model {self.model}...")
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model}
                )
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model {self.model}")
                    return True
                else:
                    logger.error(f"Failed to pull model: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False

    async def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text using Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama generation failed: {response.text}")
                    return f"Error: Failed to generate content (status {response.status_code})"

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"

# Initialize Ollama client
ollama_client = OllamaClient()

class EquityReportGenerator:
    """Enhanced equity research report generator using Mistral:7b or Llama models"""

    def __init__(self):
        self.ollama = ollama_client

    async def generate_executive_summary(self, ticker: str, company_data: Dict, financial_data: Dict) -> str:
        """Generate executive summary section"""
        system_prompt = """You are a senior equity research analyst. Generate a professional executive summary for an equity research report. Focus on key investment highlights, financial performance, and strategic positioning."""

        prompt = f"""
Company: {company_data.get('longName', ticker)}
Sector: {company_data.get('sector', 'N/A')}
Industry: {company_data.get('industry', 'N/A')}

Key Financial Metrics:
- Market Cap: ${financial_data.get('marketCap', 'N/A'):,} if isinstance(financial_data.get('marketCap'), (int, float)) else 'N/A'
- Revenue (TTM): ${financial_data.get('totalRevenue', 'N/A'):,} if isinstance(financial_data.get('totalRevenue'), (int, float)) else 'N/A'
- P/E Ratio: {financial_data.get('trailingPE', 'N/A')}
- Profit Margin: {financial_data.get('profitMargins', 'N/A'):.2%} if isinstance(financial_data.get('profitMargins'), (int, float)) else 'N/A'

Generate a concise 3-4 paragraph executive summary covering:
1. Company overview and market position
2. Financial performance highlights
3. Key investment thesis
4. Risk factors and outlook
"""

        return await self.ollama.generate_text(prompt, system_prompt)

    async def generate_financial_analysis(self, ticker: str, company_data: Dict, financial_data: Dict) -> str:
        """Generate financial analysis section"""
        system_prompt = """You are a financial analyst specializing in equity research. Provide detailed financial analysis focusing on profitability, growth, and financial health."""

        prompt = f"""
Analyze the financial performance of {company_data.get('longName', ticker)} ({ticker}):

Revenue and Profitability:
- Total Revenue: ${financial_data.get('totalRevenue', 'N/A')}
- Gross Profit: ${financial_data.get('grossProfits', 'N/A')}
- Operating Income: ${financial_data.get('operatingIncome', 'N/A')}
- Net Income: ${financial_data.get('netIncomeToCommon', 'N/A')}
- Profit Margins: {financial_data.get('profitMargins', 'N/A')}

Valuation Metrics:
- P/E Ratio: {financial_data.get('trailingPE', 'N/A')}
- P/B Ratio: {financial_data.get('priceToBook', 'N/A')}
- EV/Revenue: {financial_data.get('enterpriseToRevenue', 'N/A')}
- EV/EBITDA: {financial_data.get('enterpriseToEbitda', 'N/A')}

Provide comprehensive financial analysis covering:
1. Revenue trends and growth drivers
2. Profitability analysis and margin trends
3. Valuation assessment vs peers
4. Financial strength and liquidity
5. Key financial risks and opportunities
"""

        return await self.ollama.generate_text(prompt, system_prompt)

    async def generate_investment_thesis(self, ticker: str, company_data: Dict, financial_data: Dict) -> str:
        """Generate investment thesis section"""
        system_prompt = """You are an equity research analyst crafting an investment thesis. Focus on long-term value drivers, competitive advantages, and investment rationale."""

        prompt = f"""
Develop an investment thesis for {company_data.get('longName', ticker)} ({ticker}):

Company Profile:
- Sector: {company_data.get('sector', 'N/A')}
- Industry: {company_data.get('industry', 'N/A')}
- Business Summary: {company_data.get('longBusinessSummary', 'N/A')[:500]}...

Financial Strength:
- Return on Equity: {financial_data.get('returnOnEquity', 'N/A')}
- Return on Assets: {financial_data.get('returnOnAssets', 'N/A')}
- Debt to Equity: {financial_data.get('debtToEquity', 'N/A')}
- Current Ratio: {financial_data.get('currentRatio', 'N/A')}

Create a compelling investment thesis covering:
1. Core business strengths and competitive moat
2. Growth catalysts and market opportunities
3. Management quality and capital allocation
4. ESG considerations and sustainability
5. Long-term value creation potential
"""

        return await self.ollama.generate_text(prompt, system_prompt)

    async def generate_risk_analysis(self, ticker: str, company_data: Dict, financial_data: Dict) -> str:
        """Generate risk analysis section"""
        system_prompt = """You are a risk analyst specializing in equity investments. Identify and analyze key risks that could impact investment returns."""

        prompt = f"""
Conduct comprehensive risk analysis for {company_data.get('longName', ticker)} ({ticker}):

Business Context:
- Sector: {company_data.get('sector', 'N/A')}
- Industry: {company_data.get('industry', 'N/A')}
- Market Cap: ${financial_data.get('marketCap', 'N/A')}

Financial Indicators:
- Beta: {financial_data.get('beta', 'N/A')}
- Debt/Equity: {financial_data.get('debtToEquity', 'N/A')}
- Quick Ratio: {financial_data.get('quickRatio', 'N/A')}
- Book Value: {financial_data.get('bookValue', 'N/A')}

Analyze key risks including:
1. Business and operational risks
2. Financial and credit risks
3. Market and competitive risks
4. Regulatory and compliance risks
5. ESG and sustainability risks
6. Macroeconomic sensitivity
"""

        return await self.ollama.generate_text(prompt, system_prompt)

    async def generate_valuation_analysis(self, ticker: str, company_data: Dict, financial_data: Dict) -> str:
        """Generate valuation analysis section"""
        system_prompt = """You are a valuation expert. Provide detailed valuation analysis using multiple methodologies and frameworks."""

        prompt = f"""
Perform valuation analysis for {company_data.get('longName', ticker)} ({ticker}):

Current Valuation:
- Current Price: ${financial_data.get('currentPrice', 'N/A')}
- Market Cap: ${financial_data.get('marketCap', 'N/A')}
- Enterprise Value: ${financial_data.get('enterpriseValue', 'N/A')}

Valuation Multiples:
- P/E Ratio: {financial_data.get('trailingPE', 'N/A')}
- Forward P/E: {financial_data.get('forwardPE', 'N/A')}
- P/B Ratio: {financial_data.get('priceToBook', 'N/A')}
- P/S Ratio: {financial_data.get('priceToSalesTrailing12Months', 'N/A')}
- EV/Revenue: {financial_data.get('enterpriseToRevenue', 'N/A')}
- EV/EBITDA: {financial_data.get('enterpriseToEbitda', 'N/A')}

Provide comprehensive valuation analysis:
1. Multiple-based valuation vs industry peers
2. DCF analysis framework and assumptions
3. Asset-based valuation considerations
4. Scenario analysis (bull, base, bear cases)
5. Fair value range and price target
6. Valuation risks and sensitivities
"""

        return await self.ollama.generate_text(prompt, system_prompt)

    async def generate_comprehensive_report(self, request: ReportRequest) -> Dict[str, str]:
        """Generate complete equity research report"""
        logger.info(f"Starting report generation for {request.ticker}")

        sections = {}

        try:
            # Default sections if not specified
            if not request.sections:
                request.sections = [
                    "executive_summary",
                    "financial_analysis",
                    "investment_thesis",
                    "valuation_analysis",
                    "risk_analysis"
                ]

            # Generate each requested section
            for section in request.sections:
                logger.info(f"Generating {section} for {request.ticker}")

                if section == "executive_summary":
                    sections[section] = await self.generate_executive_summary(
                        request.ticker, request.company_data, request.financial_data
                    )
                elif section == "financial_analysis":
                    sections[section] = await self.generate_financial_analysis(
                        request.ticker, request.company_data, request.financial_data
                    )
                elif section == "investment_thesis":
                    sections[section] = await self.generate_investment_thesis(
                        request.ticker, request.company_data, request.financial_data
                    )
                elif section == "valuation_analysis":
                    sections[section] = await self.generate_valuation_analysis(
                        request.ticker, request.company_data, request.financial_data
                    )
                elif section == "risk_analysis":
                    sections[section] = await self.generate_risk_analysis(
                        request.ticker, request.company_data, request.financial_data
                    )
                else:
                    sections[section] = f"Section '{section}' not implemented"

                # Add delay between sections to prevent overwhelming the model
                await asyncio.sleep(1)

            logger.info(f"Completed report generation for {request.ticker}")
            return sections

        except Exception as e:
            logger.error(f"Error generating report for {request.ticker}: {e}")
            raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# Initialize generator
report_generator = EquityReportGenerator()

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting LLM Backend Service...")

    # Check Ollama health
    if await ollama_client.health_check():
        logger.info(f"Ollama is healthy with model {MODEL_NAME}")
    else:
        logger.warning("Ollama health check failed - some features may not work")

    yield

    # Shutdown
    logger.info("Shutting down LLM Backend Service...")

# Create FastAPI app
app = FastAPI(
    title="ERT LLM Backend",
    description="Mistral:7b/Llama backend for equity research report generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_healthy = await ollama_client.health_check()

    return {
        "status": "healthy" if ollama_healthy else "degraded",
        "model": MODEL_NAME,
        "ollama_status": "online" if ollama_healthy else "offline",
        "uptime": time.time() - app_start_time,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/status", response_model=StatusResponse)
async def get_status(api_key: str = Depends(verify_api_key)):
    """Get service status"""
    return StatusResponse(
        status="online",
        model=MODEL_NAME,
        uptime=time.time() - app_start_time,
        total_reports=total_reports_generated,
        active_requests=active_requests
    )

@app.post("/generate_report", response_model=ReportResponse)
async def generate_report(
    request: ReportRequestModel,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Generate equity research report"""
    global active_requests, total_reports_generated

    active_requests += 1
    start_time = time.time()
    report_id = f"{request.ticker}_{int(start_time)}"

    try:
        # Convert to internal format
        report_request = ReportRequest(
            ticker=request.ticker,
            company_data=request.company_data,
            financial_data=request.financial_data,
            market_data=request.market_data,
            sections=request.sections
        )

        # Generate report
        sections = await report_generator.generate_comprehensive_report(report_request)

        generation_time = time.time() - start_time
        total_reports_generated += 1

        # Prepare metadata
        metadata = {
            "generation_time": generation_time,
            "model": MODEL_NAME,
            "timestamp": datetime.utcnow().isoformat(),
            "sections_count": len(sections),
            "total_words": sum(len(content.split()) for content in sections.values())
        }

        logger.info(f"Generated report {report_id} in {generation_time:.2f}s")

        return ReportResponse(
            success=True,
            report_id=report_id,
            sections=sections,
            metadata=metadata,
            generation_time=generation_time
        )

    except Exception as e:
        logger.error(f"Error generating report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        active_requests -= 1

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ERT LLM Backend",
        "model": MODEL_NAME,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "generate_report": "/generate_report"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting LLM Backend on {host}:{port}")

    uvicorn.run(
        "llm_backend:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )