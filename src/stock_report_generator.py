# src/stock_report_generator.py
import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import yfinance as yf
import math
from dataclasses import dataclass, asdict
import requests
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path for clean absolute imports
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.ai_engine import AIEngine
from src.utils.data_validator import DataValidator, DataIntegrityError, ValidationResult
from src.data_pipeline import DataOrchestrator
from src.retrieval import FileBackedDocumentStore, KeywordRetriever
from src.report.template_loader import load_template

#PDF Importer
try:
    from src.report.pdf_writer import InstitutionalPDFGenerator, convert_markdown_to_pdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import your existing modules (with error handling)
try:
    from src.fetch import *  # Your existing data fetching modules
except ImportError:
    print("Warning: Could not import src.fetch modules")

try:
    from src.analyze import *  # Your existing analysis modules  
except ImportError:
    print("Warning: Could not import src.analyze modules")

try:
    from src.utils import *  # Your existing utility functions
except ImportError:
    print("Warning: Could not import src.utils modules")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompanyProfile:
    """Enhanced company profile with comprehensive data"""
    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    current_price: float
    target_price: float
    recommendation: str
    financial_metrics: Dict
    competitors: List[str]
    esg_data: Dict
    risk_factors: List[str]
    deterministic: Dict[str, Any]

class StockReportGenerator:
    """
    Equity Research Generator using a given AI engine.
    Generates institutional-quality 35-50 page research reports
    """

    def __init__(self, ai_engine: AIEngine):
        """
        Initializes the StockReportGenerator.

        Args:
            ai_engine: An instance of a class that implements the AIEngine interface.
        """
        self.ai_engine = ai_engine
        self.report_date = datetime.now().strftime("%B %d, %Y")
        self.data_cache = {}
        self.config = self._load_config()
        deterministic_config = self._build_deterministic_config()
        self.data_orchestrator = DataOrchestrator(deterministic_config=deterministic_config)
        self.document_store = FileBackedDocumentStore()
        self.retriever = KeywordRetriever(self.document_store)
        self.guardrail_notes: List[str] = []
        self.data_validator = DataValidator()
        try:
            self.template_config = load_template()
        except FileNotFoundError as exc:
            logger.warning("Report template configuration missing: %s", exc)
            self.template_config = {"sections": [], "appendices": []}
        self.default_section_min_words = self.template_config.get("default_min_words", 500)
        self.section_sources: Dict[str, List[Dict[str, Any]]] = {}

        if not self.ai_engine.test_connection():
            logger.warning("AI engine connection test failed. Please check your setup.")

    def _load_config(self) -> Dict:
        """Load configuration with improved fallback handling"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                logger.info("Configuration loaded successfully")
                return config
        except Exception as e:
            logger.warning(f"Could not load config: {e}")

        # Default configuration
        default_config = {
            "ai_engine": "ollama",
            "ollama_engine": {
                "model": "llama3.1:8b",
                "base_url": "http://localhost:11434",
                "max_tokens": 4000,
                "temperature": 0.3
            },
            "report_settings": {
                "default_recommendation": "HOLD",
                "report_style": "institutional"
            },
            "analysis_parameters": {
                "peer_group_size": 5
            },
            "forecast_defaults": {
                "default_revenue_growth": 0.05,
                "default_eps_growth": 0.05
            },
            "valuation": {
                "risk_free_rate": 0.045,
                "equity_risk_premium": 0.055,
                "default_beta": 1.0,
                "sector_beta_overrides": {},
                "base_cost_of_debt": 0.04,
                "tax_rate": 0.25,
                "wacc_floor": 0.06,
                "wacc_ceiling": 0.14,
                "scenarios": {}
            }
        }

        # Save default config for future use
        try:
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info("Created default configuration file")
        except Exception as e:
            logger.warning(f"Could not save default config: {e}")

        return default_config

    def _build_deterministic_config(self) -> Dict[str, Dict[str, Any]]:
        """Assemble configuration payload for deterministic analytics."""
        forecast_cfg = self.config.get("forecast_defaults", {}) or {}
        valuation_cfg = self.config.get("valuation", {}) or {}
        return {
            "forecast": forecast_cfg,
            "valuation": valuation_cfg,
        }

    def _build_deterministic_config(self) -> Dict[str, Dict[str, Any]]:
        """Assemble configuration payload for deterministic analytics."""
        forecast_cfg = self.config.get("forecast_defaults", {}) or {}
        valuation_cfg = self.config.get("valuation", {}) or {}
        return {
            "forecast": forecast_cfg,
            "valuation": valuation_cfg,
        }

    def fetch_comprehensive_data(self, ticker: str) -> CompanyProfile:
        """Fetch normalized company data via the orchestration layer."""

        try:
            dataset = self.data_orchestrator.refresh_company_data(ticker)
            stock = yf.Ticker(ticker)
            info = stock.info

            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            financial_metrics = self._calculate_key_metrics(info, financials, balance_sheet, cash_flow)

            competitors = self._fetch_competitors(ticker, dataset.snapshot.sector)
            esg_data = self._fetch_esg_data(ticker)
            risk_factors = self._identify_risk_factors(info, ticker)

            current_price = dataset.snapshot.current_price or info.get('currentPrice')
            if current_price is None:
                try:
                    current_price = float(stock.history(period="1d").get('Close').iloc[-1])
                except Exception:
                    current_price = 0

            market_cap = dataset.snapshot.market_cap or info.get('marketCap') or 0

            company_profile = CompanyProfile(
                ticker=dataset.snapshot.ticker,
                company_name=dataset.snapshot.name,
                sector=dataset.snapshot.sector,
                industry=dataset.snapshot.industry,
                market_cap=market_cap,
                current_price=current_price or 0,
                target_price=0,
                recommendation="HOLD",
                financial_metrics=financial_metrics,
                competitors=competitors,
                esg_data=esg_data,
                risk_factors=risk_factors,
                deterministic=dataset.supplemental.get("deterministic", {}),
            )

            # Validate data integrity before proceeding
            validation_result = self.data_validator.validate_company_profile(company_profile)

            if not validation_result.is_valid:
                error_summary = "; ".join(validation_result.errors)
                logger.error(f"Data validation failed for {ticker}: {error_summary}")
                raise DataIntegrityError(f"Critical data issues for {ticker}: {error_summary}")

            if validation_result.warnings:
                warning_summary = "; ".join(validation_result.warnings)
                logger.warning(f"Data validation warnings for {ticker}: {warning_summary}")
                self.guardrail_notes.append(f"Data quality warnings: {warning_summary}")

            logger.info(f"✅ Data validation passed for {ticker} (score: {validation_result.score:.1f}/100)")

            self.data_cache[dataset.snapshot.ticker] = dataset
            return company_profile

        except Exception as e:
            logger.error(f"Error fetching comprehensive data for {ticker}: {e}")
            raise

    def export_model_to_excel(self, ticker: str, output_path: str, force_refresh: bool = True) -> str:
        """Export deterministic model outputs to a multi-sheet Excel workbook."""

        dataset = self.data_orchestrator.refresh_company_data(ticker, force=force_refresh)
        deterministic = dataset.supplemental.get("deterministic")

        if not deterministic:
            raise ValueError("Deterministic outputs unavailable; refresh company data first.")

        valuation = deterministic.get("valuation", {})
        fcf_projection = deterministic.get("fcf_projection", {})
        forecast = deterministic.get("forecast", {})
        peer_metrics = deterministic.get("peer_metrics", [])
        history = deterministic.get("history", {})
        segments = dataset.supplemental.get("segment_forecast", [])
        statements = dataset.supplemental.get("three_statement_model", {})

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path) as writer:
            schedule = fcf_projection.get("schedule")
            if schedule:
                pd.DataFrame(schedule).to_excel(writer, sheet_name="FCF Projection", index=False)

            scenarios = valuation.get("scenarios")
            if scenarios:
                scenario_rows = [
                    {"scenario": name, "intrinsic_value": value}
                    for name, value in scenarios.items()
                ]
                pd.DataFrame(scenario_rows).to_excel(writer, sheet_name="Valuation Scenarios", index=False)

            assumptions = valuation.get("assumptions")
            if assumptions:
                pd.DataFrame([assumptions]).to_excel(writer, sheet_name="Valuation Assumptions", index=False)

            scenario_inputs = valuation.get("scenario_parameters")
            if scenario_inputs:
                rows = []
                for name, params in scenario_inputs.items():
                    row = {"scenario": name}
                    row.update(params)
                    rows.append(row)
                if rows:
                    pd.DataFrame(rows).to_excel(writer, sheet_name="Scenario Inputs", index=False)

            multiples = valuation.get("multiples_summary")
            if multiples:
                pd.DataFrame([multiples]).to_excel(writer, sheet_name="Valuation Multiples", index=False)

            sensitivity = valuation.get("sensitivity") or {}
            if sensitivity.get("dcf_matrix"):
                tg_values = sensitivity.get("terminal_growth_values", [])
                wacc_values = sensitivity.get("wacc_values", [])
                df = pd.DataFrame(
                    sensitivity["dcf_matrix"],
                    index=[f"TG {val:.2%}" if isinstance(val, float) else val for val in tg_values],
                    columns=[f"WACC {val:.2%}" if isinstance(val, float) else val for val in wacc_values],
                )
                df.to_excel(writer, sheet_name="DCF Sensitivity")

            if forecast:
                pd.DataFrame(forecast).to_excel(writer, sheet_name="Forecast", index=True)

            if peer_metrics:
                pd.DataFrame(peer_metrics).to_excel(writer, sheet_name="Peer Metrics", index=False)

            if history:
                hist_df = pd.DataFrame(history)
                if not hist_df.empty:
                    hist_df.to_excel(writer, sheet_name="Historical Trends", index=False)

            if segments:
                pd.DataFrame(segments).to_excel(writer, sheet_name="Segments", index=False)

            for sheet, table in statements.items():
                df = pd.DataFrame(table)
                if df.empty:
                    continue
                sheet_name = sheet.replace("_", " ").title()
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        logger.info("Exported financial model to %s", output_path)
        return str(output_path)


    def _calculate_key_metrics(self, info: Dict, financials: pd.DataFrame,
                             balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> Dict:
        """Calculate key financial metrics"""
        try:
            metrics = {
                # Valuation Metrics
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'peg_ratio': info.get('pegRatio', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 'N/A'),
                'ev_to_revenue': info.get('enterpriseToRevenue', 'N/A'),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 'N/A'),

                # Profitability Metrics
                'gross_margin': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 'N/A',
                'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 'N/A',
                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 'N/A',
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 'N/A',
                'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 'N/A',
                'roic': self._calculate_roic(info),

                # Growth Metrics
                'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 'N/A',
                'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 'N/A',

                # Financial Health
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'current_ratio': info.get('currentRatio', 'N/A'),
                'quick_ratio': info.get('quickRatio', 'N/A'),

                # Cash Flow
                'operating_cash_flow': info.get('operatingCashflow', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'fcf_yield': self._calculate_fcf_yield(info),

                # Other
                'beta': info.get('beta', 'N/A'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'payout_ratio': info.get('payoutRatio', 'N/A')
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            return {}

    def _calculate_roic(self, info: Dict) -> float:
        """Calculate Return on Invested Capital"""
        try:
            total_assets = info.get('totalAssets', 0)
            total_liabilities = info.get('totalLiabilities', 0)
            invested_capital = total_assets - total_liabilities

            ebit = info.get('ebitda', 0) - info.get('depreciation', 0)
            tax_rate = 0.25  # Assume 25% tax rate
            nopat = ebit * (1 - tax_rate)

            if invested_capital > 0:
                return (nopat / invested_capital) * 100
            return 'N/A'
        except:
            return 'N/A'

    def _calculate_fcf_yield(self, info: Dict) -> float:
        """Calculate Free Cash Flow Yield"""
        try:
            fcf = info.get('freeCashflow', 0)
            market_cap = info.get('marketCap', 0)

            if market_cap > 0:
                return (fcf / market_cap) * 100
            return 'N/A'
        except:
            return 'N/A'

    def _fetch_competitors(self, ticker: str, sector: str) -> List[str]:
        """Fetch competitor information"""
        # Use your existing competitor identification logic
        # or integrate with external APIs
        competitor_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON']
        }

        competitors = competitor_mapping.get(sector, [])
        # Remove the target ticker from competitors list
        return [comp for comp in competitors if comp != ticker.upper()]

    def _fetch_esg_data(self, ticker: str) -> Dict:
        """Fetch ESG data (placeholder - integrate with ESG data providers)"""
        return {
            'esg_score': 'N/A',
            'environmental_score': 'N/A',
            'social_score': 'N/A',
            'governance_score': 'N/A',
            'controversy_level': 'Low',
            'sustainability_initiatives': []
        }

    def _identify_risk_factors(self, info: Dict, ticker: str) -> List[str]:
        """Identify key risk factors"""
        risks = [
            'Market volatility and economic uncertainty',
            'Competitive pressure from industry peers',
            'Regulatory changes affecting operations',
            'Interest rate and inflation risks'
        ]

        # Add sector-specific risks
        sector = info.get('sector', '')
        if 'Technology' in sector:
            risks.extend([
                'Rapid technological obsolescence',
                'Cybersecurity threats',
                'Data privacy regulations'
            ])
        elif 'Financial' in sector:
            risks.extend([
                'Credit risk and loan defaults',
                'Regulatory capital requirements',
                'Interest rate sensitivity'
            ])

        return risks

    def _get_retrieval_context(self, ticker: str, query: str) -> Tuple[str, List[Any]]:
        documents = self.retriever.retrieve(ticker, query, top_k=5)
        if not documents:
            return "", []

        def accession_key(metadata: Dict[str, Any]) -> tuple[int, int]:
            accession = metadata.get('accession') if isinstance(metadata, dict) else None
            if accession:
                parts = str(accession).split('-')
                if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                    return (int(parts[1]), int(parts[2]))
            return (0, 0)

        documents = sorted(documents, key=lambda doc: accession_key(doc.metadata), reverse=True)

        formatted = []
        for doc in documents:
            snippet = doc.text.strip().replace('\n', ' ')
            if len(snippet) > 800:
                snippet = snippet[:800] + '...'
            source = doc.metadata.get('source', 'Filing')
            formatted.append(
                f"[source:{doc.chunk_id} | {source}] {snippet}"
            )

        return '\n\n'.join(formatted), documents

    def _augment_prompt_with_context(
        self,
        ticker: str,
        base_prompt: str,
        topic: str,
        section_key: Optional[str] = None
    ) -> str:
        context, documents = self._get_retrieval_context(ticker, topic)
        if section_key and documents:
            existing = self.section_sources.setdefault(section_key, [])
            for doc in documents:
                existing.append(
                    {
                        "chunk_id": doc.chunk_id,
                        "source": doc.metadata.get('source', 'Filing'),
                        "metadata": doc.metadata,
                    }
                )
        if not context:
            return base_prompt

        return (
            f"{base_prompt}\n\nUse the following primary-source context when crafting the analysis."
            f"\n\n{context}\n\nCite key figures or statements from the context using the provided [source:chunk] tags."
        )

    def _apply_recommendation_guardrails(
        self, company_profile: CompanyProfile, recommendation: str, target_price: float
    ) -> Tuple[str, float, List[str]]:
        notes: List[str] = []

        current_price = company_profile.current_price or 0
        if current_price > 0 and target_price > 0:
            upside_ratio = target_price / current_price
            if upside_ratio > 2.0:
                notes.append(
                    f"Target price implied upside ({upside_ratio:.1f}x) exceeds guardrail; scaled to 2x."
                )
                target_price = round(current_price * 2, 2)
            elif upside_ratio < 0.5:
                notes.append(
                    f"Target price implies more than 50% downside; clipped to 0.5x guardrail."
                )
                target_price = round(current_price * 0.5, 2)

        if recommendation not in {"BUY", "HOLD", "SELL"}:
            notes.append(f"Unknown recommendation '{recommendation}' adjusted to HOLD.")
            recommendation = "HOLD"

        deterministic = company_profile.deterministic or {}
        raw_dcf = (
            deterministic.get("valuation", {}).get("dcf_value")
            if isinstance(deterministic, dict)
            else None
        )
        dcf_value = None
        try:
            if raw_dcf is not None:
                dcf_value = float(raw_dcf)
        except (TypeError, ValueError):
            dcf_value = None

        if dcf_value and target_price > 0:
            try:
                deviation = abs(target_price - dcf_value) / abs(dcf_value)
                if deviation > 0.3:
                    notes.append(
                        f"AI target (${target_price:.2f}) deviates >30% from DCF (${dcf_value:.2f})."
                    )
            except ZeroDivisionError:
                pass

        return recommendation, target_price, notes

    def _format_deterministic_summary(self, deterministic: Dict[str, Any]) -> str:
        if not deterministic:
            return ""

        forecast = deterministic.get("forecast", {})
        valuation = deterministic.get("valuation", {})
        risk = deterministic.get("risk", {})

        def fmt_currency(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                value = float(value)
            except (TypeError, ValueError):
                return str(value)

            if abs(value) >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            if abs(value) >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            if abs(value) >= 1_000:
                return f"${value/1_000:.2f}K"
            return f"${value:.2f}"

        def fmt_number(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)

        def fmt_percent(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                return f"{float(value)*100:.1f}%"
            except (TypeError, ValueError):
                return str(value)

        revenue = forecast.get("revenue", {})
        eps = forecast.get("eps", {})

        lines = ["Deterministic analytics summary:"]

        if revenue:
            lines.append(
                "- Revenue forecast: TTM {} → Next year {} (Year two {})".format(
                    fmt_currency(revenue.get("ttm")),
                    fmt_currency(revenue.get("next_year")),
                    fmt_currency(revenue.get("year_two")),
                )
            )
        if eps:
            lines.append(
                "- EPS forecast: TTM {} → Next year {} (Year two {})".format(
                    fmt_number(eps.get("ttm")),
                    fmt_number(eps.get("next_year")),
                    fmt_number(eps.get("year_two")),
                )
            )

        dcf_value = valuation.get("dcf_value")
        if dcf_value:
            lines.append(f"- Base DCF intrinsic value: {fmt_currency(dcf_value)}")

        scenarios = valuation.get("scenarios", {})
        if scenarios:
            scenario_parts = [f"{name.title()}: {fmt_currency(value)}" for name, value in scenarios.items() if value is not None]
            if scenario_parts:
                lines.append("- Scenario valuation (per share): " + ", ".join(scenario_parts))

        multiples = valuation.get("multiples_summary", {})
        if multiples:
            parts = []
            for key in sorted(multiples):
                value = multiples[key]
                if value is None:
                    continue
                key_label = key.replace('_', ' ').upper()
                parts.append(f"{key_label}: {fmt_number(value)}")
            if parts:
                lines.append("- Key multiples: " + ", ".join(parts))

        assumptions = valuation.get("assumptions", {})
        assumption_parts = []
        if "wacc" in assumptions:
            assumption_parts.append(f"WACC {assumptions['wacc']:.2%}")
        if "projection_growth" in assumptions:
            assumption_parts.append(f"Growth {assumptions['projection_growth']:.2%}")
        if "terminal_growth" in assumptions:
            assumption_parts.append(f"Terminal {assumptions['terminal_growth']:.2%}")
        if assumption_parts:
            lines.append("- DCF assumptions: " + ", ".join(assumption_parts))

        trends = deterministic.get("trends", {})
        if trends:
            trend_parts = []
            for key in ["return_1m", "return_3m", "return_6m", "return_1y"]:
                if key in trends and trends[key] is not None:
                    trend_parts.append(f"{key.replace('return_', '').upper()} {fmt_percent(trends[key])}")
            if trend_parts:
                lines.append("- Price performance: " + ", ".join(trend_parts))
            if trends.get("annualized_volatility") is not None:
                lines.append(f"- Annualized volatility: {fmt_percent(trends['annualized_volatility'])}")

        risk_notes = risk.get("notes", {})
        if risk_notes:
            summary = "; ".join(risk_notes.values())
            if summary:
                lines.append("- Risk alerts: " + summary)

        return "\n".join(lines)

    def _generate_template_section(
        self,
        company_profile: CompanyProfile,
        section_cfg: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        key = section_cfg.get("key", "generic_section")
        title = section_cfg.get("title", key.replace('_', ' ').title())
        description = section_cfg.get("description", "Provide a detailed analysis.")
        min_words = section_cfg.get("min_word_count", self.default_section_min_words)
        include_deterministic = section_cfg.get("include_deterministic", False)
        require_citations = section_cfg.get("require_citations", False)
        context_query = section_cfg.get("context_query", title)

        deterministic_summary = ""
        if include_deterministic:
            deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)

        base_prompt = (
            f"Write a comprehensive section titled '{title}' for {company_profile.company_name} ({company_profile.ticker}). "
            f"Purpose: {description} Provide an institutional-quality narrative with clear sub-headings, bullet lists, tables where appropriate, and quantified insights. "
            f"Target a minimum of {min_words} words."
        )
        if key == "executive_summary":
            base_prompt += (
                f"\n\nUse the established investment stance: rating {company_profile.recommendation} with price target ${company_profile.target_price:.2f}. "
                "Do not change the recommendation or target price; reinforce them consistently."
            )
        else:
            base_prompt += (
                f"\n\nEnsure all commentary aligns with the overall recommendation ({company_profile.recommendation}) and price target ${company_profile.target_price:.2f}."
            )
        if deterministic_summary:
            base_prompt += (
                f"\n\nReference deterministic analytics summary below when grounding forecasts and valuation commentary:\n{deterministic_summary}"
            )
        if require_citations:
            base_prompt += (
                "\n\nEvery major claim must cite a source using the format [source:chunk_id]."
            )
        base_prompt += (
            "\n\nStructure Guidance:\n"
            "1. Begin with a concise overview of why this section matters.\n"
            "2. Provide detailed analysis with quantitative support, comparing against peers where possible.\n"
            "3. Avoid duplicating sentences or bullet points from earlier sections; only introduce genuinely new insights.\n"
            "4. Close with actionable takeaways or implications."
        )

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            base_prompt,
            context_query,
            section_key=key
        )

        # Robust section generation with comprehensive error handling
        content = self._generate_section_with_fallback(key, prompt, company_profile, min_words)
        attempts = self._last_generation_attempts  # Track attempts from fallback method
        word_count = len(content.split())

        if require_citations and "[source:" not in content:
            note = f"Section '{title}' lacks required citations."
            self.guardrail_notes.append(note)

        if word_count < min_words:
            self.guardrail_notes.append(
                f"Section '{title}' word count {word_count} below target {min_words}."
            )

        section_meta = {
            "key": key,
            "title": title,
            "word_count": word_count,
            "min_word_count": min_words,
            "attempts": attempts,
            "require_citations": require_citations,
            "has_citations": "[source:" in content,
        }

        return content, section_meta

    def _generate_section_with_fallback(
        self, section_key: str, prompt: str, company_profile: CompanyProfile, min_words: int
    ) -> str:
        """Generate section content with comprehensive fallback mechanisms"""
        self._last_generation_attempts = 1
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                # Test AI engine connectivity before attempting generation
                if not self.ai_engine.test_connection():
                    logger.warning(f"AI engine not available for {section_key} (attempt {attempt + 1})")
                    if attempt == max_attempts - 1:
                        return self._generate_comprehensive_fallback_section(section_key, company_profile)
                    continue

                # Attempt content generation
                content = self.ai_engine.generate_section(section_key, prompt)

                # Validate generated content
                if self._validate_generated_content(content, section_key, min_words):
                    logger.info(f"✅ Successfully generated {section_key} section on attempt {attempt + 1}")
                    self._last_generation_attempts = attempt + 1

                    # Try to expand if below word count
                    if len(content.split()) < min_words and attempt < max_attempts - 1:
                        expanded_content = self._try_expand_content(
                            content, section_key, company_profile, min_words
                        )
                        if expanded_content and len(expanded_content.split()) > len(content.split()):
                            content = expanded_content

                    return content
                else:
                    logger.warning(f"Generated content for {section_key} failed validation (attempt {attempt + 1})")

            except Exception as e:
                logger.warning(f"Section generation attempt {attempt + 1} failed for {section_key}: {e}")

            # Brief pause before retry
            if attempt < max_attempts - 1:
                time.sleep(2)

        # All attempts failed - generate comprehensive fallback
        logger.error(f"All attempts failed for {section_key} - using fallback content")
        self._last_generation_attempts = max_attempts
        return self._generate_comprehensive_fallback_section(section_key, company_profile)

    def _validate_generated_content(self, content: str, section_key: str, min_words: int) -> bool:
        """Validate that generated content meets quality standards"""
        if not content or len(content.strip()) < 50:
            return False

        # Check for error messages in content
        error_indicators = [
            "temporarily unavailable",
            "error generating",
            "service unavailable",
            "connection failed",
            "please try again",
            "unable to generate"
        ]

        content_lower = content.lower()
        for indicator in error_indicators:
            if indicator in content_lower:
                return False

        # Check minimum word count (relaxed for validation)
        word_count = len(content.split())
        min_acceptable = max(100, min_words * 0.3)  # At least 30% of target or 100 words

        return word_count >= min_acceptable

    def _try_expand_content(
        self, content: str, section_key: str, company_profile: CompanyProfile, min_words: int
    ) -> Optional[str]:
        """Try to expand content to meet word count requirements"""
        try:
            current_words = len(content.split())
            deficit = min_words - current_words

            if deficit <= 0:
                return content

            expansion_prompt = (
                f"Expand the following {section_key.replace('_', ' ')} section for {company_profile.company_name}. "
                f"Add {deficit} more words with additional insights, specific data points, and detailed analysis. "
                "Do not repeat existing content - provide new information and perspectives.\n\n"
                f"EXISTING CONTENT:\n{content}\n\n"
                "ADDITIONAL CONTENT (continue seamlessly):"
            )

            additional_content = self.ai_engine.generate_section(f"{section_key}_expansion", expansion_prompt)

            if additional_content and len(additional_content.split()) > 50:
                return f"{content}\n\n{additional_content}"

            return content

        except Exception as e:
            logger.warning(f"Content expansion failed for {section_key}: {e}")
            return content

    def _generate_comprehensive_fallback_section(self, section_key: str, company_profile: CompanyProfile) -> str:
        """Generate comprehensive fallback content when AI generation completely fails"""
        section_title = section_key.upper().replace('_', ' ')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Comprehensive fallback templates with real data integration
        fallback_templates = {
            'executive_summary': f"""# EXECUTIVE SUMMARY

**Investment Recommendation:** {company_profile.recommendation} (Generated via Fallback System)
**Price Target:** ${company_profile.target_price:.2f}
**Current Price:** ${company_profile.current_price:.2f}
**Upside/Downside:** {((company_profile.target_price - company_profile.current_price) / company_profile.current_price * 100):+.1f}%

## Company Overview
{company_profile.company_name} ({company_profile.ticker}) operates in the {company_profile.sector} sector, specifically within the {company_profile.industry} industry. With a market capitalization of ${company_profile.market_cap/1e9:.1f} billion, the company represents a {"large-cap" if company_profile.market_cap > 10e9 else "mid-cap" if company_profile.market_cap > 2e9 else "small-cap"} investment opportunity.

## Key Financial Metrics
Based on available financial data:
- **P/E Ratio:** {company_profile.financial_metrics.get('pe_ratio', 'N/A')}
- **Revenue Growth:** {company_profile.financial_metrics.get('revenue_growth', 'N/A')}%
- **Profit Margin:** {company_profile.financial_metrics.get('profit_margin', 'N/A')}%
- **ROE:** {company_profile.financial_metrics.get('roe', 'N/A')}%

## Investment Rationale
The {company_profile.recommendation} recommendation reflects current market conditions and fundamental analysis. Key considerations include the company's competitive position within the {company_profile.sector} sector and its financial performance metrics.

## Primary Risk Factors
{"; ".join(company_profile.risk_factors[:3]) if company_profile.risk_factors else "Standard market and operational risks apply."}

**Note:** This analysis was generated using fallback systems due to AI service limitations. For complete analysis, please ensure AI services are properly configured.

*Generated: {timestamp}*""",

            'financial_analysis': f"""# FINANCIAL ANALYSIS

## Revenue and Profitability Assessment

**Company Overview:** {company_profile.company_name} operates in the {company_profile.sector} sector with current market capitalization of ${company_profile.market_cap/1e9:.1f} billion.

## Key Financial Metrics Analysis

### Valuation Metrics
- **P/E Ratio:** {company_profile.financial_metrics.get('pe_ratio', 'N/A')} - {"Above sector average" if str(company_profile.financial_metrics.get('pe_ratio', 'N/A')).replace('.','').isdigit() and float(company_profile.financial_metrics.get('pe_ratio', 0)) > 15 else "Within reasonable range"}
- **Price-to-Book:** {company_profile.financial_metrics.get('price_to_book', 'N/A')}
- **Price-to-Sales:** {company_profile.financial_metrics.get('price_to_sales', 'N/A')}

### Profitability Analysis
- **Gross Margin:** {company_profile.financial_metrics.get('gross_margin', 'N/A')}%
- **Operating Margin:** {company_profile.financial_metrics.get('operating_margin', 'N/A')}%
- **Net Profit Margin:** {company_profile.financial_metrics.get('profit_margin', 'N/A')}%

### Returns Analysis
- **Return on Equity (ROE):** {company_profile.financial_metrics.get('roe', 'N/A')}%
- **Return on Assets (ROA):** {company_profile.financial_metrics.get('roa', 'N/A')}%

### Financial Health Indicators
- **Current Ratio:** {company_profile.financial_metrics.get('current_ratio', 'N/A')}
- **Debt-to-Equity:** {company_profile.financial_metrics.get('debt_to_equity', 'N/A')}
- **Free Cash Flow:** ${company_profile.financial_metrics.get('free_cash_flow', 0)/1e9:.2f}B

## Competitive Positioning
The company competes with major players including {', '.join(company_profile.competitors[:3]) if company_profile.competitors else 'industry peers'} in the {company_profile.sector} space.

## Financial Outlook
Based on available metrics and industry positioning, the company demonstrates {"strong" if company_profile.financial_metrics.get('roe', 0) and float(str(company_profile.financial_metrics.get('roe', 0)).replace('N/A', '0')) > 15 else "moderate"} financial fundamentals.

**Analysis Limitation:** Complete financial statement analysis unavailable due to service limitations.
*Generated: {timestamp}*""",

            'valuation_analysis': f"""# VALUATION ANALYSIS

## Current Valuation Overview

**Current Share Price:** ${company_profile.current_price:.2f}
**Market Capitalization:** ${company_profile.market_cap/1e9:.1f} billion
**Price Target:** ${company_profile.target_price:.2f}
**Implied Return:** {((company_profile.target_price - company_profile.current_price) / company_profile.current_price * 100):+.1f}%

## Valuation Methodology

### Relative Valuation Metrics
Current trading multiples for {company_profile.company_name}:

- **P/E Ratio:** {company_profile.financial_metrics.get('pe_ratio', 'N/A')}x
- **EV/Revenue:** {company_profile.financial_metrics.get('ev_to_revenue', 'N/A')}x
- **EV/EBITDA:** {company_profile.financial_metrics.get('ev_to_ebitda', 'N/A')}x
- **Price/Book Value:** {company_profile.financial_metrics.get('price_to_book', 'N/A')}x

### Peer Comparison Context
Within the {company_profile.sector} sector, key comparable companies include {', '.join(company_profile.competitors[:3]) if company_profile.competitors else 'industry leaders'}.

### Valuation Assessment
Based on current multiples and sector positioning, {company_profile.company_name} appears to be trading at {"a premium" if str(company_profile.financial_metrics.get('pe_ratio', 'N/A')).replace('.','').isdigit() and float(company_profile.financial_metrics.get('pe_ratio', 0)) > 20 else "reasonable levels"} to fundamental value.

## Target Price Methodology

The ${company_profile.target_price:.2f} price target reflects:
- Current market positioning within {company_profile.sector}
- Fundamental valuation metrics
- Sector risk-adjusted returns
- {"Conservative" if abs((company_profile.target_price - company_profile.current_price) / company_profile.current_price) < 0.15 else "Moderate"} target approach

## Risk Factors to Valuation
Primary risks to valuation include {'; '.join(company_profile.risk_factors[:2]) if company_profile.risk_factors else 'market volatility and sector-specific challenges'}.

**Valuation Limitation:** Detailed DCF modeling unavailable due to service limitations.
*Generated: {timestamp}*"""
        }

        # Default fallback for any section not specifically defined
        default_fallback = f"""# {section_title}

## Analysis Overview for {company_profile.company_name}

**Company:** {company_profile.company_name} ({company_profile.ticker})
**Sector:** {company_profile.sector}
**Current Price:** ${company_profile.current_price:.2f}
**Market Cap:** ${company_profile.market_cap/1e9:.1f}B

## {section_title} Assessment

This section would provide detailed analysis of {section_title.lower()} for {company_profile.company_name}. Due to technical limitations with AI content generation services, detailed analysis is temporarily unavailable.

## Key Considerations

Based on available data:
- The company operates in the {company_profile.sector} sector
- Current market positioning reflects {company_profile.recommendation} investment stance
- Key metrics support price target of ${company_profile.target_price:.2f}

## Financial Context
- **Revenue Growth:** {company_profile.financial_metrics.get('revenue_growth', 'N/A')}%
- **Profitability:** {company_profile.financial_metrics.get('profit_margin', 'N/A')}% net margin
- **Returns:** {company_profile.financial_metrics.get('roe', 'N/A')}% ROE

## Conclusion

{company_profile.company_name} presents a {company_profile.recommendation} investment opportunity based on current fundamental and market analysis. Detailed {section_title.lower()} would require complete AI service functionality.

**Service Note:** This analysis was generated using fallback systems. For comprehensive {section_title.lower()}, please ensure AI generation services are properly configured and operational.

*Generated: {timestamp}*"""

        return fallback_templates.get(section_key, default_fallback)

    def _build_appendix_sections(self, company_profile: CompanyProfile) -> List[Tuple[str, str]]:
        appendices = []
        deterministic = company_profile.deterministic or {}
        for appendix in self.template_config.get("appendices", []):
            key = appendix.get("key")
            title = appendix.get("title", key.title() if key else "Appendix")
            if key == "deterministic_tables":
                content = self._build_deterministic_appendix(deterministic)
            elif key == "peer_benchmark_table":
                content = self._build_peer_benchmark_appendix(deterministic)
            elif key == "retrieval_sources":
                content = self._build_sources_appendix()
            else:
                content = "Additional data forthcoming."

            if content:
                appendices.append((title, content))
        return appendices

    def _build_deterministic_appendix(self, deterministic: Dict[str, Any]) -> str:
        if not deterministic:
            return "Deterministic analytics not available."

        forecast = deterministic.get("forecast", {})
        valuation = deterministic.get("valuation", {})
        risk = deterministic.get("risk", {})
        trends = deterministic.get("trends", {})
        fcf_projection = deterministic.get("fcf_projection", {})
        sensitivity = valuation.get("sensitivity", {})

        def fmt_currency(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                value = float(value)
            except (TypeError, ValueError):
                return str(value)
            if abs(value) >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            if abs(value) >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            if abs(value) >= 1_000:
                return f"${value/1_000:.2f}K"
            return f"${value:.2f}"

        def fmt_number(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)

        def fmt_percent(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                return f"{float(value)*100:.1f}%"
            except (TypeError, ValueError):
                return str(value)

        lines = ["### Forecast Summary", "| Metric | Value |", "|---|---|"]
        revenue = forecast.get("revenue", {})
        eps = forecast.get("eps", {})
        if revenue:
            for label, value in revenue.items():
                lines.append(f"| Revenue {label.replace('_', ' ').title()} | {fmt_currency(value)}|")
        if eps:
            for label, value in eps.items():
                lines.append(f"| EPS {label.replace('_', ' ').title()} | {fmt_number(value)}|")

        lines.extend(["\n### Valuation Summary", "| Metric | Value |", "|---|---|"])
        dcf_value = valuation.get("dcf_value")
        if dcf_value is not None:
            lines.append(f"| DCF Value per Share | {fmt_currency(dcf_value)}|")
        for metric, value in (valuation.get("multiples_summary") or {}).items():
            lines.append(f"| {metric.replace('_', ' ').title()} | {fmt_number(value)}|")
        scenarios = valuation.get("scenarios") or {}
        if scenarios:
            lines.append("\n### Scenario Valuation")
            lines.append("| Scenario | Value |")
            lines.append("|---|---|")
            for name, value in scenarios.items():
                lines.append(f"| {name.title()} | {fmt_currency(value)}|")

        if sensitivity:
            wacc_vals = sensitivity.get("wacc_values", [])
            tg_vals = sensitivity.get("terminal_growth_values", [])
            matrix = sensitivity.get("dcf_matrix", [])
            if wacc_vals and tg_vals and matrix:
                header = "| Terminal \\ WACC | " + " | ".join(f"{val*100:.1f}%" for val in wacc_vals) + " |"
                lines.append("\n### DCF Sensitivity (per share)")
                lines.append(header)
                lines.append("|" + "---|" * (len(wacc_vals) + 1))
                for tg, row in zip(tg_vals, matrix):
                    row_values = " | ".join(fmt_currency(val) if val is not None else "N/A" for val in row)
                    lines.append(f"| {tg*100:.1f}% | {row_values} |")

        lines.extend(["\n### Risk Indicators", "| Category | Indicator |", "|---|---|"])
        for category, note in (risk.get("notes") or {}).items():
            lines.append(f"| {category.title()} | {note}|")

        if trends:
            lines.extend(["\n### Price & Liquidity Trends", "| Metric | Value |", "|---|---|"])
            for metric, value in trends.items():
                if metric.startswith("return"):
                    lines.append(f"| {metric.replace('_', ' ').title()} | {fmt_percent(value)}|")
                elif "volatility" in metric:
                    lines.append(f"| {metric.replace('_', ' ').title()} | {fmt_percent(value)}|")
                elif "volume" in metric:
                    lines.append(f"| {metric.replace('_', ' ').title()} | {fmt_number(value)}|")
                else:
                    lines.append(f"| {metric.replace('_', ' ').title()} | {fmt_number(value)}|")

        if fcf_projection:
            assumptions = fcf_projection.get("assumptions", {})
            schedule = fcf_projection.get("schedule", [])
            lines.append("\n### FCF Forecast Assumptions")
            lines.append("| Assumption | Value |")
            lines.append("|---|---|")
            for key, value in assumptions.items():
                if key.endswith("_pct") or "margin" in key or key.endswith("growth") or key.endswith("rate"):
                    lines.append(f"| {key.replace('_', ' ').title()} | {fmt_percent(value)}|")
                else:
                    lines.append(f"| {key.replace('_', ' ').title()} | {fmt_currency(value) if 'revenue' in key else fmt_number(value)}|")

            if schedule:
                lines.append("\n### Five-Year Free Cash Flow Forecast")
                lines.append("| Year | Revenue | EBIT | NOPAT | Depreciation | Capex | Δ Working Capital | FCF |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for row in schedule:
                    lines.append(
                        "| Year {year} | {rev} | {ebit} | {nopat} | {dep} | {capex} | {wc} | {fcf} |".format(
                            year=row.get('year'),
                            rev=fmt_currency(row.get('revenue')),
                            ebit=fmt_currency(row.get('ebit')),
                            nopat=fmt_currency(row.get('nopat')),
                            dep=fmt_currency(row.get('depreciation')),
                            capex=fmt_currency(row.get('capex')),
                            wc=fmt_currency(row.get('change_working_capital')),
                            fcf=fmt_currency(row.get('free_cash_flow')),
                        )
                    )

        return "\n".join(lines)

    def _build_peer_benchmark_appendix(self, deterministic: Dict[str, Any]) -> str:
        peer_metrics = deterministic.get("peer_metrics") if deterministic else None
        if not peer_metrics:
            return "Peer benchmarking data not available."

        def fmt_currency(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                value = float(value)
            except (TypeError, ValueError):
                return str(value)
            if abs(value) >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            if abs(value) >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            return f"${value:.2f}"

        def fmt_percent(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                return f"{float(value)*100:.1f}%"
            except (TypeError, ValueError):
                return str(value)

        def fmt_number(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)

        lines = ["| Ticker | Market Cap | Revenue Growth | EBITDA Margin | P/E |", "|---|---|---|---|---|"]
        for peer in peer_metrics:
            lines.append(
                "| {ticker} | {market_cap} | {revenue_growth} | {ebitda_margin} | {pe_ratio} |".format(
                    ticker=peer.get("ticker"),
                    market_cap=fmt_currency(peer.get("market_cap")),
                    revenue_growth=fmt_percent(peer.get("revenue_growth")),
                    ebitda_margin=fmt_percent(peer.get("ebitda_margin")),
                    pe_ratio=fmt_number(peer.get("pe_ratio")),
                )
            )
        return "\n".join(lines)

    def _build_sources_appendix(self) -> str:
        if not self.section_sources:
            return "No retrieval sources recorded."

        unique_sources = {}
        for section_key, sources in self.section_sources.items():
            for source in sources:
                chunk_id = source.get("chunk_id")
                if chunk_id and chunk_id not in unique_sources:
                    unique_sources[chunk_id] = source

        if not unique_sources:
            return "No retrieval sources recorded."

        lines = ["| Chunk ID | Source | Details |", "|---|---|---|"]
        for chunk_id, meta in unique_sources.items():
            source = meta.get("source", "Filing")
            details = ", ".join(f"{k}: {v}" for k, v in meta.get("metadata", {}).items())
            lines.append(f"| {chunk_id} | {source} | {details} |")

        return "\n".join(lines)

    def _build_segment_appendix(self, company_profile: CompanyProfile) -> str:
        dataset = self.data_cache.get(company_profile.ticker)
        if not dataset:
            return ""
        segments = dataset.supplemental.get('segment_forecast') if hasattr(dataset, 'supplemental') else None
        if not segments:
            return ""

        lines = ["| Segment | Year | Revenue | EBIT |", "|---|---|---|---|"]
        for segment in segments:
            name = segment.get('name', 'Segment')
            revenue_series = segment.get('revenue', []) or []
            ebit_series = segment.get('ebit', []) or []
            for idx, revenue in enumerate(revenue_series, start=1):
                ebit = ebit_series[idx-1] if idx-1 < len(ebit_series) else None
                lines.append(
                    f"| {name} | Year {idx} | {self._format_currency(revenue)} | {self._format_currency(ebit)} |"
                )
        return "\n".join(lines)

    def _build_three_statement_appendix(self, company_profile: CompanyProfile) -> str:
        dataset = self.data_cache.get(company_profile.ticker)
        if not dataset:
            return ""
        model = dataset.supplemental.get('three_statement_model') if hasattr(dataset, 'supplemental') else None
        if not model:
            return ""

        income = model.get('income_statement', [])
        balance = model.get('balance_sheet', [])
        cash_flow = model.get('cash_flow', [])

        lines = [
            "### Income Statement Forecast",
            "| Year | Revenue | EBIT | Net Income | Dividends |",
            "|---|---|---|---|---|"
        ]
        for row in income:
            lines.append(
                f"| {row.get('year')} | {self._format_currency(row.get('revenue'))} | "
                f"{self._format_currency(row.get('ebit'))} | {self._format_currency(row.get('net_income'))} | "
                f"{self._format_currency(row.get('dividends'))} |"
            )

        lines.extend(["\n### Balance Sheet Forecast", "| Year | Cash | Debt | Equity |", "|---|---|---|---|"])
        for row in balance:
            lines.append(
                f"| {row.get('year')} | {self._format_currency(row.get('cash'))} | "
                f"{self._format_currency(row.get('total_debt'))} | {self._format_currency(row.get('total_equity'))} |"
            )

        lines.extend(["\n### Cash Flow Forecast", "| Year | Operating Cash Flow | Capex | FCF | Dividends |", "|---|---|---|---|---|"])
        for row in cash_flow:
            lines.append(
                f"| {row.get('year')} | {self._format_currency(row.get('operating_cash_flow'))} | "
                f"{self._format_currency(row.get('capex'))} | {self._format_currency(row.get('free_cash_flow'))} | "
                f"{self._format_currency(row.get('dividends'))} |"
            )

        return "\n".join(lines)

    def _format_currency(self, value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"${value/1_000:.2f}K"
        return f"${value:.2f}"

    def generate_executive_summary_with_ai(self, company_profile: CompanyProfile) -> str:
        """Generate executive summary using the AI engine"""

        prompt = f"""
        Generate a comprehensive executive summary for an institutional equity research report on {company_profile.company_name} ({company_profile.ticker}).

        Company Information:
        - Sector: {company_profile.sector}
        - Industry: {company_profile.industry}
        - Market Cap: ${company_profile.market_cap/1e9:.1f}B
        - Current Price: ${company_profile.current_price:.2f}
        - P/E Ratio: {company_profile.financial_metrics.get('pe_ratio', 'N/A')}
        - Revenue Growth: {company_profile.financial_metrics.get('revenue_growth', 'N/A')}%

        Please include:
        1. Investment recommendation (BUY/HOLD/SELL) with rationale
        2. Price target with upside/downside potential
        3. 3-4 key investment highlights
        4. Financial overview with key metrics
        5. Primary growth catalysts and risk factors

        Format as a professional institutional research executive summary.
        """

        deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)
        if deterministic_summary:
            prompt += f"\n\nDeterministic analytics reference:\n{deterministic_summary}"

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            prompt,
            "executive summary",
        )

        try:
            response = self.ai_engine.generate_section(
                'executive_summary',
                prompt
            )

            return response

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return self._fallback_executive_summary(company_profile)

    def generate_market_research_with_ai(self, company_profile: CompanyProfile) -> str:
        """Generate comprehensive market research using the AI engine"""

        prompt = f"""
        Generate comprehensive market research analysis for {company_profile.company_name} covering these 6 dimensions:

        1. INDUSTRY ANALYSIS - Market size, growth trends, competitive dynamics
        2. COMPETITIVE LANDSCAPE - Market positioning, competitor analysis
        3. MARKET POSITIONING & CUSTOMER ANALYSIS - Target markets, value proposition
        4. REGULATORY ANALYSIS - Policy environment, compliance requirements
        5. ESG RESEARCH - Environmental, social, governance performance
        6. MACROECONOMIC IMPACT - Economic sensitivity, geopolitical factors

        Company Context:
        - Sector: {company_profile.sector}
        - Industry: {company_profile.industry}
        - Competitors: {', '.join(company_profile.competitors[:5])}
        - Market Cap: ${company_profile.market_cap/1e9:.1f}B

        Provide detailed, institutional-quality analysis for each dimension with specific insights and data points.
        """

        deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)
        if deterministic_summary:
            prompt += f"\n\nDeterministic analytics reference:\n{deterministic_summary}"

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            prompt,
            "market analysis",
        )

        try:
            response = self.ai_engine.generate_section(
                'market_research',
                prompt
            )

            return response

        except Exception as e:
            logger.error(f"Error generating market research: {e}")
            return self._fallback_market_research(company_profile)

    def generate_financial_analysis_with_ai(self, company_profile: CompanyProfile) -> str:
        """Generate detailed financial analysis using the AI engine"""

        metrics = company_profile.financial_metrics

        prompt = f"""
        Generate detailed financial analysis for {company_profile.company_name} covering:

        1. REVENUE ANALYSIS - Growth trends, segment breakdown, quality assessment
        2. PROFITABILITY METRICS - Margins, ROE, ROA, efficiency ratios, peer comparison
        3. BALANCE SHEET STRENGTH - Liquidity, capital structure, working capital
        4. CASH FLOW ANALYSIS - Operating CF, FCF, capital allocation strategy

        Key Financial Metrics:
        - P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
        - Gross Margin: {metrics.get('gross_margin', 'N/A')}%
        - Operating Margin: {metrics.get('operating_margin', 'N/A')}%
        - ROE: {metrics.get('roe', 'N/A')}%
        - ROA: {metrics.get('roa', 'N/A')}%
        - Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}
        - Current Ratio: {metrics.get('current_ratio', 'N/A')}
        - FCF: ${metrics.get('free_cash_flow', 0)/1e9:.2f}B

        Provide institutional-quality financial analysis with specific insights on strengths, weaknesses, and trends.
        """

        deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)
        if deterministic_summary:
            prompt += f"\n\nDeterministic analytics reference:\n{deterministic_summary}"

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            prompt,
            "financial performance",
        )

        try:
            response = self.ai_engine.generate_section(
                'financial_analysis',
                prompt
            )

            return response

        except Exception as e:
            logger.error(f"Error generating financial analysis: {e}")
            return self._fallback_financial_analysis(company_profile)

    def generate_valuation_analysis_with_ai(self, company_profile: CompanyProfile) -> str:
        """Generate comprehensive valuation analysis using the AI engine"""

        metrics = company_profile.financial_metrics

        prompt = f"""
        Generate comprehensive valuation analysis for {company_profile.company_name} including:

        1. DCF VALUATION MODEL - Revenue assumptions, margin projections, terminal value, WACC calculation
        2. COMPARABLE COMPANY ANALYSIS - Peer multiples, relative valuation
        3. VALUATION SUMMARY - Target price methodology, sensitivity analysis

        Current Valuation Metrics:
        - Current Price: ${company_profile.current_price:.2f}
        - P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
        - P/B Ratio: {metrics.get('price_to_book', 'N/A')}
        - P/S Ratio: {metrics.get('price_to_sales', 'N/A')}
        - EV/Revenue: {metrics.get('ev_to_revenue', 'N/A')}
        - EV/EBITDA: {metrics.get('ev_to_ebitda', 'N/A')}
        - FCF Yield: {metrics.get('fcf_yield', 'N/A')}%

        Competitors: {', '.join(company_profile.competitors[:5])}

        Provide detailed valuation methodology with specific price target and upside/downside scenarios.
        """

        deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)
        if deterministic_summary:
            prompt += f"\n\nDeterministic analytics reference:\n{deterministic_summary}"

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            prompt,
            "valuation",
        )

        try:
            response = self.ai_engine.generate_section(
                'valuation_analysis',
                prompt
            )

            return response

        except Exception as e:
            logger.error(f"Error generating valuation analysis: {e}")
            return self._fallback_valuation_analysis(company_profile)

    def generate_investment_thesis_with_ai(self, company_profile: CompanyProfile) -> str:
        """Generate investment thesis using the AI engine"""

        prompt = f"""
        Generate comprehensive investment thesis for {company_profile.company_name} including:

        1. BULL CASE - Growth drivers, competitive advantages, catalyst timeline
        2. BEAR CASE - Key risks, competitive threats, downside scenarios

        Company Context:
        - Current Price: ${company_profile.current_price:.2f}
        - Market Cap: ${company_profile.market_cap/1e9:.1f}B
        - Sector: {company_profile.sector}
        - Key Competitors: {', '.join(company_profile.competitors[:3])}
        - Revenue Growth: {company_profile.financial_metrics.get('revenue_growth', 'N/A')}%
        - ROE: {company_profile.financial_metrics.get('roe', 'N/A')}%

        Provide balanced analysis with specific catalysts, timeframes, and risk scenarios.
        """

        deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)
        if deterministic_summary:
            prompt += f"\n\nDeterministic analytics reference:\n{deterministic_summary}"

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            prompt,
            "investment thesis",
        )

        try:
            response = self.ai_engine.generate_section(
                'investment_thesis',
                prompt
            )

            return response

        except Exception as e:
            logger.error(f"Error generating investment thesis: {e}")
            return self._fallback_investment_thesis(company_profile)

    def generate_risk_analysis_with_ai(self, company_profile: CompanyProfile) -> str:
        """Generate comprehensive risk analysis using the AI engine"""

        prompt = f"""
        Generate comprehensive risk analysis for {company_profile.company_name} covering:

        1. BUSINESS RISKS - Competitive threats, market risks, operational vulnerabilities
        2. FINANCIAL RISKS - Leverage, liquidity, credit risks, market exposure
        3. REGULATORY RISKS - Policy changes, compliance, antitrust concerns

        Company Risk Profile:
        - Sector: {company_profile.sector} (sector-specific risks)
        - Debt/Equity: {company_profile.financial_metrics.get('debt_to_equity', 'N/A')}
        - Current Ratio: {company_profile.financial_metrics.get('current_ratio', 'N/A')}
        - Beta: {company_profile.financial_metrics.get('beta', 'N/A')}
        - Key Risk Factors: {', '.join(company_profile.risk_factors[:5])}

        Provide detailed risk assessment with impact analysis and mitigation strategies.
        """

        deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)
        if deterministic_summary:
            prompt += f"\n\nDeterministic analytics reference:\n{deterministic_summary}"

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            prompt,
            "risk factors",
        )

        try:
            response = self.ai_engine.generate_section(
                'risk_analysis',
                prompt
            )

            return response

        except Exception as e:
            logger.error(f"Error generating risk analysis: {e}")
            return self._fallback_risk_analysis(company_profile)

    def determine_recommendation_and_target(
        self, company_profile: CompanyProfile
    ) -> Tuple[str, float, List[str]]:
        """Determine investment recommendation and price target using the AI engine"""

        metrics = company_profile.financial_metrics

        prompt = f"""
        Based on comprehensive analysis, provide investment recommendation (BUY/HOLD/SELL) and 12-month price target for {company_profile.company_name}.

        Current Metrics:
        - Current Price: ${company_profile.current_price:.2f}
        - P/E: {metrics.get('pe_ratio', 'N/A')}
        - Revenue Growth: {metrics.get('revenue_growth', 'N/A')}%
        - ROE: {metrics.get('roe', 'N/A')}%
        - Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}
        - FCF Yield: {metrics.get('fcf_yield', 'N/A')}%

        Consider:
        1. Valuation relative to peers and historical levels
        2. Growth prospects and competitive position
        3. Financial health and balance sheet strength
        4. Risk factors and market conditions

        Provide your response in this exact format:
        RECOMMENDATION: [BUY/HOLD/SELL]
        TARGET_PRICE: [numerical value only, no $ symbol]
        RATIONALE: [brief explanation]
        """

        deterministic_summary = self._format_deterministic_summary(company_profile.deterministic)
        if deterministic_summary:
            prompt += f"\n\nDeterministic analytics reference:\n{deterministic_summary}"

        prompt = self._augment_prompt_with_context(
            company_profile.ticker,
            prompt,
            "investment recommendation",
        )

        try:
            response = self.ai_engine.call_ai(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )

            # Parse response
            lines = response.split('\n')
            recommendation = 'HOLD'
            target_price = company_profile.current_price * 1.05

            for line in lines:
                line = line.strip()
                if line.startswith('RECOMMENDATION:'):
                    recommendation = line.split(':', 1)[1].strip()
                elif line.startswith('TARGET_PRICE:'):
                    price_str = line.split(':', 1)[1].strip()
                    try:
                        # Clean price string
                        price_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
                        target_price = float(price_str)
                        # Sanity check
                        if target_price <= 0 or target_price > 10000:
                            target_price = company_profile.current_price * 1.05
                    except (ValueError, TypeError):
                        target_price = company_profile.current_price * 1.05

            recommendation, target_price, notes = self._apply_recommendation_guardrails(
                company_profile, recommendation, target_price
            )
            return recommendation, target_price, notes

        except Exception as e:
            logger.error(f"Error determining recommendation: {e}")
            fallback_target = company_profile.current_price * 1.05
            _, fallback_target, notes = self._apply_recommendation_guardrails(
                company_profile, 'HOLD', fallback_target
            )
            return 'HOLD', fallback_target, notes

    def generate_comprehensive_report(self, ticker: str) -> str:
        """Generate complete comprehensive equity research report using the AI engine"""

        logger.info(f"Generating comprehensive research report for {ticker}")

        # Initialize variables at the start
        full_report = ""
        self.guardrail_notes = []
        self.section_sources = {}

        try:
            # Fetch comprehensive company data
            company_profile = self.fetch_comprehensive_data(ticker)

            # Determine recommendation and target price
            recommendation, target_price, guardrail_notes = self.determine_recommendation_and_target(company_profile)
            company_profile.recommendation = recommendation
            company_profile.target_price = target_price
            self.guardrail_notes = guardrail_notes

            # Calculate upside potential
            upside_potential = ((target_price - company_profile.current_price) / company_profile.current_price) * 100

            section_configs = self.template_config.get("sections", [])
            section_outputs: List[Tuple[int, str, str]] = []
            section_metadata: List[Dict[str, Any]] = []

            if not section_configs:
                logger.warning("Template configuration missing sections; falling back to legacy structure")
                legacy_sections = [
                    ("Executive Summary", self.generate_executive_summary_with_ai(company_profile)),
                    ("Comprehensive Market Research", self.generate_market_research_with_ai(company_profile)),
                    ("Detailed Financial Analysis", self.generate_financial_analysis_with_ai(company_profile)),
                    ("Comprehensive Valuation Analysis", self.generate_valuation_analysis_with_ai(company_profile)),
                    ("Investment Thesis", self.generate_investment_thesis_with_ai(company_profile)),
                    ("Comprehensive Risk Analysis", self.generate_risk_analysis_with_ai(company_profile)),
                ]
                for idx, (title, content) in enumerate(legacy_sections, start=1):
                    section_outputs.append((idx, title, content))
                    section_metadata.append({
                        "key": title.lower().replace(' ', '_'),
                        "title": title,
                        "word_count": len(content.split()),
                        "min_word_count": 0,
                        "attempts": 1,
                        "require_citations": False,
                        "has_citations": "[source:" in content,
                    })
            else:
                for idx, section_cfg in enumerate(section_configs, start=1):
                    title = section_cfg.get("title", section_cfg.get("key", f"Section {idx}"))
                    logger.info("Generating section %s...", title)
                    content, meta = self._generate_template_section(company_profile, section_cfg)
                    meta["index"] = idx
                    section_metadata.append(meta)
                    section_outputs.append((idx, title, content))

            # Compile appendices
            appendices = self._build_appendix_sections(company_profile)
            appendices.insert(0, ("Key Financial Metrics", self._format_financial_metrics_table(company_profile.financial_metrics)))

            guardrail_block = ""
            if self.guardrail_notes:
                guardrail_lines = "\n".join(f"- {note}" for note in self.guardrail_notes)
                guardrail_block = f"\n**Guardrail Alerts:**\n{guardrail_lines}\n\n"

            report_header = f"""
# COMPREHENSIVE EQUITY RESEARCH REPORT
## {company_profile.company_name} ({company_profile.ticker})
### {recommendation} Rating | Price Target: ${target_price:.2f} | Upside: {upside_potential:.1f}%
### Report Date: {self.report_date}

---

**INVESTMENT SUMMARY**
- **Current Price:** ${company_profile.current_price:.2f}
- **Target Price:** ${target_price:.2f}
- **Recommendation:** {recommendation}
- **Sector:** {company_profile.sector}
- **Market Cap:** ${company_profile.market_cap/1e9:.1f}B

---
{guardrail_block}
"""

            report_parts = [report_header]

            for idx, title, content in section_outputs:
                report_parts.append(f"## SECTION {idx}: {title}\n{content}\n\n---")

            if appendices:
                report_parts.append("## APPENDICES")
                for appendix_title, appendix_content in appendices:
                    report_parts.append(f"### {appendix_title}\n{appendix_content}\n\n---")

            report_parts.append(
                "**DISCLAIMER:** This research report is generated using an AI model and is for informational purposes only.\n"
                "This is not investment advice. Past performance does not guarantee future results.\n"
                "Please consult with qualified financial advisors before making investment decisions.\n\n"
                f"**Data Sources:** Yahoo Finance, Public Company Filings, Market Data Providers, AI Analysis\n"
                f"**Report Generated:** {self.report_date}\n"
                f"**AI Engine:** {self.ai_engine.get_model_info().get('model', 'N/A')}"
            )

            full_report = "\n\n".join(report_parts)

            # Final report validation before saving
            report_metadata = {
                'ticker': ticker,
                'recommendation': recommendation,
                'target_price': target_price,
                'section_count': len(section_outputs),
                'appendix_count': len(appendices)
            }

            final_validation = self.data_validator.validate_complete_report(full_report, report_metadata)

            if not final_validation.is_valid:
                error_summary = "; ".join(final_validation.errors)
                logger.error(f"Final report validation failed for {ticker}: {error_summary}")
                self.guardrail_notes.extend(final_validation.errors)

            if final_validation.warnings:
                warning_summary = "; ".join(final_validation.warnings)
                logger.warning(f"Final report validation warnings for {ticker}: {warning_summary}")
                self.guardrail_notes.extend(final_validation.warnings)

            logger.info(f"Final report validation score: {final_validation.score:.1f}/100")

            total_words = len(full_report.split())
            estimated_pages = max(1, math.ceil(total_words / 250))
            full_report += f"\n\n**Total Word Count:** {total_words} words\n**Estimated Page Count:** {estimated_pages} pages\n"

            # Add validation summary to report if there were issues
            if self.guardrail_notes:
                validation_summary = "\n".join(f"- {note}" for note in self.guardrail_notes[-5:])  # Last 5 notes
                full_report += f"\n\n**Validation Notes:**\n{validation_summary}\n"

            # Save report to reports directory
            self._save_report(
                ticker,
                full_report,
                recommendation,
                target_price,
                section_metadata=section_metadata,
                appendices=[title for title, _ in appendices],
                total_words=total_words,
                estimated_pages=estimated_pages,
            )

            logger.info(f"Comprehensive report generation completed for {ticker}")
            return full_report

        except Exception as e:
            logger.error(f"Error generating comprehensive report for {ticker}: {e}")
            # Return a fallback report instead of raising exception
            if not full_report:
                full_report = f"""
    # ERROR REPORT FOR {ticker}

    An error occurred while generating the comprehensive report: {str(e)}

    Please check:
    1. Internet connection for data fetching
    2. AI engine is configured and running correctly
    3. Ticker symbol accuracy

    Timestamp: {self.report_date}
    """
            return full_report


    def _format_financial_metrics_table(self, metrics: Dict) -> str:
        """Format financial metrics into a table"""
        table = """
| Metric | Value |
|--------|-------|
"""
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, (int, float)) and value != 'N/A':
                if value > 1000000:
                    formatted_value = f"${value/1e9:.2f}B"
                elif value > 1000:
                    formatted_value = f"${value/1e6:.2f}M"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            table += f"| {formatted_key} | {formatted_value} |\n"

        return table

    def _save_report(
        self,
        ticker: str,
        report: str,
        recommendation: str,
        target_price: float,
        section_metadata: Optional[List[Dict[str, Any]]] = None,
        appendices: Optional[List[str]] = None,
        total_words: Optional[int] = None,
        estimated_pages: Optional[int] = None,
    ):
        """Enhanced save report method"""
        try:
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"reports/{ticker}_{recommendation}_{target_price:.0f}_{timestamp}"

            # Save markdown report
            md_filename = f"{base_filename}.md"
            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"Markdown report saved to {md_filename}")

            # Save JSON summary for tracking
            deduped_sources: Dict[str, List[Dict[str, Any]]] = {}
            for section_key, items in (self.section_sources or {}).items():
                unique: Dict[str, Dict[str, Any]] = {}
                for item in items:
                    chunk_id = item.get("chunk_id")
                    if chunk_id and chunk_id not in unique:
                        unique[chunk_id] = item
                deduped_sources[section_key] = list(unique.values())

            summary = {
                'ticker': ticker,
                'recommendation': recommendation,
                'target_price': target_price,
                'report_date': self.report_date,
                'markdown_file': md_filename,
                'word_count': total_words or len(report.split()),
                'estimated_pages': estimated_pages,
                'ai_engine': self.ai_engine.get_model_info().get('model', 'N/A'),
                'guardrail_notes': self.guardrail_notes,
                'sections': section_metadata or [],
                'appendices': appendices or [],
                'retrieval_sources': deduped_sources,
                'deterministic': self.data_cache.get(ticker).supplemental.get('deterministic') if self.data_cache.get(ticker) else None,
            }

            summary_filename = f"{base_filename}_summary.json"
            with open(summary_filename, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving report: {e}")


    # Fallback methods (for when AI fails)
    def _fallback_executive_summary(self, company_profile: CompanyProfile) -> str:
        """Fallback executive summary when AI fails"""
        return f"""
# EXECUTIVE SUMMARY

**Investment Recommendation:** HOLD
**Price Target:** ${company_profile.current_price * 1.05:.2f}
**Current Price:** ${company_profile.current_price:.2f}
**Upside Potential:** 5.0%

## Key Investment Highlights

• **Market Position**: {company_profile.company_name} operates in the {company_profile.industry} sector with established market presence
• **Financial Metrics**: Current P/E ratio of {company_profile.financial_metrics.get('pe_ratio', 'N/A')} reflects market valuation
• **Growth Profile**: Revenue growth trends support moderate expansion outlook
• **Balance Sheet**: Financial position provides operational stability

## Risk Factors
- Market volatility and competitive pressure
- Sector-specific regulatory changes
- Economic uncertainty impact on operations

*Note: This fallback summary was generated due to AI service unavailability.*
"""

    def _fallback_market_research(self, company_profile: CompanyProfile) -> str:
        """Fallback market research when AI fails"""
        return f"""
# COMPREHENSIVE MARKET RESEARCH

## Industry Analysis
The {company_profile.industry} industry continues to evolve with changing market dynamics and competitive pressures.

## Competitive Landscape
Key competitors include: {', '.join(company_profile.competitors[:5])}

## Market Positioning
{company_profile.company_name} maintains its position through operational focus and market presence.

## Regulatory Environment
Standard industry regulations apply with ongoing compliance requirements.

## ESG Considerations
Environmental, social, and governance factors continue to influence operations.

## Macroeconomic Impact
General economic conditions affect sector performance and outlook.

*Note: This fallback analysis was generated due to AI service unavailability.*
"""

    def _fallback_financial_analysis(self, company_profile: CompanyProfile) -> str:
        """Fallback financial analysis when AI fails"""
        metrics = company_profile.financial_metrics
        return f"""
# DETAILED FINANCIAL ANALYSIS

## Revenue Performance
Current revenue trends reflect business operations and market conditions.

## Profitability Analysis
- Gross Margin: {metrics.get('gross_margin', 'N/A')}%
- Operating Margin: {metrics.get('operating_margin', 'N/A')}%
- ROE: {metrics.get('roe', 'N/A')}%

## Balance Sheet Strength
Financial position demonstrates stability with current ratio of {metrics.get('current_ratio', 'N/A')}.

## Cash Flow Analysis
Operating cash flow supports business operations and capital requirements.

*Note: This fallback analysis was generated due to AI service unavailability.*
"""

    def _fallback_valuation_analysis(self, company_profile: CompanyProfile) -> str:
        """Fallback valuation analysis when AI fails"""
        return f"""
# COMPREHENSIVE VALUATION ANALYSIS

## Current Valuation Metrics
- P/E Ratio: {company_profile.financial_metrics.get('pe_ratio', 'N/A')}
- Price-to-Book: {company_profile.financial_metrics.get('price_to_book', 'N/A')}
- EV/Revenue: {company_profile.financial_metrics.get('ev_to_revenue', 'N/A')}

## Peer Comparison
Valuation relative to sector peers suggests fair value trading range.

## Target Price Methodology
Conservative approach yields target price of ${company_profile.current_price * 1.05:.2f}.

*Note: This fallback analysis was generated due to AI service unavailability.*
"""

    def _fallback_investment_thesis(self, company_profile: CompanyProfile) -> str:
        """Fallback investment thesis when AI fails"""
        return f"""
# INVESTMENT THESIS

## Bull Case
- Established market position provides competitive advantages
- Financial stability supports continued operations
- Potential for market expansion and growth

## Bear Case
- Competitive pressure may impact market share
- Economic uncertainty affects sector performance
- Regulatory changes could increase compliance costs

## Investment Merit
Balanced risk-reward profile with moderate upside potential.

*Note: This fallback analysis was generated due to AI service unavailability.*
"""

    def _fallback_risk_analysis(self, company_profile: CompanyProfile) -> str:
        """Fallback risk analysis when AI fails"""
        return f"""
# COMPREHENSIVE RISK ANALYSIS

## Business Risks
- Competitive market environment
- Operational execution challenges
- Customer concentration concerns

## Financial Risks
- Interest rate sensitivity
- Credit and liquidity risks
- Capital allocation decisions

## Regulatory Risks
- Industry regulation changes
- Compliance cost increases
- Policy uncertainty impact

Key risk factors: {', '.join(company_profile.risk_factors[:5])}

*Note: This fallback analysis was generated due to AI service unavailability.*
"""
