# src/utils/openai_engine.py
import os
import time
import logging
from openai import OpenAI
from typing import Dict, List, Optional, Any

from src.utils.ai_engine import AIEngine

# Configure logging
logger = logging.getLogger(__name__)

class OpenAIEngine(AIEngine):
    """
    GPT Engine for handling OpenAI API calls with retry logic and error handling
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize GPT Engine

        Args:
            api_key: OpenAI API key (if None, will get from environment)
            config: Configuration dictionary
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.config = config or {}

        # Default configuration
        self.default_model = self.config.get('model', 'gpt-3.5-turbo')
        self.default_max_tokens = self.config.get('max_tokens', 1000)
        self.default_temperature = self.config.get('temperature', 0.3)

    def _call_openai_with_retry(self, **kwargs):
        """Call OpenAI API with automatic retry logic"""
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                error_msg = str(e).lower()

                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}")
                    raise

                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)

                if "rate limit" in error_msg:
                    delay = max(delay, 60)  # Wait at least 1 minute for rate limits
                    logger.warning(f"Rate limit hit, waiting {delay} seconds before retry {attempt + 1}")
                else:
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {delay} seconds: {e}")

                time.sleep(delay)

    def call_gpt(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> str:
        """
        Call GPT with a simple prompt and return the response text

        Args:
            prompt: The user prompt
            model: Model to use (defaults to config model)
            max_tokens: Maximum tokens (defaults to config max_tokens)
            temperature: Temperature setting (defaults to config temperature)
            system_prompt: Optional system prompt

        Returns:
            Response text from GPT
        """
        try:
            # Use provided values or fall back to defaults
            model = model or self.default_model
            max_tokens = max_tokens or self.default_max_tokens
            temperature = temperature or self.default_temperature

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call OpenAI with retry logic
            response = self._call_openai_with_retry(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error in call_gpt: {e}")
            raise

    def call_ai(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """
        A generic method to call the AI model with a prompt.
        """
        return self.call_gpt(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_section(self, section_name: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a specific report section with appropriate system prompt

        Args:
            section_name: Name of the section being generated
            prompt: The content prompt
            system_prompt: Optional custom system prompt

        Returns:
            Generated section content
        """
        try:
            # Default system prompts for different sections
            default_system_prompts = {
                'executive_summary': "You are a senior equity research analyst at a top-tier investment bank. Generate institutional-quality research content.",
                'market_research': "You are a senior industry analyst specializing in comprehensive market research. Provide detailed, data-driven analysis.",
                'financial_analysis': "You are a senior financial analyst specializing in comprehensive financial statement analysis. Provide detailed, quantitative analysis.",
                'valuation_analysis': "You are a senior valuation expert specializing in DCF models and comparable company analysis. Provide detailed, methodical valuation analysis.",
                'sentiment_analysis': "You are a senior market sentiment analyst providing institutional-quality sentiment analysis for equity research.",
                'investment_thesis': "You are a senior investment strategist providing balanced investment thesis analysis. Consider both opportunities and risks.",
                'risk_analysis': "You are a senior risk analyst specializing in comprehensive risk assessment. Provide detailed risk analysis with quantified impact where possible.",
                'business_overview': "You are a senior corporate strategist dissecting business models, operational structure, and revenue architecture for institutional investors.",
                'segment_analysis': "You are an equity analyst performing deep segment and geographic breakdowns with quantified performance insights and comparisons.",
                'technology_innovation': "You are a technology analyst covering R&D, innovation pipelines, emerging technologies, and product roadmaps in depth.",
                'go_to_market': "You are a commercial strategy expert detailing distribution, sales, marketing, and customer monetization levers.",
                'competitive_landscape': "You are a competitive intelligence analyst benchmarking position, share dynamics, and strategic moats versus peers.",
                'financial_statement_deep_dive': "You are a senior financial controller analyzing income statements, balance sheets, and cash flows with historical trend interpretation.",
                'forecast_scenarios': "You are a quantitative strategist building multi-scenario forecasts with explicit assumptions and sensitivities.",
                'capital_allocation': "You are a corporate finance specialist evaluating capital allocation, liquidity, leverage, and shareholder return policies.",
                'esg_regulatory': "You are an ESG and regulatory analyst assessing sustainability, governance, and compliance landscapes for institutional audiences.",
                'risk_matrix': "You are a risk officer constructing comprehensive risk matrices with probability, impact, and mitigations.",
                'valuation_deep_dive': "You are a valuation specialist producing exhaustive DCF, multiples, and precedent analyses with bridges and sensitivities.",
                'catalysts_timeline': "You are an investment strategist outlining catalysts, timelines, monitoring indicators, and potential inflection points."
            }

            # Use custom system prompt or default for section
            final_system_prompt = system_prompt or default_system_prompts.get(section_name,
                "You are a senior financial analyst providing institutional-quality research analysis.")

            logger.info(f"Generating {section_name} section...")

            return self.call_gpt(
                prompt=prompt,
                system_prompt=final_system_prompt,
                max_tokens=self.config.get('max_tokens', 4000),  # Longer sections
                temperature=self.config.get('temperature', 0.3)
            )

        except Exception as e:
            logger.error(f"Error generating {section_name} section: {e}")
            # Return fallback content instead of crashing
            return f"# {section_name.upper().replace('_', ' ')}\n\nSection generation temporarily unavailable. Please try again later."

    def test_connection(self) -> bool:
        """Test if the OpenAI API connection is working"""
        try:
            self.call_gpt("Test", max_tokens=5)
            logger.info("OpenAI API connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            'model': self.default_model,
            'max_tokens': self.default_max_tokens,
            'temperature': self.default_temperature,
            'api_key_configured': bool(self.api_key)
        }
