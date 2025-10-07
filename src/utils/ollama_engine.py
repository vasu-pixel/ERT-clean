# src/utils/ollama_engine.py
import os
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Any

from src.utils.ai_engine import AIEngine

# Configure logging
logger = logging.getLogger(__name__)

# Global flag to skip health checks in production
SKIP_HEALTH_CHECK = os.environ.get('FLASK_ENV') == 'production'

class OllamaEngine(AIEngine):
    """
    Ollama Engine for handling local LLM API calls with retry logic and error handling
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict] = None,
        auto_check: bool = True,
    ):
        """
        Initialize Ollama Engine

        Args:
            base_url: Ollama server URL (defaults to http://localhost:11434)
            model: Default model to use (defaults to llama2)
            config: Configuration dictionary
        """
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.default_model = model or os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')
        self.config = config or {}

        # Ensure base_url doesn't end with a slash
        self.base_url = self.base_url.rstrip('/')

        # Default configuration
        self.default_max_tokens = self.config.get('max_tokens', 4000)
        self.default_temperature = self.config.get('temperature', 0.3)

        # Cached health check state
        self._health_status: Optional[bool] = None
        self._last_health_check: float = 0.0
        self._model_verified: bool = False
        self._health_check_interval: int = int(os.getenv('OLLAMA_HEALTH_INTERVAL', '120'))

        # Test connection on initialization (optional)
        if auto_check and not self.test_connection(force=True):
            logger.warning("Ollama server connection failed. Please ensure Ollama is running.")

    def test_connection(self, force: bool = False) -> bool:
        """Test if Ollama server is accessible with comprehensive health checks"""
        # Reuse cached result unless forced or cache expired
        if not force and self._health_status is not None:
            if (time.time() - self._last_health_check) < self._health_check_interval:
                return self._health_status

        try:
            # Test basic connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error(f"Ollama server returned status {response.status_code}")
                self._health_status = False
                self._last_health_check = time.time()
                return False

            # Verify models are available
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]

            if not available_models:
                logger.warning("No models available in Ollama")
                self._health_status = False
                self._last_health_check = time.time()
                return False

            # Check if default model is available
            if self.default_model not in available_models:
                logger.warning(f"Default model {self.default_model} not found. Available: {available_models}")
                # Try to use first available model
                if available_models:
                    self.default_model = available_models[0]
                    logger.info(f"Switched to available model: {self.default_model}")
                else:
                    self._health_status = False
                    self._last_health_check = time.time()
                    return False

            # Only perform an actual generation test once per process (or when forced)
            if not self._model_verified or force:
                test_payload = {
                    "model": self.default_model,
                    "prompt": "System: You are a calculator.\n\nHuman: 1+1= ?\n\nAssistant:",
                    "stream": False,
                    "options": {"num_predict": 5}
                }

                test_response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=test_payload,
                    timeout=10
                )
                if test_response.status_code != 200:
                    logger.error(f"Ollama generation test failed: {test_response.status_code}")
                    self._health_status = False
                    self._last_health_check = time.time()
                    return False

                self._model_verified = True

            logger.info(f"✅ Ollama health check passed - Model: {self.default_model}")
            self._health_status = True
            self._last_health_check = time.time()
            return True

        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            self._health_status = False
            self._last_health_check = time.time()
            return False

    def _call_ollama_with_retry(self, **kwargs):
        """Call Ollama API with automatic retry logic"""
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=kwargs,
                    timeout=300  # 5 minute timeout for generation
                )
                response.raise_for_status()
                return response
            except Exception as e:
                error_msg = str(e).lower()

                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    logger.error(f"Ollama API call failed after {max_retries} attempts: {e}")
                    raise

                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)

                if "timeout" in error_msg or "connection" in error_msg:
                    delay = max(delay, 30)  # Wait at least 30 seconds for connection issues
                    logger.warning(f"Connection issue, waiting {delay} seconds before retry {attempt + 1}")
                else:
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {delay} seconds: {e}")

                time.sleep(delay)

    def call_ollama(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> str:
        """
        Call Ollama with a simple prompt and return the response text

        Args:
            prompt: The user prompt
            model: Model to use (defaults to config model)
            max_tokens: Maximum tokens (defaults to config max_tokens)
            temperature: Temperature setting (defaults to config temperature)
            system_prompt: Optional system prompt

        Returns:
            Response text from Ollama
        """
        try:
            # Use provided values or fall back to defaults
            model = model or self.default_model
            max_tokens = max_tokens or self.default_max_tokens
            temperature = temperature or self.default_temperature

            # Build the full prompt with system context if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nHuman: {prompt}\n\nAssistant:"

            # Prepare request payload
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["Human:", "System:"]
                }
            }

            # Call Ollama with retry logic
            response = self._call_ollama_with_retry(**payload)

            # Parse response
            result = response.json()
            return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Error in call_ollama: {e}")
            raise
            
    def call_ai(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """
        A generic method to call the AI model with a prompt.
        """
        return self.call_ollama(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    def call_ollama_chat(self, messages: List[Dict], model: Optional[str] = None,
                        max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """
        Call Ollama chat API with a list of messages

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to config model)
            max_tokens: Maximum tokens (defaults to config max_tokens)
            temperature: Temperature setting (defaults to config temperature)

        Returns:
            Response text from Ollama
        """
        try:
            # Use provided values or fall back to defaults
            model = model or self.default_model
            max_tokens = max_tokens or self.default_max_tokens
            temperature = temperature or self.default_temperature

            # Convert messages to a single prompt (Ollama doesn't have chat API like OpenAI)
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"Human: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")

            prompt_parts.append("Assistant:")
            full_prompt = "\n\n".join(prompt_parts)

            # Prepare request payload
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["Human:", "System:"]
                }
            }

            # Call Ollama with retry logic
            response = self._call_ollama_with_retry(**payload)

            # Parse response
            result = response.json()
            return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Error in call_ollama_chat: {e}")
            raise

    def generate_section(self, section_name: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a specific report section with comprehensive error handling and retry logic

        Args:
            section_name: Name of the section being generated
            prompt: The content prompt
            system_prompt: Optional custom system prompt

        Returns:
            Generated section content
        """
        # Pre-generation health check
        if not self.test_connection():
            logger.error(f"Ollama unavailable for {section_name} generation")
            return self._generate_detailed_fallback_section(section_name, prompt)

        try:
            # Default system prompts for different sections
            default_system_prompts = {
                'executive_summary': "You are a senior equity research analyst at a top-tier investment bank. Generate institutional-quality research content with clear investment recommendations.",
                'market_research': "You are a senior industry analyst specializing in comprehensive market research. Provide detailed, data-driven analysis with specific insights.",
                'financial_analysis': "You are a senior financial analyst specializing in comprehensive financial statement analysis. Provide detailed, quantitative analysis with specific metrics.",
                'valuation_analysis': "You are a senior valuation expert specializing in DCF models and comparable company analysis. Provide detailed, methodical valuation analysis.",
                'sentiment_analysis': "You are a senior market sentiment analyst providing institutional-quality sentiment analysis for equity research.",
                'investment_thesis': "You are a senior investment strategist providing balanced investment thesis analysis. Consider both opportunities and risks with specific catalysts.",
                'risk_analysis': "You are a senior risk analyst specializing in comprehensive risk assessment. Provide detailed risk analysis with quantified impact where possible.",
                'business_overview': "You are a senior corporate strategist dissecting business models, revenue architecture, and operating structure for institutional investors.",
                'segment_analysis': "You are an equity analyst delivering segment-by-segment and geographic performance breakdowns with quantified comparisons.",
                'technology_innovation': "You are a technology analyst covering R&D pipelines, innovation priorities, patents, and partnerships.",
                'go_to_market': "You are a commercial strategy advisor detailing go-to-market motions, customer acquisition, retention, and monetization levers.",
                'competitive_landscape': "You are a competitive intelligence expert benchmarking the company against peers with qualitative and quantitative evidence.",
                'financial_statement_deep_dive': "You are a senior controller unpacking P&L, balance sheet, and cash flow dynamics with historical trend analysis.",
                'forecast_scenarios': "You are a quantitative strategist crafting base, bull, and bear forecasts with clear assumptions and sensitivities.",
                'capital_allocation': "You are a corporate finance specialist evaluating capital deployment, leverage, liquidity, and shareholder returns.",
                'esg_regulatory': "You are an ESG and regulatory analyst assessing sustainability programs, governance, and regulatory outlook.",
                'risk_matrix': "You are a chief risk officer constructing a risk matrix with probability, impact, and mitigation strategies.",
                'valuation_deep_dive': "You are a valuation specialist producing extensive DCF, relative valuation, and precedent transaction analysis.",
                'catalysts_timeline': "You are an investment strategist mapping catalysts, milestones, and monitoring indicators across time horizons."
            }

            # Use custom system prompt or default for section
            final_system_prompt = system_prompt or default_system_prompts.get(section_name,
                "You are a senior financial analyst providing institutional-quality research analysis.")

            logger.info(f"Generating {section_name} section using Ollama...")

            # Generate with multiple attempts and validation
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    content = self.call_ollama(
                        prompt=prompt,
                        system_prompt=final_system_prompt,
                        max_tokens=self.config.get('max_tokens', 4000),
                        temperature=self.config.get('temperature', 0.3)
                    )

                    # Validate content quality
                    if self._validate_section_content(content, section_name):
                        logger.info(f"✅ Successfully generated {section_name} section ({len(content.split())} words)")
                        return content
                    else:
                        logger.warning(f"Generated content for {section_name} failed validation (attempt {attempt + 1})")
                        if attempt < max_attempts - 1:
                            continue

                except Exception as e:
                    logger.warning(f"Generation attempt {attempt + 1} failed for {section_name}: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(5)  # Brief pause before retry
                        continue

            # If all attempts failed, return detailed fallback
            logger.error(f"All generation attempts failed for {section_name}")
            return self._generate_detailed_fallback_section(section_name, prompt)

        except Exception as e:
            logger.error(f"Critical error generating {section_name} section: {e}")
            return self._generate_detailed_fallback_section(section_name, prompt)

    def _validate_section_content(self, content: str, section_name: str) -> bool:
        """Validate that generated content meets minimum quality standards"""
        if not content or len(content.strip()) < 100:
            return False

        # Check for error messages
        error_indicators = [
            "temporarily unavailable",
            "error generating",
            "please try again",
            "connection failed",
            "unable to generate"
        ]

        content_lower = content.lower()
        for indicator in error_indicators:
            if indicator in content_lower:
                return False

        # Check minimum word count based on section
        word_count = len(content.split())
        min_words = {
            'executive_summary': 300,
            'financial_analysis': 400,
            'valuation_analysis': 400,
            'investment_thesis': 300,
            'risk_analysis': 300
        }

        required_words = min_words.get(section_name, 200)
        return word_count >= required_words

    def _generate_detailed_fallback_section(self, section_name: str, original_prompt: str) -> str:
        """Generate detailed fallback content when Ollama fails"""
        section_title = section_name.upper().replace('_', ' ')

        fallback_templates = {
            'executive_summary': f"""# EXECUTIVE SUMMARY

**Investment Recommendation:** HOLD (Service Unavailable)

Due to technical limitations, detailed analysis is temporarily unavailable. This section would typically include:

- Investment recommendation with price target
- Key financial metrics and valuation summary
- Primary investment highlights and risks
- Growth catalysts and competitive positioning

**Status:** Report generation encountered technical difficulties. Please ensure Ollama service is running and configured properly, then regenerate this report for complete analysis.

*Word count requirement: 300+ words*
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*""",

            'financial_analysis': f"""# FINANCIAL ANALYSIS

**Analysis Status:** Technical difficulties prevented complete financial analysis generation.

This section would typically cover:

## Revenue Analysis
- Historical revenue trends and growth rates
- Segment revenue breakdown and performance
- Revenue quality and sustainability assessment

## Profitability Metrics
- Gross, operating, and net margin analysis
- Return on equity (ROE) and return on assets (ROA)
- Efficiency ratios and peer comparisons

## Balance Sheet Strength
- Asset quality and composition analysis
- Debt levels and capital structure
- Working capital management

## Cash Flow Analysis
- Operating cash flow trends
- Free cash flow generation
- Capital allocation priorities

**Status:** Complete analysis unavailable due to service limitations.
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*""",

            'valuation_analysis': f"""# VALUATION ANALYSIS

**Valuation Status:** Technical difficulties prevented complete valuation analysis.

This section would typically include:

## DCF Valuation
- Revenue and margin assumptions
- Terminal value calculations
- Weighted average cost of capital (WACC)
- Sensitivity analysis

## Relative Valuation
- Peer multiple comparisons
- Industry benchmarking
- Historical trading ranges

## Valuation Summary
- Target price methodology
- Upside/downside scenarios
- Key valuation risks

**Status:** Detailed valuation models unavailable due to service limitations.
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*"""
        }

        default_fallback = f"""# {section_title}

**Section Status:** Content generation temporarily unavailable due to technical difficulties.

This section would provide detailed analysis relevant to {section_title.lower()}. Please ensure the AI service is properly configured and running, then regenerate this report.

**Technical Note:** This appears to be a service connectivity issue rather than a data problem. Check:
1. Ollama service status
2. Model availability
3. Network connectivity
4. System resources

*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*"""

        return fallback_templates.get(section_name, default_fallback)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        connection_ok = self.test_connection()
        return {
            'model': self.default_model,
            'base_url': self.base_url,
            'max_tokens': self.default_max_tokens,
            'temperature': self.default_temperature,
            'connection_status': connection_ok
        }

    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models_data = response.json()
            return [model['name'] for model in models_data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        try:
            logger.info(f"Pulling model {model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={{"name": model_name}},
                timeout=600  # 10 minute timeout for model downloads
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False


# Utility functions for backward compatibility with OpenAI interface
def call_ollama(prompt: str, model: str = "gpt-oss:20b", max_tokens: int = 1000,
               temperature: float = 0.3, base_url: Optional[str] = None) -> str:
    """
    Standalone function to call Ollama (for backward compatibility with OpenAI interface)

    Args:
        prompt: The prompt to send to Ollama
        model: Model to use
        max_tokens: Maximum tokens
        temperature: Temperature setting
        base_url: Ollama server URL (optional)

    Returns:
        Response text from Ollama
    """
    config = {
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    engine = OllamaEngine(base_url=base_url, model=model, config=config)
    return engine.call_ollama(prompt)


# Initialize global engine instance
_global_engine = None

def get_global_engine(config: Optional[Dict] = None) -> OllamaEngine:
    """Get or create global Ollama engine instance"""
    global _global_engine

    if _global_engine is None:
        _global_engine = OllamaEngine(config=config)

    return _global_engine


if __name__ == "__main__":
    # Test the Ollama engine
    try:
        engine = OllamaEngine()

        print("Testing Ollama Engine...")
        print(f"Model info: {engine.get_model_info()}")

        print("Available models:", engine.list_models())

        if engine.test_connection():
            print("✅ Ollama Engine working correctly!")

            # Test basic call
            response = engine.call_ollama("What is 2+2? Answer in one sentence.", max_tokens=50)
            print(f"Test response: {response}")
        else:
            print("❌ Ollama Engine connection failed")
            print("Please ensure Ollama is running with: ollama serve")

    except Exception as e:
        print(f"❌ Error testing Ollama Engine: {e}")
