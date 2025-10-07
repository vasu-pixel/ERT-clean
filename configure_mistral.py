#!/usr/bin/env python3
# configure_mistral.py - Configure ERT system for optimal Mistral:7b performance
import os
import json
import sys
import requests
from pathlib import Path

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)

def check_mistral_model():
    """Check if Mistral:7b model is available"""
    is_running, data = check_ollama_server()

    if not is_running:
        return False, "Ollama server not running"

    if isinstance(data, dict) and 'models' in data:
        models = [model['name'] for model in data['models']]
        mistral_models = [model for model in models if 'mistral' in model.lower()]

        if any('7b' in model for model in mistral_models):
            return True, mistral_models
        else:
            return False, f"Available models: {models}"

    return False, "Could not parse model list"

def pull_mistral_model():
    """Pull Mistral:7b model if not available"""
    print("🤖 Pulling Mistral:7b model from Ollama library...")

    try:
        response = requests.post(
            'http://localhost:11434/api/pull',
            json={'name': 'mistral:7b'},
            timeout=600  # 10 minute timeout
        )

        if response.status_code == 200:
            print("✅ Mistral:7b model pulled successfully")
            return True
        else:
            print(f"❌ Failed to pull model: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def update_mistral_configuration():
    """Update configuration files for optimal Mistral performance"""

    # Mistral-optimized configuration
    mistral_config = {
        "model": "mistral:7b",
        "max_tokens": 3500,  # Slightly lower for Mistral efficiency
        "temperature": 0.4,  # Slightly higher for Mistral creativity
        "default_recommendation": "HOLD",
        "report_style": "institutional",
        "ollama": {
            "model": "mistral:7b",
            "max_tokens": 3500,
            "temperature": 0.4,
            "backup_model": "llama3.1:8b",
            "base_url": "http://localhost:11434",
            "timeout": 150,  # Longer timeout for Mistral
            "retry_attempts": 3
        },
        "openai": {
            "model": "gpt-4o",
            "max_tokens": 16000,
            "temperature": 0.2,
            "backup_model": "gpt-3.5-turbo"
        },
        "analysis_parameters": {
            "dcf_terminal_growth": 2.5,
            "wacc_risk_free_rate": 4.5,
            "equity_risk_premium": 6.0,
            "default_tax_rate": 25.0,
            "peer_group_size": 5
        },
        "ai_engine": "ollama",
        "mistral_optimizations": {
            "section_length_target": 800,  # Optimal for Mistral output
            "reasoning_depth": "moderate",
            "technical_detail_level": "institutional",
            "prompt_engineering": "mistral_optimized"
        }
    }

    # Update config.json
    config_path = Path('config.json')
    with open(config_path, 'w') as f:
        json.dump(mistral_config, f, indent=2)

    print(f"✅ Updated {config_path} with Mistral:7b optimization")

    # Update .env file
    env_content = """# AI Engine Configuration
# Use 'ollama' for local LLM or 'openai' for OpenAI API
AI_ENGINE=ollama

# Ollama Configuration (for local LLM with Mistral)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Mistral-specific optimizations
MISTRAL_TEMPERATURE=0.4
MISTRAL_MAX_TOKENS=3500
MISTRAL_TIMEOUT=150

# OpenAI Configuration (backup/alternative)
OPENAI_API_KEY=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
"""

    env_path = Path('.env')
    with open(env_path, 'w') as f:
        f.write(env_content)

    print(f"✅ Updated {env_path} with Mistral:7b configuration")

def create_mistral_test_script():
    """Create a test script specifically for Mistral model"""

    test_script = '''#!/usr/bin/env python3
# test_mistral.py - Test Mistral:7b integration with ERT
import os
import sys
sys.path.append('src')

from src.utils.ollama_engine import OllamaEngine
from src.stock_report_generator_ollama import EnhancedEquityResearchGenerator

def test_mistral_connection():
    """Test basic Mistral connection"""
    print("🧪 Testing Mistral:7b connection...")

    try:
        engine = OllamaEngine(model='mistral:7b')

        # Simple test prompt
        response = engine.call_ollama(
            "Explain the importance of P/E ratio in equity analysis in 2 sentences.",
            max_tokens=100,
            temperature=0.4
        )

        print(f"✅ Mistral response: {response}")
        return True

    except Exception as e:
        print(f"❌ Mistral test failed: {e}")
        return False

def test_mistral_report_generation():
    """Test actual report generation with Mistral"""
    print("🧪 Testing Mistral report generation...")

    try:
        generator = EnhancedEquityResearchGenerator()

        # Test with a simple company profile
        print("   📊 Testing executive summary generation...")

        # Mock company profile for testing
        class MockProfile:
            def __init__(self):
                self.ticker = "TEST"
                self.company_name = "Test Company"
                self.sector = "Technology"
                self.industry = "Software"
                self.market_cap = 100000000000
                self.current_price = 150.00
                self.financial_metrics = {
                    'pe_ratio': 25.5,
                    'revenue_growth': 15.2,
                    'roe': 18.7
                }

        mock_profile = MockProfile()

        # Test executive summary generation
        summary = generator.generate_executive_summary_with_ai(mock_profile)

        if summary and len(summary) > 100:
            print(f"✅ Mistral generated {len(summary)} characters")
            print(f"   Preview: {summary[:200]}...")
            return True
        else:
            print(f"❌ Generated content too short: {len(summary) if summary else 0} characters")
            return False

    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False

def main():
    print("🚀 Mistral:7b Integration Test Suite")
    print("=" * 40)

    # Test 1: Basic connection
    connection_ok = test_mistral_connection()

    # Test 2: Report generation
    if connection_ok:
        generation_ok = test_mistral_report_generation()

        if generation_ok:
            print("\\n🎉 All Mistral tests passed!")
            print("   System is ready for Mistral-powered report generation")
        else:
            print("\\n⚠️  Connection OK but report generation failed")
    else:
        print("\\n❌ Basic connection failed - check Ollama and Mistral model")

if __name__ == '__main__':
    main()
'''

    test_path = Path('test_mistral.py')
    with open(test_path, 'w') as f:
        f.write(test_script)

    # Make executable
    os.chmod(test_path, 0o755)

    print(f"✅ Created {test_path} for testing Mistral integration")

def create_mistral_prompt_templates():
    """Create optimized prompt templates for Mistral"""

    templates = {
        "mistral_system_prompts": {
            "executive_summary": """You are a senior equity research analyst with 15+ years of experience at top-tier investment banks. Your analysis is trusted by institutional investors worldwide. Generate a comprehensive executive summary that includes clear investment recommendations, detailed financial analysis, and strategic insights. Be concise but thorough, focusing on actionable investment insights.""",

            "market_research": """You are a leading industry analyst specializing in comprehensive market research and competitive intelligence. Your reports are referenced by Fortune 500 companies for strategic planning. Provide detailed market analysis with specific data points, competitive positioning insights, and forward-looking industry trends. Focus on quantitative metrics and strategic implications.""",

            "financial_analysis": """You are a CFA charterholder and senior financial analyst with expertise in financial statement analysis and valuation modeling. Your work sets the standard for institutional-quality financial analysis. Provide detailed quantitative analysis with specific metrics, ratio analysis, and peer comparisons. Include margin analysis, efficiency metrics, and financial health indicators.""",

            "valuation_analysis": """You are a senior valuation expert with expertise in DCF modeling, comparable company analysis, and sophisticated valuation methodologies. Your models are used by investment banks for M&A transactions. Provide detailed valuation analysis with specific methodologies, assumptions, and sensitivity analysis. Include multiple valuation approaches and price target derivation.""",

            "investment_thesis": """You are a seasoned portfolio manager and investment strategist with a track record of identifying value creation opportunities. Your investment thesis framework is used by leading asset management firms. Provide balanced analysis of investment opportunities and risks, with specific catalysts, timeframes, and scenario analysis.""",

            "risk_analysis": """You are a senior risk analyst with expertise in comprehensive risk assessment across multiple dimensions. Your risk frameworks are implemented by major financial institutions. Provide detailed risk analysis with quantified impact assessment, mitigation strategies, and stress testing scenarios. Focus on business, financial, and operational risks."""
        },

        "mistral_prompt_formatting": {
            "instruction_prefix": "### TASK:",
            "context_prefix": "### CONTEXT:",
            "output_prefix": "### ANALYSIS:",
            "reasoning_prompt": "Think step-by-step and provide detailed reasoning for your analysis.",
            "format_instruction": "Structure your response with clear headings and bullet points for institutional readability."
        }
    }

    templates_path = Path('mistral_prompt_templates.json')
    with open(templates_path, 'w') as f:
        json.dump(templates, f, indent=2)

    print(f"✅ Created {templates_path} with Mistral-optimized prompts")

def main():
    """Main configuration function"""
    print("🔧 Configuring ERT for Mistral:7b")
    print("=" * 40)

    # Step 1: Check Ollama server
    print("1. Checking Ollama server status...")
    is_running, data = check_ollama_server()

    if not is_running:
        print("❌ Ollama server is not running")
        print("   Please start Ollama with: ollama serve")
        return False

    print("✅ Ollama server is running")

    # Step 2: Check Mistral model
    print("\\n2. Checking Mistral:7b model availability...")
    has_mistral, info = check_mistral_model()

    if not has_mistral:
        print(f"❌ Mistral:7b not found: {info}")
        print("   Attempting to pull Mistral:7b model...")

        if not pull_mistral_model():
            print("❌ Failed to pull Mistral model")
            return False
    else:
        print(f"✅ Mistral model available: {info}")

    # Step 3: Update configuration
    print("\\n3. Updating configuration files...")
    update_mistral_configuration()

    # Step 4: Create test scripts
    print("\\n4. Creating Mistral test utilities...")
    create_mistral_test_script()
    create_mistral_prompt_templates()

    # Step 5: Final summary
    print("\\n🎉 Mistral:7b configuration complete!")
    print("\\nNext steps:")
    print("1. Test the configuration: python test_mistral.py")
    print("2. Generate a report: python run_generate_report_ollama.py --ticker AAPL")
    print("3. Launch dashboard: python launch_dashboard.py")
    print("\\nMistral:7b Performance Characteristics:")
    print("• Faster generation (~30% faster than Llama3.1:8b)")
    print("• Good reasoning capabilities for financial analysis")
    print("• Optimized for instruction following")
    print("• Lower memory usage (~6GB vs 8GB)")
    print("• Excellent for structured output generation")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)