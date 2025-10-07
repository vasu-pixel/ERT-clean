#!/usr/bin/env python3
"""Test environment variable loading for Render deployment"""

import os

print("=== ENVIRONMENT VARIABLE TEST ===")
print(f"REMOTE_LLM_INTEGRATION: {os.getenv('REMOTE_LLM_INTEGRATION', 'NOT SET')}")
print(f"OLLAMA_INTEGRATION: {os.getenv('OLLAMA_INTEGRATION', 'NOT SET')}")
print(f"ERT_LLM_BACKEND_URL: {os.getenv('ERT_LLM_BACKEND_URL', 'NOT SET')}")
print(f"ERT_LLM_API_KEY: {os.getenv('ERT_LLM_API_KEY', 'NOT SET')}")
print("=== END TEST ===")

# Test the configuration loading
try:
    from src.ui.status_server_config import StatusServerConfig
    config = StatusServerConfig.from_environment()
    print(f"Config remote_llm_integration: {config.features.remote_llm_integration}")
    print(f"Config ollama_integration: {config.features.ollama_integration}")
except Exception as e:
    print(f"Config loading error: {e}")