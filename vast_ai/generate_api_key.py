#!/usr/bin/env python3
"""
Generate secure API key for ERT LLM Backend
"""

import secrets
import string
import hashlib
from datetime import datetime

def generate_secure_api_key(length: int = 32) -> str:
    """Generate a cryptographically secure API key"""
    alphabet = string.ascii_letters + string.digits + "-_"
    api_key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"ert-{api_key}"

def generate_api_key_hash(api_key: str) -> str:
    """Generate a hash of the API key for verification"""
    return hashlib.sha256(api_key.encode()).hexdigest()

if __name__ == "__main__":
    # Generate a new API key
    api_key = generate_secure_api_key()
    api_key_hash = generate_api_key_hash(api_key)

    print("=" * 60)
    print("ERT LLM Backend API Key Generated")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"API Key: {api_key}")
    print(f"Hash (SHA256): {api_key_hash}")
    print("=" * 60)
    print("\nUSAGE:")
    print("1. Set this as ERT_API_KEY environment variable in Vast.ai")
    print("2. Set this as ERT_LLM_API_KEY in Render deployment")
    print("3. Keep this key secure and do not share it")
    print("\nVast.ai Environment:")
    print(f"ERT_API_KEY={api_key}")
    print("\nRender Environment:")
    print(f"ERT_LLM_API_KEY={api_key}")
    print("=" * 60)