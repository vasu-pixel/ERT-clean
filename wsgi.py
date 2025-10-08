#!/usr/bin/env python3
"""
WSGI entry point for production deployment with gunicorn
Used by Render and other cloud platforms
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set environment for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('ERT_DEBUG', 'false')

# Import the production server with real data fetching and LLM integration
from production_server import app

if __name__ == "__main__":
    # For local testing
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)