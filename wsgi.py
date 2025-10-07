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

from src.ui.status_server import create_app
from src.ui.status_server_config import StatusServerConfig

def create_wsgi_app():
    """Create Flask application for WSGI deployment"""
    try:
        # Load configuration from environment
        config = StatusServerConfig.from_environment()

        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid configuration")

        # Create Flask app
        app = create_app(config)

        return app

    except Exception as e:
        print(f"Failed to create WSGI app: {e}")
        raise

# Create the application instance
app = create_wsgi_app()

if __name__ == "__main__":
    # For local testing
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)