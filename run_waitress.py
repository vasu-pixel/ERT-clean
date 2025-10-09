#!/usr/bin/env python3
"""
Production server using Waitress (simple, reliable, no monkey-patching)
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set environment
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('ERT_DEBUG', 'false')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("ERT - Equity Research Tool")
logger.info("="*80)

# Import app
from production_server import app, start_background_worker, _ensure_directories

# Setup
logger.info("Creating required directories...")
_ensure_directories()

logger.info("Starting background report generator worker...")
start_background_worker()

# Get port
port = int(os.environ.get('PORT', 10000))
host = '0.0.0.0'

logger.info(f"🚀 Starting Waitress WSGI server")
logger.info(f"   Host: {host}")
logger.info(f"   Port: {port}")
logger.info(f"   Threads: 4")
logger.info(f"   URL: http://{host}:{port}")
logger.info("="*80)

# Start Waitress
from waitress import serve
serve(
    app,
    host=host,
    port=port,
    threads=4,
    url_scheme='https',
    channel_timeout=120,
    connection_limit=1000,
    cleanup_interval=30,
    asyncore_use_poll=True
)
