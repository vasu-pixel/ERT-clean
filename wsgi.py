#!/usr/bin/env python3
"""
WSGI entry point for production deployment with gunicorn
Used by Render and other cloud platforms
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set environment for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('ERT_DEBUG', 'false')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the production server with real data fetching and LLM integration
from production_server import app, start_background_worker, _ensure_directories

# Ensure required directories exist
_ensure_directories()

# Track if worker is started to prevent duplicates
_worker_started = False

def on_starting(server):
    """Called just before the master process is initialized."""
    logger.info("Gunicorn master process starting...")
    _ensure_directories()

def post_fork(server, worker):
    """
    Called just after a worker has been forked.
    Start background worker ONLY in the first worker to avoid duplicates.
    """
    global _worker_started

    logger.info(f"Worker {worker.pid} forked")

    # Only start background worker in first worker (worker with ID 1)
    # Gunicorn assigns worker IDs starting from 1
    if worker.age == 0 and not _worker_started:
        logger.info(f"Starting background report worker in worker {worker.pid}")
        start_background_worker()
        _worker_started = True
    else:
        logger.info(f"Skipping background worker start in worker {worker.pid}")

def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT signal."""
    logger.info(f"Worker {worker.pid} received interrupt signal")

def worker_abort(worker):
    """Called when a worker receives SIGABRT signal."""
    logger.info(f"Worker {worker.pid} aborted")

# Gunicorn configuration hooks
# These will be automatically called by Gunicorn if present in wsgi.py
__all__ = ['app', 'on_starting', 'post_fork', 'worker_int', 'worker_abort']

if __name__ == "__main__":
    # For local testing without Gunicorn
    _ensure_directories()
    start_background_worker()
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)