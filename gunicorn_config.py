"""
Gunicorn configuration for ERT production deployment
Optimized for Render.com with proper worker lifecycle management
"""

import os
import multiprocessing
import logging

# CRITICAL: Import eventlet and monkey-patch BEFORE any other imports
import eventlet
eventlet.monkey_patch()

# Logging
loglevel = 'info'
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Worker configuration
# Use only 1 worker for background thread consistency
# Render free tier has limited resources, so optimize for reliability over throughput
workers = 1  # Critical: Only 1 worker to ensure background thread persists
worker_class = 'eventlet'  # Use eventlet for SocketIO WebSocket support
# Note: eventlet doesn't use 'threads' parameter - it uses greenlets instead
worker_connections = 1000  # Max concurrent connections per worker

# Timeouts
timeout = 120  # 2 minutes for long report generation
graceful_timeout = 30
keepalive = 5

# Server mechanics
max_requests = 1000  # Restart worker after this many requests (prevents memory leaks)
max_requests_jitter = 50  # Add randomness to prevent all workers restarting simultaneously

# Preloading
preload_app = False  # Don't preload to allow proper worker initialization

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
backlog = 2048

# Process naming
proc_name = 'ert-production'

# Lifecycle hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    logging.info("ERT Gunicorn master process starting...")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    logging.info(f"Worker {worker.pid} spawned (age: {worker.age})")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass

def pre_exec(server):
    """Called just before a new master process is forked."""
    logging.info("Forking new master process")

def when_ready(server):
    """Called just after the server is started."""
    logging.info("ERT server is ready. Spawning workers...")

def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    logging.info(f"Worker {worker.pid} interrupted")

def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    logging.warning(f"Worker {worker.pid} aborted")

def on_exit(server):
    """Called just before exiting gunicorn."""
    logging.info("ERT Gunicorn master process exiting")

# SSL (not needed for Render, but kept for reference)
# keyfile = None
# certfile = None
