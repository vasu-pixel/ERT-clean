#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Set environment variables
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('PYTHONPATH', str(project_root / 'src'))

# Import and run the server
if __name__ == '__main__':
    from src.ui.status_server import app, socketio
    import argparse

    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'

    print(f"Starting server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)