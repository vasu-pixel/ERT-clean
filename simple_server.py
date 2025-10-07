#!/usr/bin/env python3
import os
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# Simple Flask app for Render deployment
app = Flask(__name__, template_folder='src/ui/templates', static_folder='src/ui/static')
app.config['SECRET_KEY'] = 'minimal-animals-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    return jsonify({
        'status': 'running',
        'message': 'Minimal Animals Equity Research Tool',
        'version': '1.0.0'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'

    print(f"Starting Minimal Animals server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)