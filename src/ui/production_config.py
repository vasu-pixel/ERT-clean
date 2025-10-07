# src/ui/production_config.py
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS

def configure_production(app):
    """Configure Flask app for production"""

    # Security settings
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    app.config['DEBUG'] = False
    app.config['TESTING'] = False

    # CORS configuration
    CORS(app, origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ])

    # Rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["100 per hour", "10 per minute"]
    )

    # Security headers
    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

    return app