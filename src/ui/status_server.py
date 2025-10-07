# src/ui/status_server.py
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import queue
from functools import lru_cache, wraps

# Import configuration (adjust path as needed)
try:
    from status_server_config import StatusServerConfig, FeatureConfig
except ImportError:
    # Fallback for different import scenarios
    from src.ui.status_server_config import StatusServerConfig, FeatureConfig

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration with debugging
print(f"DEBUG: REMOTE_LLM_INTEGRATION={os.getenv('REMOTE_LLM_INTEGRATION')}")
print(f"DEBUG: OLLAMA_INTEGRATION={os.getenv('OLLAMA_INTEGRATION')}")
logger.info(f"Environment variables: REMOTE_LLM_INTEGRATION={os.getenv('REMOTE_LLM_INTEGRATION')}, OLLAMA_INTEGRATION={os.getenv('OLLAMA_INTEGRATION')}")

config = StatusServerConfig.from_environment()
print(f"DEBUG: Loaded config remote_llm={config.features.remote_llm_integration}, ollama={config.features.ollama_integration}")
logger.info(f"Loaded config: remote_llm={config.features.remote_llm_integration}, ollama={config.features.ollama_integration}")

if not config.validate():
    logger.error("Configuration validation failed")
    sys.exit(1)

# Add project root to path
project_root = config.project_root
sys.path.insert(0, project_root)

# Feature-aware imports with robust mocking
class MockEnhancedEquityResearchGenerator:
    """Mock implementation when advanced features are disabled"""

    def __init__(self, *args, **kwargs):
        logger.info("Using mock equity research generator")

    def fetch_comprehensive_data(self, ticker: str) -> None:
        logger.info(f"Mock data fetch for {ticker}")
        time.sleep(1)  # Simulate work

    def generate_comprehensive_report(self, ticker: str) -> str:
        logger.info(f"Mock report generation for {ticker}")
        return f"Mock report for {ticker} - Advanced features not available in this environment"

def mock_get_system_status() -> Dict[str, Any]:
    """Mock system status when advanced features are disabled"""
    return {
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'ollama_available': False,
        'mode': 'basic',
        'features': 'limited'
    }

# Initialize features based on configuration
print(f"DEBUG: Feature selection: advanced_features={config.features.advanced_features}")
logger.info(f"Feature selection: advanced_features={config.features.advanced_features}")

if config.features.advanced_features:
    try:
        print(f"DEBUG: Checking integrations: ollama={config.features.ollama_integration}, remote_llm={config.features.remote_llm_integration}")
        logger.info(f"Checking integrations: ollama={config.features.ollama_integration}, remote_llm={config.features.remote_llm_integration}")

        if config.features.ollama_integration:
            print("DEBUG: Loading Ollama integration...")
            from src.stock_report_generator_ollama import EnhancedEquityResearchGenerator, get_system_status
            from src.utils.ollama_engine import OllamaEngine
            logger.info("Advanced features with Ollama integration loaded")
        elif config.features.remote_llm_integration:
            print("DEBUG: Loading Remote LLM integration...")
            from src.utils.remote_llm_client import RemoteEquityResearchGenerator
            EnhancedEquityResearchGenerator = RemoteEquityResearchGenerator
            get_system_status = mock_get_system_status  # Use mock for system status
            print("DEBUG: Remote LLM integration loaded successfully")
            logger.info("Advanced features with Remote LLM integration loaded")
        else:
            # No AI engine configured
            print("DEBUG: No AI engine configured, using mock")
            logger.info("No AI engine configured, using mock")
            EnhancedEquityResearchGenerator = MockEnhancedEquityResearchGenerator
            get_system_status = mock_get_system_status
            logger.warning("No AI engine configured, using mock implementations")

    except ImportError as e:
        logger.warning(f"Advanced features import failed: {e}")
        EnhancedEquityResearchGenerator = MockEnhancedEquityResearchGenerator
        get_system_status = mock_get_system_status
else:
    logger.info("Advanced features explicitly disabled in configuration")
    EnhancedEquityResearchGenerator = MockEnhancedEquityResearchGenerator
    get_system_status = mock_get_system_status

import yfinance as yf

# TTL cache implementation
def ttl_cache(ttl_seconds: int):
    """Time-to-live cache decorator"""
    def decorator(func):
        cache = {}
        lock = threading.RLock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()

            with lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl_seconds:
                        return value
                    else:
                        del cache[key]

                result = func(*args, **kwargs)
                cache[key] = (result, now)
                return result

        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache), 'ttl': ttl_seconds}
        return wrapper
    return decorator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ttl_cache(ttl_seconds=config.features.cache_ttl_seconds if hasattr(config.features, 'cache_ttl_seconds') else 60)
def get_stock_info(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        info = stock.info
        end_time = time.time()
        logger.info(f"yfinance call for {ticker} took {end_time - start_time:.2f} seconds")

        if not info.get('shortName'):
            return None

        price = info.get('regularMarketPrice') or info.get('currentPrice')
        previous_close = info.get('previousClose')
        price_change = price - previous_close if price and previous_close else 0
        percent_change = (price_change / previous_close) * 100 if previous_close else 0

        return {
            'ticker': ticker,
            'name': info.get('shortName'),
            'price': price,
            'price_change': price_change,
            'percent_change': percent_change,
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
        }
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        return None

def serialize_report_progress(report: 'ReportProgress') -> Dict:
    """Safely serialize ReportProgress object for JSON transmission"""
    data = asdict(report)
    # Convert datetime objects to ISO strings
    if 'start_time' in data and isinstance(report.start_time, datetime):
        data['start_time'] = report.start_time.isoformat()
    if 'estimated_completion' in data and report.estimated_completion and isinstance(report.estimated_completion, datetime):
        data['estimated_completion'] = report.estimated_completion.isoformat()
    return data

@dataclass
class ReportProgress:
    """Track progress of report generation"""
    ticker: str
    status: str  # 'queued', 'fetching_data', 'generating_sections', 'completed', 'failed'
    progress: int  # 0-100
    current_section: str
    start_time: datetime
    estimated_completion: Optional[datetime]
    error_message: Optional[str] = None
    word_count: int = 0
    report_path: Optional[str] = None
    abort_requested: bool = False

class AbortSignal(Exception):
    """Raised to interrupt report generation when an abort is requested."""

class ReportStatusManager:
    """Manages report generation status and progress tracking"""

    def __init__(self, config: Optional[StatusServerConfig] = None):
        self.config = config or StatusServerConfig.from_environment()
        self.active_reports: Dict[str, ReportProgress] = {}
        self.completed_reports: List[Dict] = []
        self.report_queue = queue.Queue(
            maxsize=getattr(self.config, 'max_queue_size', 100) if hasattr(self.config, 'max_queue_size') else 100
        )
        self.socketio = None
        self.generator = None
        self.system_status = {}
        self.abort_event = threading.Event()
        self.abort_lock = threading.Lock()
        self.abort_active = False
        self.update_system_status()

    def set_socketio(self, socketio):
        """Set SocketIO instance for real-time updates"""
        self.socketio = socketio

    def update_system_status(self):
        """Update system status information"""
        try:
            self.system_status = get_system_status()
            self.system_status['active_reports'] = len(self.active_reports)
            self.system_status['queue_size'] = self.report_queue.qsize()
        except Exception as e:
            logger.error(f"Error updating system status: {e}")

    def add_report_to_queue(self, ticker: str) -> str:
        """Add a report to the generation queue"""
        ticker = ticker.upper()

        max_length = getattr(self.config.features, 'max_ticker_length', 12) if hasattr(self.config, 'features') else 12
        if not ticker or len(ticker) > max_length:
            raise ValueError(f'Invalid ticker symbol (max length: {max_length})')

        report_id = f"{ticker}_{int(time.time())}"

        progress = ReportProgress(
            ticker=ticker.upper(),
            status='queued',
            progress=0,
            current_section='Waiting in queue',
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(minutes=5)
        )

        self.active_reports[report_id] = progress
        self.report_queue.put(report_id)

        # Emit update to connected clients
        if self.socketio:
            self.socketio.emit('report_update', {
                'report_id': report_id,
                'data': serialize_report_progress(progress)
            })

        logger.info(f"Added {ticker} to report queue with ID: {report_id}")
        return report_id

    def update_progress(self, report_id: str, status: str = None, progress: int = None,
                       current_section: str = None, error_message: str = None,
                       word_count: int = None, report_path: str = None):
        """Update progress for a specific report"""
        if report_id not in self.active_reports:
            return

        report = self.active_reports[report_id]

        if status:
            report.status = status
        if progress is not None:
            report.progress = progress
        if current_section:
            report.current_section = current_section
        if error_message:
            report.error_message = error_message
        if word_count:
            report.word_count = word_count
        if report_path:
            report.report_path = report_path

        # Update estimated completion based on progress
        if progress and progress > 10:
            elapsed = datetime.now() - report.start_time
            estimated_total = elapsed * (100 / progress)
            report.estimated_completion = report.start_time + estimated_total

        # Emit update to connected clients
        if self.socketio:
            self.socketio.emit('report_update', {
                'report_id': report_id,
                'data': serialize_report_progress(report)
            })

    def request_abort(self) -> Dict[str, int]:
        """Abort all pending and active reports."""
        aborted_queued = 0
        aborted_active = 0

        # Set abort state atomically
        with self.abort_lock:
            self.abort_active = True

        # Safely drain the queue
        pending_ids = []
        try:
            with self.report_queue.mutex:
                # Get current queue size for counting
                aborted_queued = self.report_queue.qsize()
                # Store pending IDs before clearing
                pending_ids = list(self.report_queue.queue)
                # Clear the entire queue
                self.report_queue.queue.clear()
        except Exception as e:
            logger.error(f"Error draining queue: {e}")

        # Mark queued reports as aborted
        for report_id in pending_ids:
            if report_id in self.active_reports:
                report = self.active_reports[report_id]
                report.abort_requested = True
                report.status = 'aborted'
                report.progress = 0
                report.current_section = 'Aborted before start'
                report.error_message = 'Aborted before start'
                self.complete_report(report_id, success=False)

        for report_id, report in list(self.active_reports.items()):
            if getattr(report, 'abort_requested', False):
                continue
            if report.status in ('completed', 'failed', 'aborted'):
                continue

            report.abort_requested = True
            report.status = 'aborting'
            report.current_section = 'Abort requested by user'
            report.error_message = 'Abort requested by user'

            self.update_progress(
                report_id,
                status=report.status,
                current_section=report.current_section,
                error_message=report.error_message
            )
            aborted_active += 1

        if aborted_active:
            self.abort_event.set()

        self.update_system_status()

        return {
            'aborted_active': aborted_active,
            'aborted_queued': aborted_queued,
            'remaining_active': len(self.active_reports),
            'queue_size': self.report_queue.qsize()
        }

    def reset_abort_state(self):
        """Reset abort state after all reports are cleaned up"""
        with self.abort_lock:
            # Only reset if no active reports remain
            if len(self.active_reports) == 0:
                self.abort_active = False
                self.abort_event.clear()
                logger.info("Abort state reset - system ready for new reports")
                return True
            else:
                logger.warning("Cannot reset abort state - active reports still exist")
                return False

    def complete_report(self, report_id: str, success: bool = True, report_path: str = None, metadata: Optional[Dict] = None):
        """Mark a report as completed"""
        if report_id not in self.active_reports:
            return

        report = self.active_reports[report_id]

        if success:
            report.status = 'completed'
            report.progress = 100
            report.current_section = 'Report completed successfully'
            if report_path:
                report.report_path = report_path
        else:
            if report.status != 'aborted':
                report.status = 'failed'
                report.current_section = 'Report generation failed'

        # Move to completed reports
        completed_data = asdict(report)
        # Convert datetime objects to ISO strings for JSON serialization
        if 'start_time' in completed_data and isinstance(report.start_time, datetime):
            completed_data['start_time'] = report.start_time.isoformat()
        if 'estimated_completion' in completed_data and report.estimated_completion and isinstance(report.estimated_completion, datetime):
            completed_data['estimated_completion'] = report.estimated_completion.isoformat()
        completed_data['completion_time'] = datetime.now().isoformat()
        if metadata:
            completed_data['report_metadata'] = metadata
        completed_data['report_id'] = report_id
        self.completed_reports.append(completed_data)

        # Remove from active reports
        del self.active_reports[report_id]

        # Check if we can reset abort state
        if self.abort_active and len(self.active_reports) == 0:
            self.reset_abort_state()

        # Emit completion update
        if self.socketio:
            self.socketio.emit('report_completed', {
                'report_id': report_id,
                'success': success,
                'data': completed_data
            })

        logger.info(f"Report {report_id} completed with status: {'success' if success else 'failed'}")

    def get_recent_reports(self, limit: int = 20) -> List[Dict]:
        """Get recent completed reports"""
        return sorted(self.completed_reports,
                     key=lambda x: x['start_time'], reverse=True)[:limit]

    def get_status_summary(self) -> Dict:
        """Get overall status summary"""
        self.update_system_status()

        # Convert active reports to JSON-serializable format
        active_reports_json = {}
        for k, v in self.active_reports.items():
            report_dict = asdict(v)
            # Convert datetime objects to ISO strings
            if 'start_time' in report_dict and report_dict['start_time']:
                report_dict['start_time'] = v.start_time.isoformat() if isinstance(v.start_time, datetime) else report_dict['start_time']
            if 'estimated_completion' in report_dict and report_dict['estimated_completion']:
                report_dict['estimated_completion'] = v.estimated_completion.isoformat() if isinstance(v.estimated_completion, datetime) else report_dict['estimated_completion']
            active_reports_json[k] = report_dict

        return {
            'system_status': self.system_status,
            'active_reports': active_reports_json,
            'queue_size': self.report_queue.qsize(),
            'recent_reports': self.get_recent_reports(10),
            'timestamp': datetime.now().isoformat()
        }

# Global status manager
status_manager = ReportStatusManager(config)

# Flask app setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'ert_status_dashboard_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")
status_manager.set_socketio(socketio)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    return jsonify(status_manager.get_status_summary())

@app.route('/health')
def health_check():
    """Health check endpoint for Render and other cloud platforms"""
    try:
        # Basic health check - ensure core components are working
        system_status = status_manager.get_status_summary()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'active_reports': len(system_status.get('active_reports', {})),
            'queue_size': system_status.get('queue_size', 0)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503

@app.route('/api/search_tickers')
def api_search_tickers():
    """API endpoint for searching tickers"""
    query = request.args.get('q', '').upper()
    if not query or len(query) < 1:
        return jsonify([])

    # In a real application, you would use a proper search index for tickers.
    # For this example, we'll just check if the query is a valid ticker.
    info = get_stock_info(query)
    if info:
        return jsonify([info])
    else:
        return jsonify([])

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint to start report generation"""
    data = request.json or {}
    ticker = data.get('ticker', '').strip().upper()

    if not ticker:
        return jsonify({'error': 'Ticker symbol required'}), 400

    info = get_stock_info(ticker)
    if not info:
        return jsonify({'error': f'Invalid ticker: {ticker}'}), 400

    try:
        report_id = status_manager.add_report_to_queue(ticker)
        return jsonify({
            'success': True,
            'report_id': report_id,
            'message': f'Report for {ticker} added to queue'
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        logger.error(f"Error adding report to queue: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/watchlist_prices', methods=['POST'])
def api_watchlist_prices():
    """API endpoint for fetching watchlist prices"""
    data = request.json or {}
    tickers = data.get('tickers', [])
    if not tickers:
        return jsonify({})

    prices = {}
    for ticker in tickers:
        info = get_stock_info(ticker)
        if info:
            prices[ticker] = {
                'price': info.get('price'),
                'price_change': info.get('price_change'),
                'percent_change': info.get('percent_change'),
            }
    return jsonify(prices)


@app.route('/api/abort', methods=['POST'])
def api_abort():
    """Abort all pending and active reports"""
    try:
        summary = status_manager.request_abort()
        return jsonify({'success': True, 'summary': summary})
    except Exception as exc:
        logger.error(f"Error processing abort request: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/reset_abort', methods=['POST'])
def api_reset_abort():
    """Reset abort state (admin function)"""
    try:
        success = status_manager.reset_abort_state()
        return jsonify({
            'success': success,
            'message': 'Abort state reset successfully' if success else 'Cannot reset - active reports exist'
        })
    except Exception as exc:
        logger.error(f"Error resetting abort state: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/reports')
def api_reports():
    """API endpoint for report history"""
    return jsonify(status_manager.get_recent_reports())

@app.route('/api/reports/<report_id>')
def api_report_detail(report_id):
    """API endpoint for specific report details"""
    # Check active reports
    if report_id in status_manager.active_reports:
        return jsonify({
            'status': 'active',
            'data': asdict(status_manager.active_reports[report_id])
        })

    # Check completed reports
    for report in status_manager.completed_reports:
        if report_id in report.get('report_path', ''):
            return jsonify({
                'status': 'completed',
                'data': report
            })

    return jsonify({'error': 'Report not found'}), 404

@app.route('/reports/<path:filename>')
def serve_report(filename):
    """Serve generated report files"""
    reports_dir = os.path.join(project_root, 'reports')
    return send_from_directory(reports_dir, filename)

@socketio.on('connect')
def handle_connect(auth=None):
    """Handle client connection"""
    logger.info('Client connected to status dashboard')
    emit('status_update', status_manager.get_status_summary())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected from status dashboard')

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    emit('status_update', status_manager.get_status_summary())

def check_abort(report_id: str):
    """Check if abort has been requested for this report and raise AbortSignal if so."""
    with status_manager.abort_lock:
        if status_manager.abort_active:
            raise AbortSignal("System abort active")

    if report_id in status_manager.active_reports:
        report = status_manager.active_reports[report_id]
        if getattr(report, 'abort_requested', False):
            raise AbortSignal(f"Abort requested for {report_id}")

    if status_manager.abort_event.is_set():
        raise AbortSignal("Global abort event set")

def report_generator_worker():
    """Background worker to process report generation queue"""
    while True:
        try:
            # Get next report from queue (blocking)
            report_id = status_manager.report_queue.get(timeout=1)

            if report_id not in status_manager.active_reports:
                continue

            report = status_manager.active_reports[report_id]
            ticker = report.ticker

            logger.info(f"Starting report generation for {ticker} (ID: {report_id})")

            try:
                check_abort(report_id)

                # Update status to fetching data
                status_manager.update_progress(
                    report_id,
                    status='fetching_data',
                    progress=10,
                    current_section='Fetching company data...'
                )

                check_abort(report_id)

                # Initialize generator if needed
                if not status_manager.generator:
                    status_manager.generator = EnhancedEquityResearchGenerator()

                check_abort(report_id)

                # Fetch company data
                status_manager.update_progress(
                    report_id,
                    progress=20,
                    current_section='Analyzing financial data...'
                )

                check_abort(report_id)

                status_manager.generator.fetch_comprehensive_data(ticker)

                sections = [
                    ('executive_summary', 'Generating executive summary...', 30),
                    ('market_research', 'Conducting market research...', 45),
                    ('financial_analysis', 'Analyzing financials...', 60),
                    ('valuation_analysis', 'Performing valuation analysis...', 75),
                    ('investment_thesis', 'Developing investment thesis...', 85),
                    ('risk_analysis', 'Assessing risks...', 95)
                ]

                status_manager.update_progress(
                    report_id,
                    status='generating_sections',
                    progress=25,
                    current_section='Starting AI analysis...'
                )

                for section_name, description, progress in sections:
                    check_abort(report_id)
                    status_manager.update_progress(
                        report_id,
                        progress=progress,
                        current_section=description
                    )
                    check_abort(report_id)
                    time.sleep(2)

                check_abort(report_id)

                status_manager.update_progress(
                    report_id,
                    progress=98,
                    current_section='Compiling final report...'
                )

                check_abort(report_id)

                report_content = status_manager.generator.generate_comprehensive_report(ticker)

                check_abort(report_id)

                if report_content:
                    word_count = len(report_content.split())

                    reports_dir = Path(project_root) / 'reports'
                    report_files = list(reports_dir.glob(f"{ticker}*.md"))

                    report_path = None
                    summary_payload = None
                    if report_files:
                        latest_file = max(report_files, key=lambda p: p.stat().st_mtime)
                        report_path = latest_file.name
                        summary_candidate = latest_file.with_name(latest_file.stem + "_summary.json")
                        if summary_candidate.exists():
                            try:
                                summary_payload = json.loads(summary_candidate.read_text())
                            except Exception as summary_exc:
                                logger.debug("Unable to parse summary metadata: %s", summary_exc)

                    status_manager.complete_report(
                        report_id,
                        success=True,
                        report_path=report_path,
                        metadata=summary_payload
                    )

                    status_manager.update_progress(
                        report_id,
                        word_count=word_count
                    )

                else:
                    status_manager.complete_report(report_id, success=False)

            except AbortSignal:
                logger.info(f"Abort requested for {ticker} (ID: {report_id})")
                status_manager.update_progress(
                    report_id,
                    status='aborted',
                    current_section='Report aborted by user',
                    error_message='Report aborted by user',
                    progress=report.progress
                )
                status_manager.complete_report(report_id, success=False)
                # Don't clear abort_event here - let reset_abort_state handle it
                continue
            except Exception as e:
                logger.error(f"Error generating report for {ticker}: {e}")
                status_manager.update_progress(
                    report_id,
                    error_message=str(e)
                )
                status_manager.complete_report(report_id, success=False)

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in report generator worker: {e}")
            time.sleep(5)

def start_background_worker():
    """Start the background worker thread"""
    worker_thread = threading.Thread(target=report_generator_worker, daemon=True)
    worker_thread.start()
    logger.info("Background report generator worker started")

def _ensure_directories():
    os.makedirs(os.path.join(project_root, 'reports'), exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

def create_app(config_override: Optional[StatusServerConfig] = None):
    """
    Application factory function for WSGI deployment
    """
    global app, socketio, status_manager, config

    # Use provided config or load from environment
    if config_override:
        config = config_override
        if not config.validate():
            raise ValueError("Invalid configuration provided")

    # Ensure directories exist
    _ensure_directories()

    # Start background worker
    start_background_worker()

    # Production configuration
    is_production = os.environ.get('FLASK_ENV') == 'production'
    if is_production:
        app.config['DEBUG'] = False
        app.config['SECRET_KEY'] = os.environ.get('ERT_SECRET_KEY', 'ert_production_secret_key')
        # Disable Ollama health checks in production for faster startup
        try:
            import src.utils.ollama_engine
            src.utils.ollama_engine.SKIP_HEALTH_CHECK = True
        except ImportError:
            pass

    return app


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="ERT Status Server")
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5001)), help='Port to run the status server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host interface to bind (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')

    args = parser.parse_args()

    _ensure_directories()

    start_background_worker()

    # Production configuration
    is_production = os.environ.get('FLASK_ENV') == 'production'
    if is_production:
        app.config['DEBUG'] = False
        # Disable Ollama health checks in production for faster startup
        import src.utils.ollama_engine
        src.utils.ollama_engine.SKIP_HEALTH_CHECK = True

    logger.info(f"Starting ERT Status Dashboard on http://{args.host}:{args.port}")
    try:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug and not is_production,
            allow_unsafe_werkzeug=True,
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise
