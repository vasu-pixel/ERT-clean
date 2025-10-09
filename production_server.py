#!/usr/bin/env python3
# production_server.py - Full-featured production server for Render
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import queue
from threading import RLock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with fallbacks for missing modules
try:
    import yfinance as yf
    from functools import lru_cache
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not available - stock data features disabled")
    YFINANCE_AVAILABLE = False

# Cache for stock info (5 minute TTL)
stock_info_cache = {}
CACHE_TTL = 300  # 5 minutes

def get_stock_info_from_fmp(ticker):
    """Fallback to FMP API when yfinance is rate limited"""
    fmp_api_key = os.getenv('FMP_API_KEY')
    if not fmp_api_key:
        return None

    try:
        import requests
        # Get quote
        quote_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={fmp_api_key}"
        response = requests.get(quote_url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                quote = data[0]
                return {
                    'ticker': ticker,
                    'name': quote.get('name', ticker),
                    'price': quote.get('price'),
                    'price_change': quote.get('change'),
                    'percent_change': quote.get('changesPercentage'),
                    'market_cap': quote.get('marketCap'),
                    'sector': quote.get('sector'),
                }
    except Exception as e:
        logger.error(f"FMP API error for {ticker}: {e}")

    return None

def _call_alpha_vantage_with_retry(url: str, max_retries: int = 3, timeout: int = 10) -> Optional[Dict[str, Any]]:
    import requests
    backoff = 2
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json() or {}
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.error(f"Alpha Vantage request failed after {max_retries} attempts: {exc}")
                return None
            sleep_time = backoff * (attempt + 1)
            logger.warning(f"Alpha Vantage request failed (attempt {attempt + 1}): {exc}; retrying in {sleep_time}s")
            time.sleep(sleep_time)
    return None


def get_stock_info_from_alpha_vantage(ticker: str) -> Optional[Dict[str, Any]]:
    """Secondary fallback using Alpha Vantage GLOBAL_QUOTE endpoint."""
    alpha_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not alpha_api_key:
        return None

    try:
        quote_url = (
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
            f"&symbol={ticker}&apikey={alpha_api_key}"
        )
        data = _call_alpha_vantage_with_retry(quote_url)
        if data is None:
            return None
        quote = data.get("Global Quote") or {}
        if not quote:
            logger.warning(f"Alpha Vantage returned no quote for {ticker}")
            return None

        def _safe_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        price = _safe_float(quote.get("05. price"))
        previous_close = _safe_float(quote.get("08. previous close"))
        price_change = _safe_float(quote.get("09. change"))
        percent_change_raw = quote.get("10. change percent")
        percent_change = None
        if isinstance(percent_change_raw, str) and percent_change_raw.endswith('%'):
            try:
                percent_change = float(percent_change_raw.strip('%'))
            except ValueError:
                percent_change = None
        else:
            percent_change = _safe_float(percent_change_raw)

        return {
            'ticker': ticker,
            'name': ticker,
            'price': price,
            'price_change': price_change,
            'percent_change': percent_change,
            'market_cap': None,
            'sector': None,
        }
    except Exception as exc:
        logger.error(f"Alpha Vantage API error for {ticker}: {exc}")
        return None

def get_stock_info(ticker):
    """Get stock info with caching and FMP as primary source"""
    # Check cache first
    cache_key = ticker
    stale_entry = None
    if cache_key in stock_info_cache:
        cached_data, timestamp = stock_info_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logger.info(f"Returning cached data for {ticker}")
            return cached_data
        stale_entry = (cached_data, timestamp)

    # Try FMP API first (more reliable, no rate limits with API key)
    logger.info(f"Fetching stock info for {ticker} from FMP API")
    result = get_stock_info_from_fmp(ticker)
    if result:
        stock_info_cache[cache_key] = (result, time.time())
        return result

    # Secondary fallback: Alpha Vantage
    logger.info(f"FMP failed, trying Alpha Vantage for {ticker}")
    av_result = get_stock_info_from_alpha_vantage(ticker)
    if av_result:
        stock_info_cache[cache_key] = (av_result, time.time())
        return av_result

    # Fallback to yfinance if FMP fails
    if not YFINANCE_AVAILABLE:
        return None

    try:
        logger.info(f"FMP failed, trying yfinance for {ticker}")
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

        result = {
            'ticker': ticker,
            'name': info.get('shortName'),
            'price': price,
            'price_change': price_change,
            'percent_change': percent_change,
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
        }

        # Cache the result
        stock_info_cache[cache_key] = (result, time.time())
        return result

    except Exception as e:
        logger.error(f"FMP, Alpha Vantage, and yfinance failed for {ticker}: {e}")
        if stale_entry:
            logger.warning(f"Serving stale cached data for {ticker} due to upstream rate limits")
            stock_info_cache[cache_key] = (stale_entry[0], time.time())
            return stale_entry[0]
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

    def __init__(self):
        self.active_reports: Dict[str, ReportProgress] = {}
        self.completed_reports: List[Dict] = []
        self.report_queue = queue.Queue()
        self.socketio = None
        self.generator = None
        self.system_status = {}
        self.abort_event = threading.Event()
        self.abort_lock = threading.Lock()
        self.abort_active = False
        self._lock = RLock()
        self.update_system_status()

    def set_socketio(self, socketio):
        """Set SocketIO instance for real-time updates"""
        self.socketio = socketio

    def get_report(self, report_id: str) -> Optional[ReportProgress]:
        """Thread-safe access to a report."""
        with self._lock:
            return self.active_reports.get(report_id)

    def update_system_status(self):
        """Update system status information"""
        try:
            self.system_status = {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'features': {
                    'yfinance': YFINANCE_AVAILABLE,
                    'report_generation': True,
                    'real_time_updates': True
                }
            }
            with self._lock:
                self.system_status['active_reports'] = len(self.active_reports)
            self.system_status['queue_size'] = self.report_queue.qsize()
        except Exception as e:
            logger.error(f"Error updating system status: {e}")

    def add_report_to_queue(self, ticker: str) -> str:
        """Add a report to the generation queue"""
        ticker = ticker.upper()

        if not ticker or len(ticker) > 12:
            raise ValueError('Invalid ticker symbol')

        report_id = f"{ticker}_{int(time.time())}"

        progress = ReportProgress(
            ticker=ticker.upper(),
            status='queued',
            progress=0,
            current_section='Waiting in queue',
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(minutes=5)
        )

        with self._lock:
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
        logger.info(
            "update_progress report_id=%s status=%s progress=%s section=%s",
            report_id,
            status,
            progress,
            current_section,
        )

        with self._lock:
            report = self.active_reports.get(report_id)
            if not report:
                return

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
            with self._lock:
                report = self.active_reports.get(report_id)
            if report:
                report.abort_requested = True
                report.status = 'aborted'
                report.progress = 0
                report.current_section = 'Aborted before start'
                report.error_message = 'Aborted before start'
                self.complete_report(report_id, success=False)

        with self._lock:
            active_snapshot = list(self.active_reports.items())

        for report_id, report in active_snapshot:
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
            with self._lock:
                no_active = len(self.active_reports) == 0
            if no_active:
                self.abort_active = False
                self.abort_event.clear()
                logger.info("Abort state reset - system ready for new reports")
                return True
            else:
                logger.warning("Cannot reset abort state - active reports still exist")
                return False

    def complete_report(self, report_id: str, success: bool = True, report_path: str = None, metadata: Optional[Dict] = None):
        """Mark a report as completed"""
        with self._lock:
            report = self.active_reports.get(report_id)
            if not report:
                logger.info("complete_report called for missing report_id=%s", report_id)
                return

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

            completed_data = asdict(report)
            if 'start_time' in completed_data and isinstance(report.start_time, datetime):
                completed_data['start_time'] = report.start_time.isoformat()
            if 'estimated_completion' in completed_data and report.estimated_completion and isinstance(report.estimated_completion, datetime):
                completed_data['estimated_completion'] = report.estimated_completion.isoformat()
            completed_data['completion_time'] = datetime.now().isoformat()
            if metadata:
                completed_data['report_metadata'] = metadata
            completed_data['report_id'] = report_id

            self.completed_reports.append(completed_data)
            if report_id in self.active_reports:
                del self.active_reports[report_id]

        logger.info(
            "complete_report report_id=%s success=%s path=%s",
            report_id,
            success,
            report_path,
        )

        # Check if we can reset abort state
        if self.abort_active:
            with self._lock:
                no_active = len(self.active_reports) == 0
            if no_active:
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
        with self._lock:
            snapshot = list(self.completed_reports)
        return sorted(snapshot, key=lambda x: x['start_time'], reverse=True)[:limit]

    def get_status_summary(self) -> Dict:
        """Get overall status summary"""
        self.update_system_status()

        # Convert active reports to JSON-serializable format
        active_reports_json = {}
        with self._lock:
            items = list(self.active_reports.items())
        for k, v in items:
            report_dict = asdict(v)
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
status_manager = ReportStatusManager()

# Flask app setup
app = Flask(__name__, template_folder='src/ui/templates', static_folder='src/ui/static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'minimal-animals-production-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
status_manager.set_socketio(socketio)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    return jsonify(status_manager.get_status_summary())

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

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'yfinance': YFINANCE_AVAILABLE,
            'dashboard': True
        }
    })

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

    report = status_manager.get_report(report_id)
    if report and getattr(report, 'abort_requested', False):
        raise AbortSignal(f"Abort requested for {report_id}")

    if status_manager.abort_event.is_set():
        raise AbortSignal("Global abort event set")

def real_report_generator_worker():
    """Real background worker using Remote LLM for report generation"""
    # Import the real generator (moved here to avoid circular import at module level)
    try:
        import sys
        import importlib

        # Import remote LLM client
        from src.utils.remote_llm_client import RemoteEquityResearchGenerator
        generator = RemoteEquityResearchGenerator()

        # Import data orchestrator with error handling for circular imports
        try:
            from src.data_pipeline.orchestrator import DataOrchestrator
            data_orchestrator = DataOrchestrator()
        except ImportError as import_err:
            logger.warning(f"DataOrchestrator import failed (circular import): {import_err}")
            logger.info("Using simplified data fetching without DataOrchestrator")
            data_orchestrator = None

        logger.info("Remote LLM generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Remote LLM generator: {e}")
        logger.warning("Falling back to mock generator")
        import traceback
        traceback.print_exc()
        # Fallback to mock if import fails
        mock_report_generator_worker()
        return

    while True:
        try:
            # Get next report from queue (blocking)
            report_id = status_manager.report_queue.get(timeout=1)

            report = status_manager.get_report(report_id)
            if not report:
                continue
            ticker = report.ticker

            logger.info(f"Starting real report generation for {ticker} (ID: {report_id})")

            try:
                check_abort(report_id)

                # Phase 1: Fetching real data
                status_manager.update_progress(
                    report_id,
                    status='fetching_data',
                    progress=10,
                    current_section='Fetching company data and financials...'
                )

                logger.info(f"Fetching comprehensive data for {ticker}")

                if data_orchestrator:
                    company_dataset = data_orchestrator.refresh_company_data(ticker)
                else:
                    # Simplified fallback - use basic yfinance data
                    logger.info("Using simplified data fetching (DataOrchestrator unavailable)")
                    import yfinance as yf
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    # Create minimal dataset
                    class SimpleDataset:
                        def __init__(self, info):
                            self.snapshot = type('obj', (object,), {
                                'name': info.get('longName', ticker),
                                'sector': info.get('sector', 'Unknown'),
                                'industry': info.get('industry', 'Unknown'),
                                'market_cap': info.get('marketCap'),
                                'current_price': info.get('currentPrice')
                            })()
                            self.financials = type('obj', (object,), {
                                'fundamentals': {k: v for k, v in info.items() if isinstance(v, (int, float))},
                                'ratios': {},
                                'price_history': None
                            })()
                            self.supplemental = {}

                    company_dataset = SimpleDataset(info)

                check_abort(report_id)

                # Phase 2: Preparing data for LLM
                status_manager.update_progress(
                    report_id,
                    status='preparing_data',
                    progress=25,
                    current_section='Preparing data for analysis...'
                )

                # Extract and structure data for the LLM
                company_data = {
                    'ticker': ticker,
                    'name': company_dataset.snapshot.name,
                    'sector': company_dataset.snapshot.sector,
                    'industry': company_dataset.snapshot.industry,
                    'market_cap': company_dataset.snapshot.market_cap,
                    'current_price': company_dataset.snapshot.current_price,
                }

                financial_data = {
                    'fundamentals': company_dataset.financials.fundamentals,
                    'ratios': company_dataset.financials.ratios,
                    'price_history': company_dataset.financials.price_history,
                }

                market_data = {
                    'analyst_estimates': company_dataset.supplemental.get('analyst_estimates', {}),
                    'recent_headlines': company_dataset.supplemental.get('recent_headlines', []),
                    'headline_sentiment': company_dataset.supplemental.get('headline_sentiment', {}),
                    'peer_metrics': company_dataset.supplemental.get('peer_metrics', {}),
                }

                # Set data in generator
                generator.ticker = ticker
                generator.set_company_data(company_data)
                generator.set_financial_data(financial_data)
                generator.set_market_data(market_data)

                check_abort(report_id)

                # Phase 3: Generating report with AI
                status_manager.update_progress(
                    report_id,
                    status='generating_sections',
                    progress=40,
                    current_section='Generating AI analysis with real data...'
                )

                # Generate the actual report
                logger.info(f"Generating report for {ticker} with real data")
                report_content = generator.generate_comprehensive_report(ticker=ticker)

                check_abort(report_id)

                # Phase 3: Saving report
                status_manager.update_progress(
                    report_id,
                    status='finalizing',
                    progress=90,
                    current_section='Saving report...'
                )

                # Save report to file
                reports_dir = Path(project_root) / 'reports'
                reports_dir.mkdir(exist_ok=True)
                report_filename = f"{ticker}_{int(time.time())}.md"
                report_path = reports_dir / report_filename

                with open(report_path, 'w') as f:
                    f.write(report_content)

                logger.info(f"Report saved to {report_path}")

                # Complete successfully
                status_manager.complete_report(
                    report_id,
                    success=True,
                    report_path=str(report_path)
                )

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
                continue
            except Exception as e:
                logger.error(f"Error generating report for {ticker}: {e}")
                import traceback
                traceback.print_exc()
                status_manager.update_progress(
                    report_id,
                    error_message=str(e)
                )
                status_manager.complete_report(report_id, success=False)

        except queue.Empty:
            continue
        except Exception as e:
            logger.exception(f"Error in report generator worker: {e}")
            time.sleep(5)
        except Exception as exc:  # should be unreachable
            logger.exception(f"Unhandled error in real report generator worker: {exc}")

def mock_report_generator_worker():
    """Mock background worker for demo purposes (fallback)"""
    while True:
        try:
            # Get next report from queue (blocking)
            report_id = status_manager.report_queue.get(timeout=1)

            report = status_manager.get_report(report_id)
            if not report:
                continue
            ticker = report.ticker

            logger.info(f"Starting mock report generation for {ticker} (ID: {report_id})")

            try:
                check_abort(report_id)

                # Mock report generation with progress updates
                stages = [
                    ('fetching_data', 'Fetching company data...', 20),
                    ('analyzing', 'Analyzing financial metrics...', 40),
                    ('generating', 'Generating insights...', 60),
                    ('formatting', 'Formatting report...', 80),
                    ('finalizing', 'Finalizing report...', 95)
                ]

                for stage, description, progress in stages:
                    check_abort(report_id)
                    status_manager.update_progress(
                        report_id,
                        status=stage,
                        progress=progress,
                        current_section=description
                    )
                    time.sleep(2)  # Simulate work

                check_abort(report_id)

                # Complete successfully
                status_manager.complete_report(
                    report_id,
                    success=True,
                    report_path=f"mock_report_{ticker}.md"
                )

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
                continue
            except Exception as e:
                logger.error(f"Error generating mock report for {ticker}: {e}")
                status_manager.update_progress(
                    report_id,
                    error_message=str(e)
                )
                status_manager.complete_report(report_id, success=False)

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in mock report generator worker: {e}")
            time.sleep(5)

def start_background_worker():
    """Start the background worker thread"""
    worker_thread = threading.Thread(target=real_report_generator_worker, daemon=True)
    worker_thread.start()
    logger.info("Background real report generator worker started")

def _ensure_directories():
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    templates_dir = project_root / 'src' / 'ui' / 'templates'
    static_dir = project_root / 'src' / 'ui' / 'static'

    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # Ensure directories exist
    _ensure_directories()

    # Start background worker
    start_background_worker()

    # Get port from environment (Render sets this)
    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'

    # Production configuration
    is_production = os.environ.get('FLASK_ENV') == 'production'
    if is_production:
        app.config['DEBUG'] = False

    logger.info(f"Starting Minimal Animals Equity Research Tool on {host}:{port}")
    logger.info(f"Production mode: {is_production}")
    logger.info(f"YFinance available: {YFINANCE_AVAILABLE}")

    socketio.run(
        app,
        host=host,
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
