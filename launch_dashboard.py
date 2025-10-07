#!/usr/bin/env python3
# launch_dashboard.py - Launch the ERT Status Dashboard
import os
import sys
import json
import subprocess
import webbrowser
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

# Configuration
@dataclass
class DashboardConfig:
    """Configuration for dashboard launcher"""
    server_timeout: int = 90  # Configurable server timeout
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    check_timeout: int = 5
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> 'DashboardConfig':
        """Create config from environment variables"""
        return cls(
            server_timeout=int(os.getenv('ERT_SERVER_TIMEOUT', '90')),
            ollama_host=os.getenv('ERT_OLLAMA_HOST', 'localhost'),
            ollama_port=int(os.getenv('ERT_OLLAMA_PORT', '11434')),
            check_timeout=int(os.getenv('ERT_CHECK_TIMEOUT', '5')),
            log_level=os.getenv('ERT_LOG_LEVEL', 'INFO')
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'flask-socketio',
        'requests'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True

def check_ollama_status(config: DashboardConfig) -> bool:
    """Check if Ollama is running and accessible"""
    ollama_url = f"http://{config.ollama_host}:{config.ollama_port}/api/tags"

    try:
        logger.info(f"Checking Ollama status at {ollama_url}")
        response = requests.get(ollama_url, timeout=config.check_timeout)

        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running with {len(models)} models available")
            logger.info(f"Ollama accessible with {len(models)} models")
            return True
        else:
            print(f"⚠️  Ollama server responded with status {response.status_code}")
            logger.warning(f"Ollama responded with status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Ollama server is not accessible at {ollama_url}")
        print("   Please start Ollama with: ollama serve")
        logger.error(f"Cannot connect to Ollama at {ollama_url}")
        return False

    except requests.exceptions.Timeout:
        print(f"⚠️  Ollama server timeout after {config.check_timeout}s")
        logger.warning(f"Ollama timeout after {config.check_timeout}s")
        return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking Ollama: {e}")
        logger.error(f"Ollama check error: {e}")
        return False

def setup_directories():
    """Ensure required directories exist"""
    project_root = Path(__file__).parent

    directories = [
        project_root / 'src' / 'ui' / 'templates',
        project_root / 'src' / 'ui' / 'static',
        project_root / 'reports',
        project_root / 'logs'
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory}")

def wait_for_server(url: str, timeout: int, process: Optional[subprocess.Popen] = None) -> bool:
    """Wait for the Flask server to start with enhanced subprocess monitoring."""
    print(f"⏳ Waiting for server to start at {url} (timeout: {timeout}s)...")
    logger.info(f"Waiting for server at {url} with {timeout}s timeout")

    start_time = time.time()
    last_log = 0
    server_logs = []

    while time.time() - start_time < timeout:
        # Check if process has exited
        if process and process.poll() is not None:
            return_code = process.returncode
            print(f"❌ Dashboard server exited early with code {return_code}")
            logger.error(f"Server process exited with code {return_code}")

            # Enhanced subprocess output handling
            try:
                # Read any remaining output
                if process.stdout:
                    stdout_lines = process.stdout.read().decode('utf-8', errors='ignore').strip().split('\n')
                    server_logs.extend([f"STDOUT: {line}" for line in stdout_lines if line])

                if process.stderr:
                    stderr_lines = process.stderr.read().decode('utf-8', errors='ignore').strip().split('\n')
                    server_logs.extend([f"STDERR: {line}" for line in stderr_lines if line])

                if server_logs:
                    print("\n📋 Server output:")
                    for log_line in server_logs[-10:]:  # Show last 10 lines
                        print(f"   {log_line}")
                        logger.error(f"Server log: {log_line}")
                else:
                    print("   No server output captured")

            except Exception as e:
                print(f"   Error reading server output: {e}")
                logger.error(f"Error reading server output: {e}")

            return False

        # Check server availability
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is ready at {url}")
                logger.info(f"Server ready at {url}")
                return True
        except requests.exceptions.RequestException:
            pass

        # Progress feedback with dynamic intervals
        elapsed = time.time() - start_time
        feedback_interval = 5 if elapsed < 30 else 10  # More frequent initially

        if elapsed - last_log >= feedback_interval:
            remaining = max(0, int(timeout - elapsed))
            print(f"   … still waiting ({int(elapsed)}s elapsed, ~{remaining}s remaining)")
            logger.debug(f"Server startup progress: {int(elapsed)}s elapsed")
            last_log = elapsed

        time.sleep(1)

    print(f"❌ Server did not start within {timeout} seconds")
    logger.error(f"Server startup timeout after {timeout}s")

    if process and process.poll() is None:
        print("   ℹ️ The server process is still running; it may need more time.")
        logger.info("Server process still running after timeout")

    return False

def find_latest_report_summary(ticker: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Return latest report summary path and payload for ticker."""
    reports_dir = Path('reports')
    if not reports_dir.exists():
        return None, None

    summaries = sorted(
        reports_dir.glob(f"{ticker}_*_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        return None, None

    summary_path = summaries[0]
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        payload = None
    return summary_path, payload


def generate_report(ticker: str, engine: str = 'ollama', create_pdf: bool = False) -> bool:
    """Generate a comprehensive report (and optional PDF) for a ticker."""
    ticker = ticker.upper()
    print(f"\n🚀 Generating report for {ticker} using {engine} engine...")

    # Add src to path if needed (though this should be handled by proper imports)
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        logger.debug(f"Added {src_path} to sys.path")

    try:
        if engine == 'openai':
            from src.utils.openai_engine import OpenAIEngine as Engine
        else:
            from src.utils.ollama_engine import OllamaEngine as Engine

        from src.stock_report_generator import StockReportGenerator

        ai_engine = Engine()
        generator = StockReportGenerator(ai_engine)
        generator.generate_comprehensive_report(ticker)

    except ImportError as exc:
        print(f"❌ Import error - missing dependency: {exc}")
        logger.error(f"Import error during report generation: {exc}")
        return False
    except ConnectionError as exc:
        print(f"❌ Connection error - check Ollama/OpenAI connectivity: {exc}")
        logger.error(f"Connection error during report generation: {exc}")
        return False
    except FileNotFoundError as exc:
        print(f"❌ File not found: {exc}")
        logger.error(f"File not found during report generation: {exc}")
        return False
    except Exception as exc:
        print(f"❌ Report generation failed: {exc}")
        logger.error(f"Unexpected error during report generation: {exc}")
        return False

    summary_path, summary = find_latest_report_summary(ticker)
    if not summary_path or not summary:
        print("⚠️ Report generated but summary could not be located.")
        return not create_pdf

    markdown_file = summary.get('markdown_file')
    print(f"✅ Report generated: {markdown_file}")

    if create_pdf and markdown_file:
        try:
            from src.report.pdf_writer import convert_markdown_to_pdf

            # Check if pandoc or other conversion tool is available
            import shutil
            if not shutil.which('pandoc'):
                print("⚠️ PDF conversion requires pandoc. Install with: brew install pandoc (macOS) or apt install pandoc (Ubuntu)")
                logger.warning("Pandoc not found for PDF conversion")
                return True  # Report generation succeeded even without PDF

            pdf_file = Path(markdown_file).with_suffix('.pdf')
            logger.info(f"Converting {markdown_file} to PDF")

            if convert_markdown_to_pdf(markdown_file, str(pdf_file)):
                print(f"📄 PDF created: {pdf_file}")
                logger.info(f"PDF created successfully: {pdf_file}")
            else:
                print("⚠️ PDF conversion reported a failure.")
                logger.warning("PDF conversion failed")

        except ImportError as exc:
            print(f"⚠️ PDF conversion skipped (missing dependency): {exc}")
            logger.warning(f"PDF conversion skipped - missing dependency: {exc}")
        except Exception as exc:
            print(f"⚠️ PDF conversion error: {exc}")
            logger.error(f"PDF conversion error: {exc}")

    return True


def launch_dashboard(dev_mode: bool = False, port: int = 5001, open_browser: bool = True, config: Optional[DashboardConfig] = None) -> bool:
    """Launch the ERT Status Dashboard with enhanced configuration"""

    if config is None:
        config = DashboardConfig.from_env()

    # Set logging level from config
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))

    print("🚀 Enhanced Equity Research Tool - Status Dashboard Launcher")
    print("=" * 60)
    logger.info(f"Starting dashboard launcher on port {port} (dev_mode={dev_mode})")

    # Check dependencies
    print("\n1. Checking Python dependencies...")
    if not check_dependencies():
        logger.error("Dependency check failed")
        return False

    # Check Ollama status
    print("\n2. Checking Ollama status...")
    ollama_running = check_ollama_status(config)
    if not ollama_running:
        print("   ⚠️  Dashboard will work but reports cannot be generated")
        logger.warning("Ollama not available - report generation disabled")

    # Setup directories
    print("\n3. Setting up directories...")
    setup_directories()

    # Launch server
    print(f"\n4. Starting dashboard server on port {port}...")

    # Set environment variables
    env = os.environ.copy()
    env['FLASK_ENV'] = 'development' if dev_mode else 'production'
    env['FLASK_DEBUG'] = '1' if dev_mode else '0'

    # Get the path to the status server
    project_root = Path(__file__).parent
    server_path = project_root / 'src' / 'ui' / 'status_server.py'

    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        return False

    try:
        # Start the Flask server in background
        server_cmd = [sys.executable, str(server_path), '--port', str(port)]
        if dev_mode:
            server_cmd.append('--debug')

        server_process = subprocess.Popen(
            server_cmd,
            env=env,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to start with configurable timeout
        if wait_for_server(f'http://localhost:{port}', timeout=config.server_timeout, process=server_process):
            dashboard_url = f'http://localhost:{port}'
            if open_browser:
                print(f"\n🌐 Opening dashboard in browser: {dashboard_url}")
                webbrowser.open(dashboard_url)
            else:
                print(f"\n🌐 Dashboard running at: {dashboard_url}")

            print(f"""
📊 ERT Status Dashboard is now running!

   🌐 Dashboard URL: {dashboard_url}
   🔧 API Endpoint: {dashboard_url}/api/status
   📁 Reports Directory: {project_root}/reports
   📝 Logs Directory: {project_root}/logs

Features:
   ✨ Real-time report generation monitoring
   📈 System status and health checks
   🚀 One-click report generation
   📊 Progress tracking with live updates
   📁 Report management and downloads
   🔄 Queue management

Usage:
   1. Enter a ticker symbol (e.g., AAPL, MSFT, GOOGL)
   2. Click "Generate Report" or use quick buttons
   3. Monitor progress in real-time
   4. Download completed reports

System Requirements:
   ✅ Ollama running: {'Yes' if ollama_running else 'No (start with: ollama serve)'}
   ✅ Python dependencies: Installed
   ✅ Dashboard server: Running

Press Ctrl+C to stop the dashboard server.
            """)

            try:
                # Keep the server running
                server_process.wait()
            except KeyboardInterrupt:
                print("\n\n⏹️  Shutting down dashboard server...")
                server_process.terminate()
                server_process.wait()
                print("✅ Dashboard server stopped")

        else:
            server_process.terminate()
            return False

    except FileNotFoundError as e:
        print(f"❌ Server file not found: {e}")
        logger.error(f"Server file not found: {e}")
        return False
    except PermissionError as e:
        print(f"❌ Permission error starting server: {e}")
        logger.error(f"Permission error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        logger.error(f"Unexpected error starting server: {e}")
        return False

    return True

def main():
    """Main function with command line argument handling"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch ERT Status Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_dashboard.py                    # Launch in production mode
  python launch_dashboard.py --dev             # Launch in development mode
  python launch_dashboard.py --port 8080       # Launch on custom port
  python launch_dashboard.py --check-only      # Only check system status
  python launch_dashboard.py --generate AAPL   # Generate report (and then launch dashboard)
        """
    )

    parser.add_argument(
        '--dev',
        action='store_true',
        help='Run in development mode with debug enabled'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5001,
        help='Port to run the dashboard on (default: 5001)'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check system status and exit'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )

    parser.add_argument(
        '--generate',
        metavar='TICKER',
        help='Generate a report for the specified ticker before launching'
    )

    parser.add_argument(
        '--engine',
        choices=['ollama', 'openai'],
        default='ollama',
        help='AI engine to use when generating reports (default: ollama)'
    )

    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Convert the generated report to PDF as well'
    )

    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Generate the report and exit without launching the dashboard'
    )

    args = parser.parse_args()

    # Optionally generate a report prior to launching
    if args.generate:
        setup_directories()
        if not check_dependencies():
            sys.exit(1)
        config = DashboardConfig.from_env()
        if args.engine == 'ollama':
            check_ollama_status(config)

        if not generate_report(args.generate, engine=args.engine, create_pdf=args.pdf):
            sys.exit(1)

        if args.generate_only:
            return

    if args.check_only:
        print("🔍 System Status Check")
        print("=" * 30)

        print("Python dependencies:", "✅" if check_dependencies() else "❌")
        print("Ollama server:", "✅" if check_ollama_status() else "❌")

        # Check if dashboard files exist
        project_root = Path(__file__).parent
        required_files = [
            project_root / 'src' / 'ui' / 'status_server.py',
            project_root / 'src' / 'ui' / 'templates' / 'dashboard.html'
        ]

        files_exist = all(f.exists() for f in required_files)
        print("Dashboard files:", "✅" if files_exist else "❌")

        if not files_exist:
            print("Missing files:")
            for f in required_files:
                if not f.exists():
                    print(f"   - {f}")

        return

    # Launch dashboard with configuration
    config = DashboardConfig.from_env()
    success = launch_dashboard(
        dev_mode=args.dev,
        port=args.port,
        open_browser=not args.no_browser,
        config=config
    )

    if not success:
        print("\n❌ Failed to launch dashboard")
        print("\nTroubleshooting:")
        print("1. Ensure all Python dependencies are installed:")
        print("   pip install flask flask-socketio requests")
        print("2. Check if port is already in use")
        print("3. Verify Ollama is installed and running:")
        print("   ollama serve")
        print("4. Check file permissions and paths")

        sys.exit(1)

if __name__ == '__main__':
    main()
