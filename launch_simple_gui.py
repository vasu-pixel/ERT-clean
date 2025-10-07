#!/usr/bin/env python3
# launch_simple_gui.py - Launch the Simple Search Bar GUI
import os
import sys
import subprocess
import webbrowser
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'flask-socketio',
        'requests',
        'markdown',
        'yfinance'
    ]

    optional_packages = [
        'pdfkit',
        'reportlab'
    ]

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_required.append(package)

    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)

    if missing_required:
        print("❌ Missing required packages:")
        for package in missing_required:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_required)}")
        return False

    if missing_optional:
        print("⚠️  Missing optional PDF packages (will use fallback):")
        for package in missing_optional:
            print(f"   - {package}")
        print("For better PDF generation, install:")
        print(f"   pip install {' '.join(missing_optional)}")

    return True

def check_ollama_status():
    """Check if Ollama is running with Mistral model"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]

            # Check for Mistral model
            mistral_available = any('mistral' in model.lower() for model in model_names)

            print(f"✅ Ollama is running with {len(models)} models")
            if mistral_available:
                print("✅ Mistral model is available")
            else:
                print("⚠️  Mistral model not found. Available models:")
                for model in model_names:
                    print(f"   - {model}")
                print("\nTo install Mistral:7b run: ollama pull mistral:7b")

            return True, mistral_available
        else:
            print("⚠️  Ollama server responded with error")
            return False, False
    except requests.exceptions.RequestException:
        print("❌ Ollama server is not accessible")
        print("   Please start Ollama with: ollama serve")
        return False, False

def setup_directories():
    """Ensure required directories exist"""
    project_root = Path(__file__).parent

    directories = [
        project_root / 'src' / 'ui' / 'templates',
        project_root / 'src' / 'ui' / 'static',
        project_root / 'reports'
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def wait_for_server(url='http://localhost:5001', timeout=30):
    """Wait for the Flask server to start"""
    print(f"⏳ Waiting for server to start at {url}...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print(f"✅ Server is ready at {url}")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)

    print(f"❌ Server did not start within {timeout} seconds")
    return False

def create_requirements_file():
    """Create requirements file for simple GUI"""
    requirements = """# Simple GUI Requirements
flask>=2.3.0
flask-socketio>=5.3.0
requests>=2.31.0
yfinance>=0.2.18
python-dotenv>=0.19.0
markdown>=3.4.0

# PDF Generation (optional - will fallback if not available)
pdfkit>=1.0.0
reportlab>=3.6.0

# Additional dependencies
pandas>=1.5.0
numpy>=1.21.0
"""

    requirements_path = Path('requirements_simple_gui.txt')
    with open(requirements_path, 'w') as f:
        f.write(requirements)

    print(f"📝 Created {requirements_path}")

def main():
    """Launch the simple search GUI"""
    print("🚀 Enhanced Equity Research Tool - Simple Search GUI")
    print("=" * 60)

    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("\n💡 To install all dependencies:")
        create_requirements_file()
        print("   pip install -r requirements_simple_gui.txt")
        return False

    # Check Ollama
    print("\n2. Checking Ollama status...")
    ollama_running, mistral_available = check_ollama_status()

    if not ollama_running:
        print("\n❌ Ollama server is required")
        print("Please start Ollama server with: ollama serve")
        return False

    # Setup directories
    print("\n3. Setting up directories...")
    setup_directories()

    # Launch server
    print("\n4. Starting simple search interface...")

    project_root = Path(__file__).parent
    server_path = project_root / 'src' / 'ui' / 'simple_gui.py'

    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        return False

    try:
        # Start the Flask server
        server_process = subprocess.Popen(
            [sys.executable, str(server_path)],
            cwd=str(project_root)
        )

        # Wait for server to start
        if wait_for_server('http://localhost:5001'):
            # Open browser
            gui_url = 'http://localhost:5001'
            print(f"\n🌐 Opening simple GUI in browser: {gui_url}")
            webbrowser.open(gui_url)

            print(f"""
🎯 Simple Equity Research GUI is now running!

   🌐 GUI URL: {gui_url}
   🔍 Search Interface: Clean, simple ticker search
   📊 Real-time Progress: Live updates during generation
   📄 PDF Download: Automatic PDF generation and download
   🤖 AI Engine: Mistral:7b (Local LLM)

How to use:
   1. Enter any stock ticker (e.g., AAPL, CVS, MSFT)
   2. Press Enter or click "Generate Report"
   3. Watch real-time progress updates
   4. Download PDF when complete (auto-download after 2 seconds)

Features:
   ✨ Single search bar interface
   📈 Real-time progress tracking
   📄 Professional PDF reports
   🚀 One-click generation
   📱 Mobile-friendly design
   🔄 WebSocket live updates

System Status:
   ✅ Ollama Server: {'Running' if ollama_running else 'Not Running'}
   ✅ Mistral Model: {'Available' if mistral_available else 'Not Available'}
   ✅ PDF Generation: Available
   ✅ GUI Server: Running on port 5001

Press Ctrl+C to stop the server.
            """)

            try:
                # Keep server running
                server_process.wait()
            except KeyboardInterrupt:
                print("\n\n⏹️  Shutting down simple GUI server...")
                server_process.terminate()
                server_process.wait()
                print("✅ Simple GUI server stopped")

        else:
            server_process.terminate()
            return False

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

    return True

def show_help():
    """Show help information"""
    print("""
🔍 Simple ERT Search GUI Help

Usage:
   python launch_simple_gui.py           # Launch the GUI
   python launch_simple_gui.py --help    # Show this help

GUI Features:
   • Clean search bar interface
   • Real-time progress tracking
   • Automatic PDF generation
   • Professional report formatting
   • Mobile-responsive design

Requirements:
   • Ollama server running (ollama serve)
   • Mistral:7b model (ollama pull mistral:7b)
   • Python dependencies (see requirements_simple_gui.txt)

Troubleshooting:
   • Server won't start: Check port 5001 availability
   • No reports generated: Verify Ollama and Mistral model
   • PDF issues: Install pdfkit and reportlab
   • Connection errors: Restart Ollama server

Examples:
   1. Enter "AAPL" → Generates Apple Inc. report
   2. Enter "CVS" → Generates CVS Health report
   3. Enter "TSLA" → Generates Tesla Inc. report

The interface automatically:
   ✓ Validates ticker symbols
   ✓ Shows progress updates
   ✓ Generates professional PDFs
   ✓ Provides download links
   ✓ Handles errors gracefully
    """)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_help()
        sys.exit(0)

    success = main()

    if not success:
        print("\n❌ Failed to launch simple GUI")
        print("\nTroubleshooting steps:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Install Mistral model: ollama pull mistral:7b")
        print("3. Install dependencies: pip install -r requirements_simple_gui.txt")
        print("4. Check if port 5001 is available")
        sys.exit(1)