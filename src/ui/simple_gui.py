# src/ui/simple_gui.py - Simplified Search Bar GUI for ERT
import os
import sys
import json
import time
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
import logging
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import queue

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.stock_report_generator import StockReportGenerator
from src.utils.ollama_engine import OllamaEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleReportGenerator:
    """Simplified report generator with PDF output"""

    def __init__(self):
        # Initialize the AI engine
        ai_engine = OllamaEngine()
        self.generator = StockReportGenerator(ai_engine)
        self.current_status = {
            'status': 'idle',
            'progress': 0,
            'message': 'Ready to generate reports',
            'ticker': '',
            'pdf_path': None,
            'error': None
        }
        self.socketio = None
        self.abort_flag = False
        self.current_thread = None

    def set_socketio(self, socketio):
        """Set SocketIO for real-time updates"""
        self.socketio = socketio

    def update_status(self, status=None, progress=None, message=None, ticker=None, pdf_path=None, error=None):
        """Update current status and emit to clients"""
        if status: self.current_status['status'] = status
        if progress is not None: self.current_status['progress'] = progress
        if message: self.current_status['message'] = message
        if ticker: self.current_status['ticker'] = ticker
        if pdf_path: self.current_status['pdf_path'] = pdf_path
        if error: self.current_status['error'] = error

        # Emit update to connected clients
        if self.socketio:
            self.socketio.emit('status_update', self.current_status)

    def abort_generation(self):
        """Abort current report generation"""
        self.abort_flag = True
        self.update_status(
            status='aborted',
            progress=0,
            message='Report generation aborted by user',
            error='Generation aborted'
        )
        logger.info("Report generation aborted by user")

    def generate_report_with_pdf(self, ticker):
        """Generate report and convert to PDF"""
        try:
            # Reset abort flag
            self.abort_flag = False

            self.update_status(
                status='starting',
                progress=5,
                message=f'Starting report generation for {ticker}...',
                ticker=ticker,
                error=None
            )

            # Check for abort
            if self.abort_flag:
                return None

            # Validate ticker
            self.update_status(progress=10, message='Validating ticker symbol...')
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            if 'longName' not in info or info.get('longName') is None:
                raise ValueError(f"Invalid ticker symbol: {ticker}")

            company_name = info.get('longName', ticker)
            self.update_status(progress=15, message=f'Generating report for {company_name}...')

            # Check for abort
            if self.abort_flag:
                return None

            # Generate markdown report
            self.update_status(progress=20, message='Fetching company data...')
            time.sleep(1)  # Brief pause for UI update

            # Check for abort
            if self.abort_flag:
                return None

            self.update_status(progress=30, message='Generating executive summary...')
            time.sleep(2)

            # Check for abort
            if self.abort_flag:
                return None

            self.update_status(progress=45, message='Conducting market research...')
            time.sleep(2)

            # Check for abort
            if self.abort_flag:
                return None

            self.update_status(progress=60, message='Analyzing financials...')
            time.sleep(2)

            # Check for abort
            if self.abort_flag:
                return None

            self.update_status(progress=75, message='Performing valuation analysis...')
            time.sleep(2)

            # Check for abort
            if self.abort_flag:
                return None

            self.update_status(progress=85, message='Developing investment thesis...')
            time.sleep(2)

            # Check for abort
            if self.abort_flag:
                return None

            self.update_status(progress=95, message='Finalizing risk analysis...')

            # Generate the actual report
            report_content = self.generator.generate_comprehensive_report(ticker)

            if not report_content:
                raise ValueError("Failed to generate report content")

            self.update_status(progress=98, message='Converting to PDF...')

            # Convert to PDF
            pdf_path = self.convert_to_pdf(ticker, report_content, company_name)

            self.update_status(
                status='completed',
                progress=100,
                message=f'Report completed! PDF ready for download.',
                pdf_path=pdf_path
            )

            return pdf_path

        except Exception as e:
            logger.error(f"Error generating report for {ticker}: {e}")
            self.update_status(
                status='error',
                progress=0,
                message=f'Error: {str(e)}',
                error=str(e)
            )
            return None

    def convert_to_pdf(self, ticker, markdown_content, company_name):
        """Convert markdown report to PDF"""
        try:
            import markdown
            import pdfkit
            from datetime import datetime

            # Convert markdown to HTML
            html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])

            # Create styled HTML for PDF
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Equity Research Report - {ticker}</title>
                <style>
                    body {{
                        font-family: 'Times New Roman', serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                        page-break-after: avoid;
                    }}
                    h1 {{
                        border-bottom: 3px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        border-bottom: 1px solid #bdc3c7;
                        padding-bottom: 5px;
                        margin-top: 30px;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    .header {{
                        text-align: center;
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: #f8f9fa;
                        border-radius: 5px;
                    }}
                    .footer {{
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        font-size: 12px;
                        color: #7f8c8d;
                    }}
                    @page {{
                        margin: 2cm;
                        @bottom-right {{
                            content: "Page " counter(page) " of " counter(pages);
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>EQUITY RESEARCH REPORT</h1>
                    <h2>{company_name} ({ticker})</h2>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p><strong>AI Engine:</strong> Mistral:7b (Local LLM)</p>
                </div>

                {html_content}

                <div class="footer">
                    <p><strong>Disclaimer:</strong> This research report is generated using AI and is for informational purposes only.
                    This is not investment advice. Past performance does not guarantee future results.
                    Please consult with qualified financial advisors before making investment decisions.</p>
                    <p><strong>Generated by:</strong> Enhanced Equity Research Tool (ERT) - Powered by Mistral:7b</p>
                </div>
            </body>
            </html>
            """

            # Create PDF
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"{ticker}_equity_report_{timestamp}.pdf"
            pdf_path = os.path.join(project_root, 'reports', pdf_filename)

            # Ensure reports directory exists
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

            # Configure PDF options
            options = {
                'page-size': 'A4',
                'margin-top': '2cm',
                'margin-right': '2cm',
                'margin-bottom': '2cm',
                'margin-left': '2cm',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None
            }

            # Try to generate PDF with pdfkit
            try:
                pdfkit.from_string(styled_html, pdf_path, options=options)
                logger.info(f"PDF generated successfully: {pdf_path}")
                return pdf_filename
            except Exception as pdf_error:
                logger.warning(f"pdfkit failed: {pdf_error}, trying alternative method...")
                return self.create_simple_pdf(ticker, markdown_content, company_name)

        except ImportError:
            logger.warning("PDF libraries not available, creating simple text PDF...")
            return self.create_simple_pdf(ticker, markdown_content, company_name)
        except Exception as e:
            logger.error(f"Error converting to PDF: {e}")
            return self.create_simple_pdf(ticker, markdown_content, company_name)

    def create_simple_pdf(self, ticker, content, company_name):
        """Create a simple PDF using reportlab as fallback"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"{ticker}_equity_report_{timestamp}.pdf"
            pdf_path = os.path.join(project_root, 'reports', pdf_filename)

            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=1*inch)

            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor='navy',
                alignment=1,  # Center
                spaceAfter=30
            )

            # Build story
            story = []

            # Title page
            story.append(Paragraph(f"EQUITY RESEARCH REPORT", title_style))
            story.append(Paragraph(f"{company_name} ({ticker})", title_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
            story.append(Paragraph("AI Engine: Mistral:7b (Local LLM)", styles['Normal']))
            story.append(PageBreak())

            # Content (simplified)
            content_lines = content.split('\n')
            for line in content_lines:
                if line.strip():
                    if line.startswith('# '):
                        story.append(Paragraph(line[2:], styles['Heading1']))
                    elif line.startswith('## '):
                        story.append(Paragraph(line[3:], styles['Heading2']))
                    elif line.startswith('### '):
                        story.append(Paragraph(line[4:], styles['Heading3']))
                    else:
                        story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))

            # Build PDF
            doc.build(story)
            logger.info(f"Simple PDF generated successfully: {pdf_path}")
            return pdf_filename

        except ImportError:
            # Final fallback - save as text file with PDF extension
            logger.warning("No PDF libraries available, saving as text file...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{ticker}_equity_report_{timestamp}.txt"
            file_path = os.path.join(project_root, 'reports', filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"EQUITY RESEARCH REPORT\n")
                f.write(f"{company_name} ({ticker})\n")
                f.write(f"Generated: {datetime.now().strftime('%B %d, %Y')}\n")
                f.write(f"AI Engine: Mistral:7b (Local LLM)\n")
                f.write("="*50 + "\n\n")
                f.write(content)

            return filename

# Flask app setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'ert_simple_gui_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global generator instance
report_generator = SimpleReportGenerator()
report_generator.set_socketio(socketio)

@app.route('/')
def index():
    """Main search interface"""
    return render_template('simple_search.html')

@app.route('/api/status')
def api_status():
    """Get current status"""
    return jsonify(report_generator.current_status)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate report for ticker"""
    data = request.json
    ticker = data.get('ticker', '').strip().upper()

    if not ticker:
        return jsonify({'error': 'Ticker symbol required'}), 400

    if report_generator.current_status['status'] in ['starting', 'generating']:
        return jsonify({'error': 'Report generation already in progress'}), 400

    # Start generation in background thread
    def generate_async():
        report_generator.generate_report_with_pdf(ticker)

    thread = threading.Thread(target=generate_async)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': f'Started generating report for {ticker}'})

@app.route('/api/abort', methods=['POST'])
def api_abort():
    """Abort current report generation"""
    if report_generator.current_status['status'] in ['starting', 'generating']:
        report_generator.abort_generation()
        return jsonify({'success': True, 'message': 'Report generation aborted'})
    else:
        return jsonify({'success': False, 'message': 'No active report generation to abort'}), 400

@app.route('/api/download/<filename>')
def api_download(filename):
    """Download generated PDF"""
    try:
        file_path = os.path.join(project_root, 'reports', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status_update', report_generator.current_status)

@socketio.on('request_status')
def handle_status_request():
    """Handle status request"""
    emit('status_update', report_generator.current_status)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(os.path.join(project_root, 'reports'), exist_ok=True)

    print("🚀 Starting Simple ERT Search Interface...")
    print("🌐 Access at: http://localhost:5001")
    print("📝 Enter any ticker symbol to generate equity research reports")
    print("📄 Automatic PDF download when complete")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)