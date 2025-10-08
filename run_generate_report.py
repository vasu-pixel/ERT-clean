# run_generate_report.py
import os
import sys
import argparse
import logging
from datetime import datetime
import json
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stock_report_generator import StockReportGenerator
from src.utils.openai_engine import OpenAIEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReportRunner:
    """
    Enhanced report runner for generating comprehensive equity research reports
    """

    def __init__(self):
        self.openai_engine = OpenAIEngine()
        self.generator = StockReportGenerator(self.openai_engine)
        self.start_time = datetime.now()

    def validate_environment(self) -> bool:
        """Validate that all required environment variables and dependencies are set"""

        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY environment variable not set")
            logger.info("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key'")
            return False

        # Check if required directories exist
        required_dirs = ['src', 'reports']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                logger.info(f"Creating missing directory: {dir_name}")
                os.makedirs(dir_name, exist_ok=True)

        # Test OpenAI connection
        try:
            if not self.openai_engine.test_connection():
                logger.error("OpenAI API connection failed.")
                return False
            logger.info("OpenAI API connection successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection failed: {e}")
            return False

    def validate_ticker(self, ticker: str) -> bool:
        """Validate that the ticker symbol is valid"""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            # Check if we got valid data
            if 'longName' not in info or info.get('longName') is None:
                logger.error(f"Invalid ticker symbol: {ticker}")
                return False

            logger.info(f"Ticker validated: {ticker} - {info.get('longName')}")
            return True

        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}")
            return False

    def generate_single_report(self, ticker: str, save_json: bool = True) -> Optional[str]:
        """Generate a single comprehensive research report"""

        logger.info(f"Starting report generation for {ticker}")

        try:
            # Validate ticker
            if not self.validate_ticker(ticker):
                return None

            # Generate comprehensive report
            report = self.generator.generate_comprehensive_report(ticker)

            # Save additional formats if requested
            if save_json:
                self._save_report_metadata(ticker, report)

            logger.info(f"Report generation completed for {ticker}")
            return report

        except Exception as e:
            logger.error(f"Error generating report for {ticker}: {e}")
            return None

    def generate_batch_reports(self, tickers: List[str], max_concurrent: int = 3) -> List[str]:
        """Generate reports for multiple tickers"""

        logger.info(f"Starting batch report generation for {len(tickers)} tickers")

        successful_reports = []
        failed_tickers = []

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker}")

            try:
                report = self.generate_single_report(ticker)
                if report:
                    successful_reports.append(ticker)
                    logger.info(f"✓ Successfully generated report for {ticker}")
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"✗ Failed to generate report for {ticker}")

            except Exception as e:
                logger.error(f"✗ Error processing {ticker}: {e}")
                failed_tickers.append(ticker)

        # Generate batch summary
        self._generate_batch_summary(successful_reports, failed_tickers)

        logger.info(f"Batch processing completed. Success: {len(successful_reports)}, Failed: {len(failed_tickers)}")
        return successful_reports

    def _save_report_metadata(self, ticker: str, report: str):
        """Save additional metadata about the generated report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            metadata = {
                'ticker': ticker,
                'generation_date': self.start_time.isoformat(),
                'completion_date': datetime.now().isoformat(),
                'word_count': len(report.split()),
                'estimated_pages': len(report.split()) // 250,
                'report_sections': [
                    'Executive Summary',
                    'Market Research',
                    'Financial Analysis',
                    'Valuation Analysis',
                    'Investment Thesis',
                    'Risk Analysis'
                ],
                'data_sources': [
                    'Yahoo Finance',
                    'OpenAI Analysis',
                    'Public Company Filings'
                ]
            }

            # Save metadata
            os.makedirs('reports', exist_ok=True)
            metadata_file = f"reports/{ticker}_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved to {metadata_file}")

        except Exception as e:
            logger.error(f"Error saving metadata for {ticker}: {e}")

    def _generate_batch_summary(self, successful: List[str], failed: List[str]):
        """Generate summary of batch processing"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            total_requests = len(successful) + len(failed)
            success_rate = (len(successful) / total_requests * 100) if total_requests else 0.0

            summary = {
                'batch_date': self.start_time.isoformat(),
                'completion_date': datetime.now().isoformat(),
                'total_tickers': total_requests,
                'successful_reports': len(successful),
                'failed_reports': len(failed),
                'success_rate': success_rate,
                'successful_tickers': successful,
                'failed_tickers': failed,
                'processing_time_minutes': (datetime.now() - self.start_time).total_seconds() / 60
            }

            # Save batch summary
            os.makedirs('reports', exist_ok=True)
            summary_file = f"reports/batch_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Batch summary saved to {summary_file}")

        except Exception as e:
            logger.error(f"Error generating batch summary: {e}")

    def interactive_mode(self):
        """Run in interactive mode for testing and development"""

        print("\n" + "="*60)
        print("  ENHANCED EQUITY RESEARCH REPORT GENERATOR")
        print("="*60)
        print("Interactive Mode - Type 'quit' to exit")
        print()

        while True:
            try:
                ticker = input("Enter ticker symbol (or 'quit'): ").strip().upper()

                if ticker in ['QUIT', 'EXIT', 'Q']:
                    print("Goodbye!")
                    break

                if not ticker:
                    continue

                print(f"\nGenerating comprehensive research report for {ticker}...")
                print("This may take 2-3 minutes depending on API response times...")

                report = self.generate_single_report(ticker)

                if report:
                    print(f"\n✓ Report generated successfully for {ticker}")
                    print(f"Word count: {len(report.split())} words")
                    print(f"Estimated pages: {len(report.split()) // 250}")

                    # Show preview
                    preview_lines = report.split('\n')[:20]
                    print("\nReport Preview:")
                    print("-" * 40)
                    for line in preview_lines:
                        print(line)
                    print("-" * 40)
                    print("(Full report saved to reports/ directory)")

                else:
                    print(f"✗ Failed to generate report for {ticker}")

                print("\n" + "-"*60)

            except KeyboardInterrupt:
                print("\n\nProcess interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function with command line interface"""

    parser = argparse.ArgumentParser(
        description="Enhanced Equity Research Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python run_generate_report.py --ticker AAPL
  python run_generate_report.py --batch AAPL MSFT GOOGL
  python run_generate_report.py --interactive
  python run_generate_report.py --ticker TSLA --no-json
        """
    )

    parser.add_argument(
        '--ticker', '-t',
        type=str,
        help='Single ticker symbol to analyze'
    )

    parser.add_argument(
        '--batch', '-b',
        nargs='+',
        help='Multiple ticker symbols for batch processing'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip saving JSON metadata'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment and exit'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize runner
    runner = ReportRunner()

    # Validate environment
    print("Validating environment...")
    if not runner.validate_environment():
        print("❌ Environment validation failed. Please check your setup.")
        sys.exit(1)

    print("✅ Environment validation successful")

    if args.validate_only:
        print("Validation complete. Exiting.")
        sys.exit(0)

    # Process based on arguments
    if args.interactive:
        runner.interactive_mode()

    elif args.ticker:
        print(f"\nGenerating report for {args.ticker}...")
        report = runner.generate_single_report(
            args.ticker,
            save_json=not args.no_json
        )

        if report:
            print(f"✅ Report generated successfully for {args.ticker}")
            print(f"📄 Word count: {len(report.split())} words")
            print(f"📊 Estimated pages: {len(report.split()) // 250}")
        else:
            print(f"❌ Failed to generate report for {args.ticker}")
            sys.exit(1)

    elif args.batch:
        print(f"\nGenerating batch reports for {len(args.batch)} tickers...")
        successful = runner.generate_batch_reports(args.batch)

        print(f"\n📈 Batch processing completed:")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(args.batch) - len(successful)}")

        if successful:
            print(f"📁 Reports saved to reports/ directory")

    else:
        print("No action specified. Use --help for usage information.")
        print("Quick start: python run_generate_report.py --interactive")

if __name__ == "__main__":
    main()
