# src/report/pdf_writer.py - COMPLETE WORKING VERSION
"""
Professional PDF Report Generator for Enhanced Equity Research Tool (ERT)
Generates institutional-quality PDF reports comparable to JP Morgan, Goldman Sachs format
"""

import os
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

# PDF Generation Libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Chart Generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import yfinance as yf
    CHARTS_AVAILABLE = True
    # Set matplotlib to non-interactive backend
    plt.switch_backend('Agg')
except ImportError:
    CHARTS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class InstitutionalPDFGenerator:
    """Professional PDF generator for equity research reports"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        if REPORTLAB_AVAILABLE:
            self.setup_styles()
        
    def setup_styles(self):
        """Setup professional document styles without conflicts"""
        # Create completely custom styles to avoid conflicts
        from reportlab.lib.styles import StyleSheet1
        
        self.styles = StyleSheet1()
        
        # Base style
        base_style = ParagraphStyle(
            'BaseStyle',
            fontName='Helvetica',
            fontSize=10,
            leading=12,
            spaceAfter=6
        )
        self.styles.add(base_style)
        
        # Title styles
        self.styles.add(ParagraphStyle(
            'ReportTitle',
            parent=base_style,
            fontSize=24,
            fontName='Helvetica-Bold',
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            'CompanyName',
            parent=base_style,
            fontSize=18,
            fontName='Helvetica-Bold',
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            'SectionHeader',
            parent=base_style,
            fontSize=14,
            fontName='Helvetica-Bold',
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            'BodyText',
            parent=base_style,
            fontSize=10,
            fontName='Helvetica',
            alignment=TA_JUSTIFY,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            'KeyMetric',
            parent=base_style,
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=colors.darkblue
        ))
    
    def generate_report_pdf(self, report_data: Dict, output_path: str) -> bool:
        """Generate PDF report from report data"""
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not available for PDF generation")
            return False
        
        try:
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=inch,
                leftMargin=inch,
                topMargin=inch,
                bottomMargin=inch
            )
            
            # Build story
            story = []
            
            # Title Page
            story.extend(self._create_title_page(report_data))
            story.append(PageBreak())
            
            dynamic_sections = report_data.get('sections')

            if dynamic_sections:
                for section in dynamic_sections:
                    title = section.get('title', 'Section')
                    content = section.get('content', '')
                    story.extend(self._create_section(title, content))
                    story.append(PageBreak())
            else:
                # Backward compatible legacy flow
                story.extend(self._create_executive_summary(report_data))
                story.append(PageBreak())

                sections = [
                    ('market_research', 'Market Research'),
                    ('financial_analysis', 'Financial Analysis'),
                    ('valuation_analysis', 'Valuation Analysis'),
                    ('investment_thesis', 'Investment Thesis'),
                    ('risk_analysis', 'Risk Analysis')
                ]

                for section_key, section_title in sections:
                    if section_key in report_data:
                        story.extend(self._create_section(section_title, report_data[section_key]))
                        story.append(PageBreak())

            appendices = report_data.get('appendices') or []
            for appendix in appendices:
                title = appendix.get('title', 'Appendix')
                content = appendix.get('content', '')
                story.extend(self._create_section(title, content))
                story.append(PageBreak())

            # Build PDF
            doc.build(story)
            logger.info(f"PDF generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return False
    
    def _create_title_page(self, report_data: Dict) -> List:
        """Create title page"""
        story = []
        
        story.append(Spacer(1, 1.5*inch))
        
        # Title
        story.append(Paragraph(
            "COMPREHENSIVE EQUITY RESEARCH REPORT",
            self.styles['ReportTitle']
        ))
        
        # Company info
        company_name = report_data.get('company_name', 'Company')
        ticker = report_data.get('ticker', 'TICKER')
        story.append(Paragraph(
            f"{company_name} ({ticker})",
            self.styles['CompanyName']
        ))
        
        # Investment summary
        recommendation = report_data.get('recommendation', 'HOLD')
        current_price = report_data.get('current_price', 0)
        target_price = report_data.get('target_price', 0)
        
        if current_price > 0 and target_price > 0:
            upside = ((target_price - current_price) / current_price * 100)
            summary_text = f"{recommendation} | ${current_price:.2f} → ${target_price:.2f} ({upside:.1f}% upside)"
        else:
            summary_text = f"{recommendation} Rating"
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(summary_text, self.styles['KeyMetric']))
        
        # Date
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
            self.styles['BodyText']
        ))
        
        return story
    
    def _create_executive_summary(self, report_data: Dict) -> List:
        """Create executive summary"""
        story = []
        
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        exec_summary = report_data.get('executive_summary', '')
        if exec_summary:
            # Clean and format the text
            paragraphs = exec_summary.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    clean_para = para.strip().replace('**', '').replace('*', '')
                    if len(clean_para) > 20:  # Only substantial paragraphs
                        story.append(Paragraph(clean_para, self.styles['BodyText']))
        
        return story
    
    def _create_section(self, title: str, content: str) -> List:
        """Create a report section"""
        story = []
        
        story.append(Paragraph(title.upper(), self.styles['SectionHeader']))
        
        if content:
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    clean_para = para.strip().replace('**', '').replace('*', '')
                    # Skip markdown headers and separators
                    if not clean_para.startswith('#') and clean_para != '---' and len(clean_para) > 20:
                        story.append(Paragraph(clean_para, self.styles['BodyText']))
        
        return story

def convert_markdown_to_pdf(markdown_file: str, output_file: str, config: Dict = None) -> bool:
    """Convert markdown report to PDF using parsed sections and appendices."""
    try:
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not available - install with: pip install reportlab")
            return False

        report_data = parse_markdown_report(markdown_file)
        if not report_data:
            print("❌ Unable to parse markdown report")
            return False

        generator = InstitutionalPDFGenerator(config)
        return generator.generate_report_pdf(report_data, output_path=output_file)

    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def parse_markdown_report(markdown_file: str) -> Dict:
    """Parse markdown report file into structured sections and appendices."""
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()

        report_data: Dict[str, Any] = {}

        company_match = re.search(r'## (.*?) \((.*?)\)', content)
        if company_match:
            report_data['company_name'] = company_match.group(1)
            report_data['ticker'] = company_match.group(2)

        rating_match = re.search(r'### (.*?) Rating.*?Price Target: \$([0-9.]+)', content)
        if rating_match:
            report_data['recommendation'] = rating_match.group(1)
            report_data['target_price'] = float(rating_match.group(2))

        current_price_match = re.search(r'Current Price.*?\$([0-9.]+)', content)
        if current_price_match:
            report_data['current_price'] = float(current_price_match.group(1))

        section_pattern = re.compile(
            r'## SECTION\s+(\d+):\s+(.*?)\n(.*?)(?=\n## SECTION|\n## APPENDICES|\Z)',
            re.DOTALL
        )
        sections: List[Dict[str, Any]] = []
        for match in section_pattern.finditer(content):
            index = int(match.group(1))
            title = match.group(2).strip()
            section_content = match.group(3).strip()
            sections.append({
                'index': index,
                'title': title,
                'content': section_content,
                'word_count': len(section_content.split()),
            })
        report_data['sections'] = sections

        appendices: List[Dict[str, Any]] = []
        appendices_match = re.search(r'## APPENDICES\n(.*)', content, re.DOTALL)
        if appendices_match:
            appendix_content = appendices_match.group(1)
            appendix_pattern = re.compile(r'###\s+(.*?)\n(.*?)(?=\n###|\Z)', re.DOTALL)
            for match in appendix_pattern.finditer(appendix_content):
                appendices.append({
                    'title': match.group(1).strip(),
                    'content': match.group(2).strip(),
                })
        report_data['appendices'] = appendices

        return report_data

    except Exception as e:
        logger.error(f"Error parsing markdown: {e}")
        return {}

def test_pdf_generation():
    """Test PDF generation"""
    try:
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not available")
            return False
        
        # Test data
        test_data = {
            'company_name': 'Apple Inc.',
            'ticker': 'AAPL',
            'recommendation': 'BUY',
            'current_price': 175.50,
            'target_price': 200.00,
            'executive_summary': 'Apple presents a compelling investment opportunity with strong fundamentals.'
        }
        
        # Create test directory
        os.makedirs('test_reports', exist_ok=True)
        
        # Generate test PDF
        generator = InstitutionalPDFGenerator()
        success = generator.generate_report_pdf(test_data, 'test_reports/test_report.pdf')
        
        if success:
            print("✅ PDF test successful")
            return True
        else:
            print("❌ PDF test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main CLI function"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_pdf_generation()
        elif sys.argv[1] == '--convert' and len(sys.argv) > 2:
            md_file = sys.argv[2]
            pdf_file = md_file.replace('.md', '.pdf')
            success = convert_markdown_to_pdf(md_file, pdf_file)
            if success:
                print(f"✅ Converted: {pdf_file}")
            else:
                print(f"❌ Failed: {md_file}")
        else:
            print("Usage: python pdf_writer.py --test | --convert file.md")
    else:
        print("ERT PDF Writer")
        print("Usage: python pdf_writer.py --test")

if __name__ == "__main__":
    main()
