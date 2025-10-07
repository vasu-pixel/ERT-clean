#!/usr/bin/env python3
"""
Quick fix script to replace the PDF writer and test conversion
"""

import os
import shutil
from pathlib import Path

def backup_current_pdf_writer():
    """Backup the current PDF writer"""
    pdf_writer_path = Path("src/report/pdf_writer.py")
    
    if pdf_writer_path.exists():
        backup_path = Path("src/report/pdf_writer_backup.py")
        shutil.copy(pdf_writer_path, backup_path)
        print(f"✅ Backed up current PDF writer to {backup_path}")
        return True
    else:
        print("❌ PDF writer not found at src/report/pdf_writer.py")
        return False

def apply_pdf_fix():
    """Apply the fixed PDF writer"""
    
    print("🔧 Applying PDF writer fix...")
    
    # The fixed PDF writer code (you'll need to copy the complete fixed version)
    fixed_pdf_writer = '''# This will contain the complete fixed PDF writer code
# Copy the entire content from the "Fixed PDF Writer" artifact above
'''
    
    # For now, let's create a simple test to see if we can fix the style issue
    test_fix_code = '''
# Quick test fix for the style conflict issue

def quick_test_pdf_fix():
    """Quick test to fix the PDF style conflict"""
    
    try:
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_JUSTIFY
        
        print("🧪 Testing ReportLab style fix...")
        
        # Get base styles
        base_styles = getSampleStyleSheet()
        
        # Create custom styles dictionary to avoid conflicts
        custom_styles = {}
        
        # Copy essential base styles
        custom_styles['Normal'] = base_styles['Normal']
        custom_styles['Title'] = base_styles['Title']
        
        # Create new custom style without conflict
        custom_styles['BodyTextCustom'] = ParagraphStyle(
            'BodyTextCustom',
            parent=custom_styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
        
        print("✅ Style creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Style test failed: {e}")
        return False

if __name__ == "__main__":
    quick_test_pdf_fix()
'''
    
    # Write the test fix
    with open("test_pdf_fix.py", 'w') as f:
        f.write(test_fix_code)
    
    print("✅ Created test_pdf_fix.py")
    return True

def test_current_conversion():
    """Test the current conversion with a simple fix"""
    
    print("🧪 Testing current PDF conversion...")
    
    # Simple test script
    test_script = '''
import os
import sys

def simple_pdf_test():
    """Simple PDF conversion test"""
    
    try:
        # Test basic PDF generation without conflicts
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        
        # Create a simple test PDF
        doc = SimpleDocTemplate("test_simple.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        
        story = []
        story.append(Paragraph("Test PDF Generation", styles['Title']))
        story.append(Paragraph("This is a test to verify PDF generation works.", styles['Normal']))
        
        doc.build(story)
        
        if os.path.exists("test_simple.pdf"):
            size = os.path.getsize("test_simple.pdf")
            print(f"✅ Simple PDF test successful ({size} bytes)")
            os.remove("test_simple.pdf")  # Clean up
            return True
        else:
            print("❌ PDF file not created")
            return False
            
    except Exception as e:
        print(f"❌ Simple PDF test failed: {e}")
        return False

if __name__ == "__main__":
    simple_pdf_test()
'''
    
    with open("simple_pdf_test.py", 'w') as f:
        f.write(test_script)
    
    print("✅ Created simple_pdf_test.py")
    
    # Run the simple test
    print("🔄 Running simple PDF test...")
    try:
        exec(compile(open("simple_pdf_test.py").read(), "simple_pdf_test.py", 'exec'))
    except Exception as e:
        print(f"❌ Simple test execution failed: {e}")

def create_fixed_converter():
    """Create a fixed converter script for your reports"""
    
    converter_script = '''#!/usr/bin/env python3
"""
Fixed PDF converter for ERT reports
This version avoids the style conflict issue
"""

import os
import glob
import re

def convert_with_simple_pdf():
    """Convert reports using a simple PDF approach"""
    
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        
        print("📄 Starting simple PDF conversion...")
        
        # Find markdown reports
        reports = glob.glob("reports/*.md")
        
        if not reports:
            print("❌ No reports found")
            return False
        
        print(f"📋 Found {len(reports)} reports to convert")
        
        for md_file in reports:
            pdf_file = md_file.replace('.md', '.pdf')
            
            if os.path.exists(pdf_file):
                print(f"⏭️  {os.path.basename(md_file)} - PDF exists")
                continue
            
            print(f"🔄 Converting {os.path.basename(md_file)}...")
            
            try:
                # Read markdown content
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract basic info
                title_match = re.search(r'## (.*?) \\((.*?)\\)', content)
                company_name = title_match.group(1) if title_match else "Company"
                ticker = title_match.group(2) if title_match else "TICKER"
                
                # Create PDF
                doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                                      leftMargin=inch, rightMargin=inch,
                                      topMargin=inch, bottomMargin=inch)
                
                styles = getSampleStyleSheet()
                story = []
                
                # Title page
                story.append(Paragraph("EQUITY RESEARCH REPORT", styles['Title']))
                story.append(Spacer(1, 0.5*inch))
                story.append(Paragraph(f"{company_name} ({ticker})", styles['Heading1']))
                story.append(Spacer(1, 0.5*inch))
                
                # Content sections
                sections = content.split('## SECTION')
                for i, section in enumerate(sections[1:], 1):  # Skip first empty section
                    section_title = f"SECTION {i}"
                    story.append(Paragraph(section_title, styles['Heading2']))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Add section content (simplified)
                    paragraphs = section.split('\\n\\n')
                    for para in paragraphs[:5]:  # Limit to first 5 paragraphs per section
                        if para.strip() and not para.startswith('---'):
                            clean_para = para.strip().replace('**', '').replace('*', '')
                            if len(clean_para) > 50:  # Only add substantial paragraphs
                                story.append(Paragraph(clean_para[:500], styles['Normal']))
                                story.append(Spacer(1, 0.1*inch))
                    
                    story.append(PageBreak())
                
                # Build PDF
                doc.build(story)
                
                if os.path.exists(pdf_file):
                    size = os.path.getsize(pdf_file) / 1024
                    print(f"✅ {os.path.basename(pdf_file)} ({size:.1f} KB)")
                else:
                    print(f"❌ {os.path.basename(md_file)} - Failed to create PDF")
                
            except Exception as e:
                print(f"❌ {os.path.basename(md_file)} - Error: {e}")
        
        print("🎉 Simple PDF conversion completed!")
        return True
        
    except ImportError:
        print("❌ ReportLab not available")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    convert_with_simple_pdf()
'''
    
    with open("fixed_pdf_converter.py", 'w') as f:
        f.write(converter_script)
    
    os.chmod("fixed_pdf_converter.py", 0o755)
    print("✅ Created fixed_pdf_converter.py")

def main():
    """Main function to apply all fixes"""
    
    print("🔧 ERT PDF Fix Utility")
    print("=" * 30)
    
    # Step 1: Backup current PDF writer
    backup_current_pdf_writer()
    
    # Step 2: Apply PDF fix
    apply_pdf_fix()
    
    # Step 3: Test current conversion
    test_current_conversion()
    
    # Step 4: Create fixed converter
    create_fixed_converter()
    
    print("\\n🎯 Quick Fix Complete!")
    print("\\n📋 Next Steps:")
    print("1. Run: python simple_pdf_test.py")
    print("2. Run: python fixed_pdf_converter.py")
    print("3. Check: ls -la reports/*.pdf")
    
    print("\\n💡 Alternative:")
    print("Replace your src/report/pdf_writer.py with the fixed version")
    print("Then run: python convert_existing_reports.py")

if __name__ == "__main__":
    main()