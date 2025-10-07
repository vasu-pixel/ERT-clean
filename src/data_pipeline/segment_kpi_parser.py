"""
Segment KPI Parser
Extracts segment-level KPIs from 10-K filings for multi-business companies
Analyzes geographic and business segment performance data
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SegmentKPIParser:
    """Parses segment-level KPIs from SEC filings"""

    def __init__(self):
        self.segment_keywords = [
            'segment', 'division', 'business unit', 'operating segment',
            'reportable segment', 'geographic', 'product line'
        ]

        self.kpi_patterns = {
            'revenue': [
                r'revenue[s]?\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
                r'net\s+sales?\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
                r'total\s+revenue[s]?\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
            ],
            'operating_income': [
                r'operating\s+income\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
                r'operating\s+profit\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
                r'segment\s+profit\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
            ],
            'assets': [
                r'total\s+assets\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
                r'segment\s+assets\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
            ],
            'depreciation': [
                r'depreciation\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
                r'amortization\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
            ],
            'capex': [
                r'capital\s+expenditures?\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
                r'capex\s*[\$\s]*([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
            ]
        }

        self.geographical_indicators = [
            'united states', 'u.s.', 'america', 'domestic',
            'europe', 'asia', 'china', 'japan', 'international',
            'americas', 'emea', 'apac', 'rest of world'
        ]

    def extract_segment_data(self, ticker: str, filing_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract comprehensive segment data from 10-K filing"""

        print(f"🏢 Extracting segment KPIs for {ticker}...")

        if not filing_dir:
            filing_dir = f"sec-edgar-filings/{ticker}/10-K"

        segment_data = {
            'ticker': ticker.upper(),
            'extraction_date': datetime.now().isoformat(),
            'business_segments': [],
            'geographic_segments': [],
            'segment_summary': {},
            'consolidation_data': {},
            'segment_trends': {}
        }

        try:
            # Find most recent 10-K filing
            filing_path = self._find_latest_filing(filing_dir)
            if not filing_path:
                print(f"❌ No 10-K filing found for {ticker}")
                return segment_data

            print(f"📄 Processing filing: {filing_path}")

            # Extract text from filing
            filing_text = self._extract_filing_text(filing_path)
            if not filing_text:
                print(f"❌ Could not extract text from filing")
                return segment_data

            # Find segment reporting sections
            segment_sections = self._find_segment_sections(filing_text)

            # Extract business segments
            segment_data['business_segments'] = self._extract_business_segments(segment_sections)

            # Extract geographic segments
            segment_data['geographic_segments'] = self._extract_geographic_segments(segment_sections)

            # Create segment summary
            segment_data['segment_summary'] = self._create_segment_summary(segment_data)

            # Extract consolidation and elimination data
            segment_data['consolidation_data'] = self._extract_consolidation_data(segment_sections)

            # Analyze segment trends (if multi-year data available)
            segment_data['segment_trends'] = self._analyze_segment_trends(segment_data)

            print(f"✅ Extracted {len(segment_data['business_segments'])} business segments")
            print(f"✅ Extracted {len(segment_data['geographic_segments'])} geographic segments")

            return segment_data

        except Exception as e:
            logger.error(f"Segment extraction failed: {e}")
            print(f"❌ Extraction failed: {e}")
            return segment_data

    def _find_latest_filing(self, filing_dir: str) -> Optional[str]:
        """Find the most recent 10-K filing"""
        try:
            base_path = Path(filing_dir)
            if not base_path.exists():
                return None

            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            if not subdirs:
                return None

            subdirs.sort(reverse=True)

            for subdir in subdirs:
                filing_file = subdir / "full-submission.txt"
                if filing_file.exists():
                    return str(filing_file)

            return None

        except Exception as e:
            logger.debug(f"Filing search failed: {e}")
            return None

    def _extract_filing_text(self, filing_path: str) -> Optional[str]:
        """Extract text from SEC filing"""
        try:
            with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract main document text
            doc_start = content.find('<DOCUMENT>')
            if doc_start == -1:
                return content

            doc_end = content.find('</DOCUMENT>')
            if doc_end == -1:
                doc_end = len(content)

            document_text = content[doc_start:doc_end]

            # Remove HTML tags
            clean_text = re.sub(r'<[^>]+>', ' ', document_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)

            return clean_text

        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            return None

    def _find_segment_sections(self, text: str) -> List[str]:
        """Find sections containing segment reporting data"""
        sections = []

        # Common patterns for segment sections
        segment_patterns = [
            r'(segment\s+information.*?)(?=\n[A-Z][A-Z\s]{15,}|\nITEM\s+\d+|$)',
            r'(business\s+segments?.*?)(?=\n[A-Z][A-Z\s]{15,}|\nITEM\s+\d+|$)',
            r'(reportable\s+segments?.*?)(?=\n[A-Z][A-Z\s]{15,}|\nITEM\s+\d+|$)',
            r'(geographic\s+information.*?)(?=\n[A-Z][A-Z\s]{15,}|\nITEM\s+\d+|$)',
            r'(segment\s+reporting.*?)(?=\n[A-Z][A-Z\s]{15,}|\nITEM\s+\d+|$)',
            r'(note\s+\d+.*?segments?.*?)(?=\nnote\s+\d+|\n[A-Z][A-Z\s]{15,}|$)'
        ]

        for pattern in segment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_text = match.group(1)
                if len(section_text) > 500:  # Filter out very short matches
                    sections.append(section_text)

        return sections

    def _extract_business_segments(self, sections: List[str]) -> List[Dict[str, Any]]:
        """Extract business segment data"""
        business_segments = []

        for section in sections:
            # Look for segment names and associated data
            segments = self._parse_business_segment_section(section)
            business_segments.extend(segments)

        return self._deduplicate_segments(business_segments, 'business')

    def _extract_geographic_segments(self, sections: List[str]) -> List[Dict[str, Any]]:
        """Extract geographic segment data"""
        geographic_segments = []

        for section in sections:
            # Look for geographic regions and associated data
            segments = self._parse_geographic_segment_section(section)
            geographic_segments.extend(segments)

        return self._deduplicate_segments(geographic_segments, 'geographic')

    def _parse_business_segment_section(self, section: str) -> List[Dict[str, Any]]:
        """Parse business segment information from a section"""
        segments = []

        # Common business segment names
        business_segment_patterns = [
            r'(consumer\s+[\w\s]*)',
            r'(enterprise\s+[\w\s]*)',
            r'(commercial\s+[\w\s]*)',
            r'(retail\s+[\w\s]*)',
            r'(wholesale\s+[\w\s]*)',
            r'(technology\s+[\w\s]*)',
            r'(healthcare\s+[\w\s]*)',
            r'(financial\s+services\s+[\w\s]*)',
            r'(automotive\s+[\w\s]*)',
            r'(aerospace\s+[\w\s]*)',
            r'(energy\s+[\w\s]*)',
            r'(media\s+[\w\s]*)',
            r'(telecommunications\s+[\w\s]*)',
            r'(pharmaceuticals?\s+[\w\s]*)',
            r'(industrial\s+[\w\s]*)'
        ]

        for pattern in business_segment_patterns:
            matches = re.finditer(pattern, section, re.IGNORECASE)
            for match in matches:
                segment_name = match.group(1).strip()
                if len(segment_name) > 3:  # Valid segment name
                    # Extract financial data around this segment
                    segment_data = self._extract_segment_financials(section, segment_name, match.start())
                    if segment_data:
                        segments.append(segment_data)

        return segments

    def _parse_geographic_segment_section(self, section: str) -> List[Dict[str, Any]]:
        """Parse geographic segment information from a section"""
        segments = []

        for geo_indicator in self.geographical_indicators:
            pattern = rf'({re.escape(geo_indicator)}[\w\s]*)'
            matches = re.finditer(pattern, section, re.IGNORECASE)

            for match in matches:
                region_name = match.group(1).strip()
                if len(region_name) > 2:
                    # Extract financial data for this region
                    segment_data = self._extract_segment_financials(section, region_name, match.start())
                    if segment_data:
                        segment_data['segment_type'] = 'geographic'
                        segments.append(segment_data)

        return segments

    def _extract_segment_financials(self, section: str, segment_name: str, segment_position: int) -> Optional[Dict[str, Any]]:
        """Extract financial metrics for a specific segment"""

        # Define search window around segment name
        start_pos = max(0, segment_position - 1000)
        end_pos = min(len(section), segment_position + 2000)
        context = section[start_pos:end_pos]

        segment_data = {
            'segment_name': segment_name,
            'segment_type': 'business',
            'financials': {}
        }

        # Extract KPIs
        for kpi_type, patterns in self.kpi_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    amount = self._parse_amount(match.group(1))
                    if amount and amount > 1000000:  # At least $1M
                        segment_data['financials'][kpi_type] = amount
                        break  # Take first valid match
            if kpi_type in segment_data['financials']:
                break  # Move to next KPI type

        # Only return if we found meaningful financial data
        if len(segment_data['financials']) >= 1:
            return segment_data

        return None

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse financial amount into numeric value"""
        try:
            # Remove commas and convert to float
            clean_amount = re.sub(r'[^\d\.]', '', amount_str)
            amount = float(clean_amount)

            # Check context for scale indicators
            return amount * 1000000  # Assume millions unless otherwise specified

        except (ValueError, AttributeError):
            return None

    def _create_segment_summary(self, segment_data: Dict) -> Dict[str, Any]:
        """Create summary statistics for segment data"""
        business_segments = segment_data.get('business_segments', [])
        geographic_segments = segment_data.get('geographic_segments', [])

        summary = {
            'total_business_segments': len(business_segments),
            'total_geographic_segments': len(geographic_segments),
            'business_segment_names': [seg['segment_name'] for seg in business_segments],
            'geographic_regions': [seg['segment_name'] for seg in geographic_segments],
            'largest_segment_by_revenue': None,
            'segment_concentration': {},
            'data_quality': 'High' if len(business_segments) >= 2 else 'Medium' if len(business_segments) >= 1 else 'Low'
        }

        # Find largest segment by revenue
        all_segments = business_segments + geographic_segments
        revenue_segments = [seg for seg in all_segments if 'revenue' in seg.get('financials', {})]

        if revenue_segments:
            largest_segment = max(revenue_segments, key=lambda x: x['financials']['revenue'])
            summary['largest_segment_by_revenue'] = {
                'name': largest_segment['segment_name'],
                'revenue': largest_segment['financials']['revenue'],
                'type': largest_segment['segment_type']
            }

            # Calculate concentration
            total_revenue = sum(seg['financials']['revenue'] for seg in revenue_segments)
            if total_revenue > 0:
                for seg in revenue_segments:
                    concentration = seg['financials']['revenue'] / total_revenue
                    summary['segment_concentration'][seg['segment_name']] = concentration

        return summary

    def _extract_consolidation_data(self, sections: List[str]) -> Dict[str, Any]:
        """Extract consolidation and elimination data"""
        consolidation = {
            'eliminations': {},
            'intersegment_revenue': None,
            'corporate_unallocated': {},
            'reconciliation_items': []
        }

        for section in sections:
            # Look for elimination and reconciliation items
            if 'elimination' in section.lower() or 'reconciliation' in section.lower():
                # Extract elimination amounts
                amount_matches = re.finditer(r'\$([0-9,]+(?:\.[0-9]+)?)', section)
                for match in amount_matches:
                    amount = self._parse_amount(match.group(1))
                    if amount:
                        consolidation['eliminations'][match.group()] = amount

        return consolidation

    def _analyze_segment_trends(self, segment_data: Dict) -> Dict[str, Any]:
        """Analyze segment trends (placeholder for multi-year analysis)"""
        trends = {
            'growth_rates': {},
            'margin_trends': {},
            'market_share_changes': {},
            'notes': 'Single-year analysis - multi-year trending requires additional data'
        }

        # This would be enhanced with multi-year data
        business_segments = segment_data.get('business_segments', [])
        for segment in business_segments:
            segment_name = segment['segment_name']
            trends['growth_rates'][segment_name] = 'N/A - requires historical data'

        return trends

    def _deduplicate_segments(self, segments: List[Dict], segment_type: str) -> List[Dict]:
        """Remove duplicate segments"""
        seen_names = set()
        unique_segments = []

        for segment in segments:
            name = segment['segment_name'].lower().strip()
            if name not in seen_names and len(name) > 2:
                seen_names.add(name)
                unique_segments.append(segment)

        return unique_segments


def test_segment_kpi_parser():
    """Test segment KPI parser"""
    parser = SegmentKPIParser()

    print("="*60)
    print("TESTING SEGMENT KPI PARSER")
    print("="*60)

    # Test with AAPL (should have filing available)
    segment_data = parser.extract_segment_data('AAPL')

    if segment_data['business_segments'] or segment_data['geographic_segments']:
        print(f"\n🏢 Segment Analysis for AAPL:")
        summary = segment_data['segment_summary']

        print(f"Business segments: {summary.get('total_business_segments', 0)}")
        print(f"Geographic segments: {summary.get('total_geographic_segments', 0)}")

        if summary.get('business_segment_names'):
            print(f"Business segments: {', '.join(summary['business_segment_names'])}")

        if summary.get('geographic_regions'):
            print(f"Geographic regions: {', '.join(summary['geographic_regions'])}")

        largest = summary.get('largest_segment_by_revenue')
        if largest:
            print(f"Largest segment: {largest['name']} (${largest['revenue']/1e9:.1f}B)")

        print(f"Data quality: {summary.get('data_quality', 'Unknown')}")

    else:
        print("❌ No segment data extracted")


if __name__ == "__main__":
    test_segment_kpi_parser()