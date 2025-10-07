"""
Debt Schedule Extractor
Extracts detailed debt schedules from 10-K filings including maturity profiles, interest rates, and covenants
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class DebtScheduleExtractor:
    """Extracts debt schedules from SEC filings"""

    def __init__(self):
        self.debt_keywords = [
            'debt', 'borrowings', 'credit facilities', 'notes payable', 'bonds',
            'term loan', 'revolving credit', 'senior notes', 'convertible',
            'commercial paper', 'capital leases', 'finance leases'
        ]

        self.maturity_patterns = [
            r'matur(?:es?|ing|ity)\s+(?:in\s+|on\s+)?(\d{4})',
            r'due\s+(?:in\s+|on\s+)?(\d{4})',
            r'expires?\s+(?:in\s+|on\s+)?(\d{4})',
            r'(\d{4})\s+maturity',
            r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})',  # Date format
        ]

        self.interest_patterns = [
            r'interest\s+rate\s+of\s+([\d\.]+)%',
            r'rate\s+of\s+([\d\.]+)%',
            r'([\d\.]+)%\s+interest',
            r'coupon\s+of\s+([\d\.]+)%',
            r'([\d\.]+)%\s+annual',
            r'LIBOR\s*\+\s*([\d\.]+)%?',
            r'SOFR\s*\+\s*([\d\.]+)%?'
        ]

    def extract_debt_schedule(self, ticker: str, filing_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract comprehensive debt schedule from 10-K filing"""

        print(f"📋 Extracting debt schedule for {ticker}...")

        if not filing_dir:
            filing_dir = f"sec-edgar-filings/{ticker}/10-K"

        debt_schedule = {
            'ticker': ticker.upper(),
            'extraction_date': datetime.now().isoformat(),
            'debt_instruments': [],
            'maturity_profile': {},
            'covenant_summary': {},
            'credit_facilities': {},
            'summary': {}
        }

        try:
            # Find most recent 10-K filing
            filing_path = self._find_latest_filing(filing_dir)
            if not filing_path:
                print(f"❌ No 10-K filing found for {ticker}")
                return debt_schedule

            print(f"📄 Processing filing: {filing_path}")

            # Extract text from filing
            filing_text = self._extract_filing_text(filing_path)
            if not filing_text:
                print(f"❌ Could not extract text from filing")
                return debt_schedule

            # Extract debt information
            debt_schedule['debt_instruments'] = self._extract_debt_instruments(filing_text)
            debt_schedule['maturity_profile'] = self._create_maturity_profile(debt_schedule['debt_instruments'])
            debt_schedule['covenant_summary'] = self._extract_covenants(filing_text)
            debt_schedule['credit_facilities'] = self._extract_credit_facilities(filing_text)
            debt_schedule['summary'] = self._create_debt_summary(debt_schedule)

            print(f"✅ Extracted {len(debt_schedule['debt_instruments'])} debt instruments")
            return debt_schedule

        except Exception as e:
            logger.error(f"Debt schedule extraction failed: {e}")
            print(f"❌ Extraction failed: {e}")
            return debt_schedule

    def _find_latest_filing(self, filing_dir: str) -> Optional[str]:
        """Find the most recent 10-K filing"""
        try:
            base_path = Path(filing_dir)
            if not base_path.exists():
                return None

            # Find subdirectories (filing dates)
            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            if not subdirs:
                return None

            # Sort by directory name (most recent first)
            subdirs.sort(reverse=True)

            # Look for filing file
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

            # Extract main document text (remove EDGAR headers)
            # Look for document boundaries
            doc_start = content.find('<DOCUMENT>')
            if doc_start == -1:
                return content

            doc_end = content.find('</DOCUMENT>')
            if doc_end == -1:
                doc_end = len(content)

            document_text = content[doc_start:doc_end]

            # Remove HTML tags for cleaner text processing
            clean_text = re.sub(r'<[^>]+>', ' ', document_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)

            return clean_text

        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            return None

    def _extract_debt_instruments(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual debt instruments with details"""
        instruments = []

        # Split text into sections that might contain debt information
        debt_sections = self._find_debt_sections(text)

        for section in debt_sections:
            # Look for debt instruments in this section
            section_instruments = self._parse_debt_section(section)
            instruments.extend(section_instruments)

        # Deduplicate and clean up
        return self._deduplicate_instruments(instruments)

    def _find_debt_sections(self, text: str) -> List[str]:
        """Find sections likely to contain debt information"""
        sections = []

        # Common section headers for debt information
        debt_section_patterns = [
            r'(debt\s+and\s+borrowings?.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|\nITEM\s+\d+|$)',
            r'(borrowings?\s+and\s+debt.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|\nITEM\s+\d+|$)',
            r'(credit\s+facilities.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|\nITEM\s+\d+|$)',
            r'(long[- ]?term\s+debt.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|\nITEM\s+\d+|$)',
            r'(notes\s+payable.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|\nITEM\s+\d+|$)'
        ]

        for pattern in debt_section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_text = match.group(1)
                if len(section_text) > 100:  # Filter out very short matches
                    sections.append(section_text)

        return sections

    def _parse_debt_section(self, section: str) -> List[Dict[str, Any]]:
        """Parse a section of text for debt instruments"""
        instruments = []

        # Look for debt amounts
        amount_patterns = [
            r'\$([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?\s*dollars?'
        ]

        for amount_pattern in amount_patterns:
            amount_matches = re.finditer(amount_pattern, section, re.IGNORECASE)

            for amount_match in amount_matches:
                # Extract context around the amount
                start = max(0, amount_match.start() - 200)
                end = min(len(section), amount_match.end() + 200)
                context = section[start:end]

                # Parse instrument details from context
                instrument = self._parse_instrument_details(context, amount_match.group())
                if instrument:
                    instruments.append(instrument)

        return instruments

    def _parse_instrument_details(self, context: str, amount_str: str) -> Optional[Dict[str, Any]]:
        """Parse details of a specific debt instrument"""
        try:
            instrument = {
                'amount_raw': amount_str,
                'amount_numeric': self._parse_amount(amount_str),
                'instrument_type': self._identify_instrument_type(context),
                'maturity_date': self._extract_maturity(context),
                'interest_rate': self._extract_interest_rate(context),
                'covenants': self._extract_instrument_covenants(context),
                'context': context[:500]  # First 500 chars for reference
            }

            # Only return if we have meaningful data
            if instrument['amount_numeric'] and instrument['amount_numeric'] > 1000000:  # At least $1M
                return instrument

        except Exception as e:
            logger.debug(f"Instrument parsing failed: {e}")

        return None

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse debt amount into numeric value"""
        try:
            # Remove currency symbols and commas
            clean_amount = re.sub(r'[^\d\.]', '', amount_str)
            amount = float(clean_amount)

            # Check for scale indicators in original string
            amount_str_lower = amount_str.lower()
            if 'billion' in amount_str_lower:
                amount *= 1e9
            elif 'million' in amount_str_lower:
                amount *= 1e6
            elif 'thousand' in amount_str_lower:
                amount *= 1e3

            return amount

        except (ValueError, AttributeError):
            return None

    def _identify_instrument_type(self, context: str) -> str:
        """Identify the type of debt instrument"""
        context_lower = context.lower()

        type_keywords = {
            'Senior Notes': ['senior notes', 'senior note'],
            'Term Loan': ['term loan', 'term facility'],
            'Revolving Credit': ['revolving credit', 'revolving facility', 'credit line'],
            'Convertible Notes': ['convertible', 'convertible notes'],
            'Commercial Paper': ['commercial paper'],
            'Bonds': ['bonds', 'bond'],
            'Capital Leases': ['capital lease', 'finance lease'],
            'Credit Facility': ['credit facility', 'credit agreement']
        }

        for instrument_type, keywords in type_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                return instrument_type

        return 'Other Debt'

    def _extract_maturity(self, context: str) -> Optional[str]:
        """Extract maturity date from context"""
        for pattern in self.maturity_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    return match.group(1)
                elif len(match.groups()) == 3:  # Date format
                    month, day, year = match.groups()
                    return f"{month}/{day}/{year}"

        return None

    def _extract_interest_rate(self, context: str) -> Optional[str]:
        """Extract interest rate from context"""
        for pattern in self.interest_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1) + '%'

        return None

    def _extract_instrument_covenants(self, context: str) -> List[str]:
        """Extract covenants mentioned for this instrument"""
        covenants = []

        covenant_keywords = [
            'debt to equity ratio', 'interest coverage', 'debt service coverage',
            'minimum net worth', 'maximum leverage', 'current ratio',
            'working capital', 'tangible net worth', 'debt to capitalization'
        ]

        context_lower = context.lower()
        for covenant in covenant_keywords:
            if covenant in context_lower:
                covenants.append(covenant)

        return covenants

    def _create_maturity_profile(self, instruments: List[Dict]) -> Dict[str, Any]:
        """Create maturity profile from debt instruments"""
        profile = {
            'by_year': {},
            'total_debt': 0,
            'weighted_average_maturity': 0
        }

        current_year = datetime.now().year
        total_amount = 0
        weighted_maturity_sum = 0

        for instrument in instruments:
            amount = instrument.get('amount_numeric', 0)
            maturity_str = instrument.get('maturity_date')

            if amount and maturity_str:
                total_amount += amount

                # Parse maturity year
                maturity_year = self._parse_maturity_year(maturity_str)
                if maturity_year:
                    years_to_maturity = maturity_year - current_year
                    weighted_maturity_sum += amount * years_to_maturity

                    year_key = str(maturity_year)
                    if year_key not in profile['by_year']:
                        profile['by_year'][year_key] = 0
                    profile['by_year'][year_key] += amount

        profile['total_debt'] = total_amount
        if total_amount > 0:
            profile['weighted_average_maturity'] = weighted_maturity_sum / total_amount

        return profile

    def _parse_maturity_year(self, maturity_str: str) -> Optional[int]:
        """Parse year from maturity string"""
        try:
            # Look for 4-digit year
            year_match = re.search(r'(\d{4})', maturity_str)
            if year_match:
                return int(year_match.group(1))
        except (ValueError, AttributeError):
            pass
        return None

    def _extract_covenants(self, text: str) -> Dict[str, Any]:
        """Extract debt covenants from filing text"""
        covenants = {
            'financial_covenants': [],
            'negative_covenants': [],
            'affirmative_covenants': []
        }

        # Look for covenant sections
        covenant_patterns = [
            r'(financial\s+covenants?.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|$)',
            r'(covenants?.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|$)',
            r'(debt\s+covenants?.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|$)'
        ]

        for pattern in covenant_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                covenant_text = match.group(1)
                parsed_covenants = self._parse_covenant_text(covenant_text)

                for category, items in parsed_covenants.items():
                    covenants[category].extend(items)

        return covenants

    def _parse_covenant_text(self, text: str) -> Dict[str, List[str]]:
        """Parse covenant text for specific covenants"""
        covenants = {
            'financial_covenants': [],
            'negative_covenants': [],
            'affirmative_covenants': []
        }

        financial_keywords = [
            'debt to equity', 'interest coverage', 'debt service coverage',
            'minimum net worth', 'maximum leverage', 'current ratio'
        ]

        text_lower = text.lower()
        for keyword in financial_keywords:
            if keyword in text_lower:
                covenants['financial_covenants'].append(keyword)

        return covenants

    def _extract_credit_facilities(self, text: str) -> Dict[str, Any]:
        """Extract credit facility information"""
        facilities = {
            'revolving_facilities': [],
            'term_facilities': [],
            'total_committed': 0,
            'total_drawn': 0
        }

        # Look for credit facility sections
        facility_patterns = [
            r'(credit\s+facilities?.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|$)',
            r'(revolving\s+credit.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|$)',
            r'(term\s+loan.*?)(?=\n[A-Z][A-Z\s]{10,}|\n\d+\.|$)'
        ]

        for pattern in facility_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                facility_text = match.group(1)
                parsed_facility = self._parse_facility_text(facility_text)

                if parsed_facility:
                    facility_type = parsed_facility.get('type', 'other')
                    if 'revolving' in facility_type.lower():
                        facilities['revolving_facilities'].append(parsed_facility)
                    elif 'term' in facility_type.lower():
                        facilities['term_facilities'].append(parsed_facility)

        return facilities

    def _parse_facility_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse credit facility details"""
        facility = {
            'type': self._identify_instrument_type(text),
            'committed_amount': None,
            'drawn_amount': None,
            'maturity': self._extract_maturity(text),
            'interest_rate': self._extract_interest_rate(text)
        }

        # Extract amounts
        amount_matches = re.finditer(r'\$([0-9,]+(?:\.[0-9]+)?)', text)
        amounts = []
        for match in amount_matches:
            amount = self._parse_amount(match.group())
            if amount:
                amounts.append(amount)

        if amounts:
            facility['committed_amount'] = max(amounts)  # Assume largest amount is committed

        return facility if facility['committed_amount'] else None

    def _create_debt_summary(self, debt_schedule: Dict) -> Dict[str, Any]:
        """Create summary statistics for debt schedule"""
        instruments = debt_schedule.get('debt_instruments', [])
        maturity_profile = debt_schedule.get('maturity_profile', {})

        summary = {
            'total_instruments': len(instruments),
            'total_debt_amount': maturity_profile.get('total_debt', 0),
            'weighted_average_maturity': maturity_profile.get('weighted_average_maturity', 0),
            'debt_by_type': {},
            'near_term_maturities': 0,  # Next 2 years
            'long_term_debt': 0,  # > 2 years
            'extraction_quality': 'High' if len(instruments) >= 3 else 'Medium' if len(instruments) >= 1 else 'Low'
        }

        # Categorize by instrument type
        for instrument in instruments:
            inst_type = instrument.get('instrument_type', 'Other')
            amount = instrument.get('amount_numeric', 0)

            if inst_type not in summary['debt_by_type']:
                summary['debt_by_type'][inst_type] = 0
            summary['debt_by_type'][inst_type] += amount

        # Calculate near-term vs long-term
        current_year = datetime.now().year
        for year_str, amount in maturity_profile.get('by_year', {}).items():
            try:
                year = int(year_str)
                if year <= current_year + 2:
                    summary['near_term_maturities'] += amount
                else:
                    summary['long_term_debt'] += amount
            except ValueError:
                continue

        return summary

    def _deduplicate_instruments(self, instruments: List[Dict]) -> List[Dict]:
        """Remove duplicate debt instruments"""
        # Simple deduplication based on amount and type
        seen = set()
        unique_instruments = []

        for instrument in instruments:
            key = (
                instrument.get('amount_numeric', 0),
                instrument.get('instrument_type', ''),
                instrument.get('maturity_date', '')
            )

            if key not in seen:
                seen.add(key)
                unique_instruments.append(instrument)

        return unique_instruments


def test_debt_schedule_extractor():
    """Test debt schedule extraction"""
    extractor = DebtScheduleExtractor()

    # Test with AAPL (should have filing available)
    print("="*60)
    print("TESTING DEBT SCHEDULE EXTRACTOR")
    print("="*60)

    debt_schedule = extractor.extract_debt_schedule('AAPL')

    if debt_schedule['debt_instruments']:
        print(f"\n📋 Debt Schedule Summary for AAPL:")
        summary = debt_schedule['summary']
        print(f"Total instruments: {summary.get('total_instruments', 0)}")
        print(f"Total debt: ${summary.get('total_debt_amount', 0)/1e9:.1f}B")
        print(f"Weighted avg maturity: {summary.get('weighted_average_maturity', 0):.1f} years")

        print(f"\n📅 Maturity Profile:")
        for year, amount in debt_schedule['maturity_profile'].get('by_year', {}).items():
            print(f"  {year}: ${amount/1e9:.1f}B")

        print(f"\n🏦 Debt by Type:")
        for debt_type, amount in summary.get('debt_by_type', {}).items():
            print(f"  {debt_type}: ${amount/1e9:.1f}B")

    else:
        print("❌ No debt instruments extracted")


if __name__ == "__main__":
    test_debt_schedule_extractor()