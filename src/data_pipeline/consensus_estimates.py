"""
Consensus Estimates Provider
Aggregates analyst estimates from multiple sources for revenue, EPS, and growth projections
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import time
from bs4 import BeautifulSoup
import json

logger = logging.getLogger(__name__)

class ConsensusEstimatesProvider:
    """Provides analyst consensus estimates from multiple sources"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def get_consensus_estimates(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive consensus estimates for a ticker"""

        print(f"📊 Fetching consensus estimates for {ticker}...")

        estimates = {
            'ticker': ticker.upper(),
            'as_of': datetime.now().isoformat(),
            'revenue_estimates': {},
            'eps_estimates': {},
            'growth_estimates': {},
            'analyst_ratings': {},
            'price_targets': {},
            'revisions': {},
            'coverage': {}
        }

        # Aggregate from multiple sources
        sources = [
            ("Yahoo Finance", self._get_yahoo_estimates),
            ("MarketWatch", self._get_marketwatch_estimates),
            ("Zacks", self._get_zacks_estimates),
            ("Seeking Alpha", self._get_seekingalpha_estimates)
        ]

        for source_name, fetch_func in sources:
            try:
                print(f"  📈 Fetching from {source_name}...")
                source_data = fetch_func(ticker)
                if source_data:
                    estimates = self._merge_estimates(estimates, source_data, source_name)
                    print(f"     ✅ {source_name}: {len(source_data.get('revenue_estimates', {}))} revenue estimates")
                else:
                    print(f"     ⚠️ {source_name}: No data available")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"     ❌ {source_name} failed: {e}")
                continue

        # Calculate consensus metrics
        estimates['consensus'] = self._calculate_consensus(estimates)

        print(f"🎯 Consensus summary: {len(estimates['consensus'].get('revenue', {}))} revenue estimates")
        return estimates

    def _get_yahoo_estimates(self, ticker: str) -> Optional[Dict]:
        """Get estimates from Yahoo Finance"""
        try:
            # Yahoo Finance Analysis page
            url = f"https://finance.yahoo.com/quote/{ticker}/analysis"
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            estimates = {
                'revenue_estimates': {},
                'eps_estimates': {},
                'growth_estimates': {}
            }

            # Parse earnings estimate table
            tables = soup.find_all('table')
            for table in tables:
                if 'Earnings Estimate' in str(table):
                    estimates['eps_estimates'] = self._parse_yahoo_estimate_table(table, 'eps')
                elif 'Revenue Estimate' in str(table):
                    estimates['revenue_estimates'] = self._parse_yahoo_estimate_table(table, 'revenue')
                elif 'Growth Estimates' in str(table):
                    estimates['growth_estimates'] = self._parse_yahoo_estimate_table(table, 'growth')

            return estimates

        except Exception as e:
            logger.debug(f"Yahoo estimates failed: {e}")
            return None

    def _get_marketwatch_estimates(self, ticker: str) -> Optional[Dict]:
        """Get estimates from MarketWatch"""
        try:
            url = f"https://www.marketwatch.com/investing/stock/{ticker}/analystestimates"
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            estimates = {
                'revenue_estimates': {},
                'eps_estimates': {},
                'price_targets': {}
            }

            # Parse estimate tables
            tables = soup.find_all('table', class_='table')
            for table in tables:
                estimates.update(self._parse_marketwatch_table(table))

            return estimates

        except Exception as e:
            logger.debug(f"MarketWatch estimates failed: {e}")
            return None

    def _get_zacks_estimates(self, ticker: str) -> Optional[Dict]:
        """Get estimates from Zacks"""
        try:
            url = f"https://www.zacks.com/stock/quote/{ticker}/detailed-estimates"
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            estimates = {
                'revenue_estimates': {},
                'eps_estimates': {},
                'revisions': {}
            }

            # Parse Zacks estimate data
            estimate_sections = soup.find_all('div', class_='zacks_table')
            for section in estimate_sections:
                estimates.update(self._parse_zacks_section(section))

            return estimates

        except Exception as e:
            logger.debug(f"Zacks estimates failed: {e}")
            return None

    def _get_seekingalpha_estimates(self, ticker: str) -> Optional[Dict]:
        """Get estimates from Seeking Alpha"""
        try:
            url = f"https://seekingalpha.com/symbol/{ticker}/earnings/estimates"
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            # Parse JSON data from page
            content = response.text
            if 'window.SSR_DATA' in content:
                start = content.find('window.SSR_DATA = ') + 18
                end = content.find(';</script>', start)
                json_data = content[start:end]
                data = json.loads(json_data)

                estimates = self._parse_seekingalpha_data(data, ticker)
                return estimates

        except Exception as e:
            logger.debug(f"Seeking Alpha estimates failed: {e}")
            return None

    def _parse_yahoo_estimate_table(self, table, estimate_type: str) -> Dict:
        """Parse Yahoo Finance estimate table"""
        estimates = {}
        try:
            rows = table.find_all('tr')
            headers = [th.get_text().strip() for th in rows[0].find_all('th')] if rows else []

            for row in rows[1:]:
                cells = [td.get_text().strip() for td in row.find_all('td')]
                if len(cells) >= 2:
                    period = cells[0]
                    values = {}
                    for i, header in enumerate(headers[1:], 1):
                        if i < len(cells):
                            values[header.lower().replace(' ', '_')] = self._parse_numeric(cells[i])
                    estimates[period] = values
        except Exception as e:
            logger.debug(f"Yahoo table parsing failed: {e}")

        return estimates

    def _parse_marketwatch_table(self, table) -> Dict:
        """Parse MarketWatch estimate table"""
        estimates = {}
        try:
            # Implementation for MarketWatch specific parsing
            pass
        except Exception as e:
            logger.debug(f"MarketWatch table parsing failed: {e}")
        return estimates

    def _parse_zacks_section(self, section) -> Dict:
        """Parse Zacks estimate section"""
        estimates = {}
        try:
            # Implementation for Zacks specific parsing
            pass
        except Exception as e:
            logger.debug(f"Zacks section parsing failed: {e}")
        return estimates

    def _parse_seekingalpha_data(self, data: Dict, ticker: str) -> Dict:
        """Parse Seeking Alpha JSON data"""
        estimates = {}
        try:
            # Implementation for Seeking Alpha JSON parsing
            pass
        except Exception as e:
            logger.debug(f"Seeking Alpha data parsing failed: {e}")
        return estimates

    def _parse_numeric(self, value: str) -> Optional[float]:
        """Parse numeric value from string"""
        try:
            # Remove currency symbols, commas, and percentages
            clean = value.replace('$', '').replace(',', '').replace('%', '').strip()
            if clean in ['N/A', '-', '', 'NaN']:
                return None

            # Handle billions/millions notation
            if 'B' in clean:
                return float(clean.replace('B', '')) * 1e9
            elif 'M' in clean:
                return float(clean.replace('M', '')) * 1e6
            else:
                return float(clean)
        except (ValueError, AttributeError):
            return None

    def _merge_estimates(self, base_estimates: Dict, source_data: Dict, source_name: str) -> Dict:
        """Merge estimates from different sources"""
        for category in ['revenue_estimates', 'eps_estimates', 'growth_estimates']:
            if category in source_data:
                for period, values in source_data[category].items():
                    if category not in base_estimates:
                        base_estimates[category] = {}
                    if period not in base_estimates[category]:
                        base_estimates[category][period] = {}

                    # Add source attribution
                    for metric, value in values.items():
                        metric_key = f"{metric}_{source_name.lower().replace(' ', '_')}"
                        base_estimates[category][period][metric_key] = value

        return base_estimates

    def _calculate_consensus(self, estimates: Dict) -> Dict:
        """Calculate consensus from multiple sources"""
        consensus = {
            'revenue': {},
            'eps': {},
            'growth': {},
            'summary': {}
        }

        # Calculate consensus revenue estimates
        for period, data in estimates.get('revenue_estimates', {}).items():
            values = [v for k, v in data.items() if v is not None and 'estimate' in k.lower()]
            if values:
                consensus['revenue'][period] = {
                    'mean': sum(values) / len(values),
                    'median': sorted(values)[len(values)//2],
                    'high': max(values),
                    'low': min(values),
                    'count': len(values)
                }

        # Calculate consensus EPS estimates
        for period, data in estimates.get('eps_estimates', {}).items():
            values = [v for k, v in data.items() if v is not None and 'estimate' in k.lower()]
            if values:
                consensus['eps'][period] = {
                    'mean': sum(values) / len(values),
                    'median': sorted(values)[len(values)//2],
                    'high': max(values),
                    'low': min(values),
                    'count': len(values)
                }

        # Summary statistics
        revenue_estimates = list(consensus['revenue'].values())
        eps_estimates = list(consensus['eps'].values())

        consensus['summary'] = {
            'total_revenue_estimates': len(revenue_estimates),
            'total_eps_estimates': len(eps_estimates),
            'coverage_quality': 'High' if len(revenue_estimates) >= 3 else 'Medium' if len(revenue_estimates) >= 2 else 'Low',
            'last_updated': datetime.now().isoformat()
        }

        return consensus


def test_consensus_estimates():
    """Test consensus estimates provider"""
    provider = ConsensusEstimatesProvider()

    # Test with major stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"Testing consensus estimates for {ticker}")
        print('='*60)

        estimates = provider.get_consensus_estimates(ticker)

        # Display results
        consensus = estimates.get('consensus', {})
        print(f"\n📊 Consensus Summary for {ticker}:")
        print(f"Revenue estimates: {consensus.get('summary', {}).get('total_revenue_estimates', 0)}")
        print(f"EPS estimates: {consensus.get('summary', {}).get('total_eps_estimates', 0)}")
        print(f"Coverage quality: {consensus.get('summary', {}).get('coverage_quality', 'Unknown')}")

        # Show sample revenue estimates
        revenue_consensus = consensus.get('revenue', {})
        if revenue_consensus:
            print(f"\n📈 Revenue Consensus (sample):")
            for period, data in list(revenue_consensus.items())[:3]:
                mean = data.get('mean', 0)
                count = data.get('count', 0)
                print(f"  {period}: ${mean/1e9:.1f}B consensus ({count} estimates)")

        time.sleep(2)  # Rate limiting between tests


if __name__ == "__main__":
    test_consensus_estimates()