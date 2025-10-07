"""
Enhanced news scraper with historical date range support
Fetches 7-30 days of historical news data for comprehensive analysis
"""

import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging
from urllib.parse import quote_plus
import time
import json

logger = logging.getLogger(__name__)

class EnhancedNewsProvider:
    """Enhanced news provider with historical data support"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def get_yahoo_finance_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Get news from Yahoo Finance with date range support"""
        headlines = []
        try:
            # Yahoo Finance news endpoint
            url = f"https://query1.finance.yahoo.com/v1/finance/search"
            params = {
                'q': ticker,
                'quotesCount': 0,
                'newsCount': 50,  # Get more news items
                'enableFuzzyQuery': False,
                'quotesQueryId': 'tss_match_phrase_query',
                'multiQuoteQueryId': 'multi_quote_single_token_query',
                'newsQueryId': 'news_cie_vespa',
                'enableCb': True,
                'enableNavLinks': True,
                'enableEnhancedTrivialQuery': True
            }

            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                news_items = data.get('news', [])

                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

                for item in news_items:
                    try:
                        pub_time = item.get('providerPublishTime', 0)
                        if pub_time:
                            pub_date = datetime.fromtimestamp(pub_time, timezone.utc)

                            if pub_date >= cutoff_date:
                                headlines.append({
                                    'ticker': ticker.upper(),
                                    'headline': item.get('title', ''),
                                    'summary': item.get('summary', ''),
                                    'publisher': item.get('publisher', 'Yahoo Finance'),
                                    'url': item.get('link', ''),
                                    'published_at': pub_date.isoformat(),
                                    'source': 'yahoo_finance'
                                })
                    except Exception as e:
                        logger.debug(f"Error processing Yahoo news item: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Failed to fetch Yahoo Finance news: {e}")

        return headlines

    def get_marketwatch_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Get news from MarketWatch with date range"""
        headlines = []
        try:
            # MarketWatch search with date filtering
            for page in range(1, 4):  # Get multiple pages
                url = f"https://www.marketwatch.com/search"
                params = {
                    'q': ticker,
                    'tab': 'All News',
                    'sort': 'date',
                    'page': page
                }

                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')

                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

                    articles = soup.find_all('div', class_='searchresult')
                    if not articles:
                        break

                    for article in articles:
                        try:
                            title_elem = article.find('h3') or article.find('a')
                            if not title_elem:
                                continue

                            title = title_elem.get_text().strip()
                            link = title_elem.get('href', '') if title_elem.name == 'a' else ''

                            # Try to extract date
                            date_elem = article.find('span', class_='date')
                            if date_elem:
                                date_text = date_elem.get_text().strip()
                                # Parse relative dates like "2 days ago"
                                pub_date = self._parse_relative_date(date_text)

                                if pub_date and pub_date >= cutoff_date:
                                    headlines.append({
                                        'ticker': ticker.upper(),
                                        'headline': title,
                                        'summary': '',
                                        'publisher': 'MarketWatch',
                                        'url': link if link.startswith('http') else f"https://www.marketwatch.com{link}",
                                        'published_at': pub_date.isoformat(),
                                        'source': 'marketwatch'
                                    })
                        except Exception as e:
                            logger.debug(f"Error processing MarketWatch article: {e}")
                            continue

                time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.warning(f"Failed to fetch MarketWatch news: {e}")

        return headlines

    def get_reuters_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Get news from Reuters with date filtering"""
        headlines = []
        try:
            # Reuters search API
            company_name = self._ticker_to_company_name(ticker)
            search_term = f"{ticker} OR {company_name}"

            url = "https://www.reuters.com/pf/api/v3/content/fetch/articles-by-search-v2"
            params = {
                'query': search_term,
                'sort': 'display_date:desc',
                'size': 30,
                'from': 0
            }

            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('result', {}).get('articles', [])

                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

                for article in articles:
                    try:
                        pub_date_str = article.get('display_date', '')
                        if pub_date_str:
                            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))

                            if pub_date >= cutoff_date:
                                headlines.append({
                                    'ticker': ticker.upper(),
                                    'headline': article.get('headlines', {}).get('basic', ''),
                                    'summary': article.get('description', {}).get('basic', ''),
                                    'publisher': 'Reuters',
                                    'url': f"https://www.reuters.com{article.get('canonical_url', '')}",
                                    'published_at': pub_date.isoformat(),
                                    'source': 'reuters'
                                })
                    except Exception as e:
                        logger.debug(f"Error processing Reuters article: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Failed to fetch Reuters news: {e}")

        return headlines

    def get_seeking_alpha_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Get news from Seeking Alpha with historical data"""
        headlines = []
        try:
            # Seeking Alpha news API
            url = f"https://seekingalpha.com/api/v3/news"
            params = {
                'filter[since]': int((datetime.now() - timedelta(days=days_back)).timestamp()),
                'filter[until]': int(datetime.now().timestamp()),
                'filter[symbols]': ticker.upper(),
                'page[size]': 40,
                'include': 'author,primaryTickers,secondaryTickers'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }

            response = self.session.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])

                for article in articles:
                    try:
                        attributes = article.get('attributes', {})
                        pub_date_str = attributes.get('publishOn', '')

                        if pub_date_str:
                            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))

                            headlines.append({
                                'ticker': ticker.upper(),
                                'headline': attributes.get('title', ''),
                                'summary': attributes.get('summary', ''),
                                'publisher': 'Seeking Alpha',
                                'url': f"https://seekingalpha.com{attributes.get('uri', '')}",
                                'published_at': pub_date.isoformat(),
                                'source': 'seeking_alpha'
                            })
                    except Exception as e:
                        logger.debug(f"Error processing Seeking Alpha article: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Failed to fetch Seeking Alpha news: {e}")

        return headlines

    def _parse_relative_date(self, date_text: str) -> Optional[datetime]:
        """Parse relative dates like '2 days ago', 'yesterday', etc."""
        try:
            now = datetime.now(timezone.utc)
            date_text = date_text.lower().strip()

            if 'hour' in date_text or 'minute' in date_text or 'just now' in date_text:
                return now
            elif 'yesterday' in date_text:
                return now - timedelta(days=1)
            elif 'day' in date_text:
                # Extract number of days
                import re
                match = re.search(r'(\d+)\s*day', date_text)
                if match:
                    days = int(match.group(1))
                    return now - timedelta(days=days)
            elif 'week' in date_text:
                import re
                match = re.search(r'(\d+)\s*week', date_text)
                if match:
                    weeks = int(match.group(1))
                    return now - timedelta(weeks=weeks)

            return now  # Default to now if can't parse
        except:
            return None

    def _ticker_to_company_name(self, ticker: str) -> str:
        """Simple ticker to company name mapping"""
        mapping = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan',
            'JNJ': 'Johnson Johnson',
            'V': 'Visa',
            'PG': 'Procter Gamble',
            'UNH': 'UnitedHealth',
            'HD': 'Home Depot',
            'MA': 'Mastercard',
            'BAC': 'Bank of America',
            'CVS': 'CVS Health'
        }
        return mapping.get(ticker.upper(), ticker)

    def fetch_comprehensive_news(self, ticker: str, days_back: int = 7, max_articles: int = 50) -> List[Dict]:
        """Fetch comprehensive news from multiple sources with date range"""
        print(f"🔍 Fetching {days_back}-day news history for {ticker}...")

        all_headlines = []

        # Fetch from all sources
        sources = [
            ("Yahoo Finance", self.get_yahoo_finance_news),
            ("MarketWatch", self.get_marketwatch_news),
            ("Reuters", self.get_reuters_news),
            ("Seeking Alpha", self.get_seeking_alpha_news),
        ]

        for source_name, fetch_func in sources:
            try:
                print(f"  📰 Fetching from {source_name}...")
                headlines = fetch_func(ticker, days_back)
                all_headlines.extend(headlines)
                print(f"     ✅ Found {len(headlines)} articles")
                time.sleep(1)  # Rate limiting between sources
            except Exception as e:
                print(f"     ❌ {source_name} failed: {e}")
                continue

        # Remove duplicates and sort by date
        seen_titles = set()
        unique_headlines = []

        for headline in all_headlines:
            title = headline.get('headline', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_headlines.append(headline)

        # Sort by publication date (newest first)
        unique_headlines.sort(key=lambda x: x.get('published_at', ''), reverse=True)

        # Limit results
        final_headlines = unique_headlines[:max_articles]

        print(f"🎯 Total unique articles: {len(final_headlines)} (last {days_back} days)")
        return final_headlines

def test_enhanced_news():
    """Test the enhanced news scraper"""
    provider = EnhancedNewsProvider()

    # Test with different date ranges
    for days in [3, 7, 14]:
        print(f"\n{'='*60}")
        print(f"Testing {days}-day news coverage for AAPL")
        print('='*60)

        headlines = provider.fetch_comprehensive_news('AAPL', days_back=days, max_articles=20)

        # Show date distribution
        dates = {}
        for headline in headlines:
            try:
                pub_date = datetime.fromisoformat(headline['published_at'].replace('Z', '+00:00'))
                date_key = pub_date.strftime('%Y-%m-%d')
                dates[date_key] = dates.get(date_key, 0) + 1
            except:
                continue

        print(f"\n📅 Date distribution:")
        for date, count in sorted(dates.items(), reverse=True):
            print(f"  {date}: {count} articles")

        print(f"\n📰 Sample headlines:")
        for i, headline in enumerate(headlines[:5]):
            date = headline['published_at'][:10]
            source = headline['source']
            title = headline['headline'][:60]
            print(f"  {i+1}. [{source}] {date} - {title}...")

if __name__ == "__main__":
    test_enhanced_news()