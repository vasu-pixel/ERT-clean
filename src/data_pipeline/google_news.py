"""
Google News RSS scraper for comprehensive historical news coverage
Provides 7-30 days of historical news data using Google News RSS feeds
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging
from urllib.parse import quote_plus
import time
import re

logger = logging.getLogger(__name__)

class GoogleNewsProvider:
    """Google News RSS provider for historical news coverage"""

    def __init__(self):
        self.base_url = "https://news.google.com/rss/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def fetch_google_news(self, ticker: str, days_back: int = 7, max_articles: int = 50) -> List[Dict]:
        """Fetch news from Google News RSS with date filtering"""
        headlines = []

        try:
            # Search terms for better coverage
            search_terms = [
                f"{ticker}",
                f"{ticker} stock",
                f"{ticker} earnings",
                f"{ticker} news",
                self._get_company_name(ticker)
            ]

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

            for search_term in search_terms:
                try:
                    # Google News RSS parameters
                    params = {
                        'q': search_term,
                        'hl': 'en-US',
                        'gl': 'US',
                        'ceid': 'US:en'
                    }

                    # Add date range (when parameter)
                    # Google News supports: 1h, 1d, 7d, 1m
                    if days_back <= 1:
                        params['when'] = '1d'
                    elif days_back <= 7:
                        params['when'] = '7d'
                    else:
                        params['when'] = '1m'

                    print(f"  🔍 Searching Google News for: '{search_term}' (last {params['when']})")

                    response = self.session.get(self.base_url, params=params, timeout=15)
                    if response.status_code == 200:
                        articles = self._parse_rss_feed(response.text, ticker, cutoff_date)
                        headlines.extend(articles)
                        print(f"     ✅ Found {len(articles)} articles")
                    else:
                        print(f"     ⚠️ HTTP {response.status_code}")

                    time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    print(f"     ❌ Search term '{search_term}' failed: {e}")
                    continue

            # Remove duplicates by URL and title
            unique_headlines = self._deduplicate_articles(headlines)

            # Sort by publication date (newest first)
            unique_headlines.sort(key=lambda x: x.get('published_at', ''), reverse=True)

            # Limit results
            final_headlines = unique_headlines[:max_articles]

            print(f"  🎯 Total unique Google News articles: {len(final_headlines)}")
            return final_headlines

        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")
            return []

    def _parse_rss_feed(self, rss_text: str, ticker: str, cutoff_date: datetime) -> List[Dict]:
        """Parse Google News RSS feed"""
        articles = []

        try:
            root = ET.fromstring(rss_text)

            # Find all item elements
            for item in root.findall('.//item'):
                try:
                    title = item.find('title')
                    title_text = title.text if title is not None else ''

                    # Skip if ticker not in title (loose filtering)
                    if ticker.upper() not in title_text.upper() and not self._is_relevant_article(title_text, ticker):
                        continue

                    link = item.find('link')
                    link_url = link.text if link is not None else ''

                    description = item.find('description')
                    desc_text = description.text if description is not None else ''

                    # Parse publication date
                    pub_date = item.find('pubDate')
                    published_at = None
                    if pub_date is not None and pub_date.text:
                        try:
                            # Parse RSS date format: "Mon, 30 Sep 2024 14:30:00 GMT"
                            published_at = datetime.strptime(pub_date.text, '%a, %d %b %Y %H:%M:%S %Z')
                            published_at = published_at.replace(tzinfo=timezone.utc)
                        except:
                            try:
                                # Alternative format
                                published_at = datetime.strptime(pub_date.text, '%a, %d %b %Y %H:%M:%S %z')
                            except:
                                published_at = datetime.now(timezone.utc)

                    # Filter by date
                    if published_at and published_at >= cutoff_date:
                        # Extract source from description or URL
                        source = self._extract_source(desc_text, link_url)

                        articles.append({
                            'ticker': ticker.upper(),
                            'headline': self._clean_title(title_text),
                            'summary': self._clean_description(desc_text),
                            'publisher': source,
                            'url': link_url,
                            'published_at': published_at.isoformat(),
                            'source': 'google_news'
                        })

                except Exception as e:
                    logger.debug(f"Error parsing RSS item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing RSS feed: {e}")

        return articles

    def _is_relevant_article(self, title: str, ticker: str) -> bool:
        """Check if article is relevant to the ticker"""
        title_lower = title.lower()
        ticker_lower = ticker.lower()

        # Company name mapping
        company_names = {
            'aapl': ['apple', 'iphone', 'ipad', 'mac'],
            'msft': ['microsoft', 'windows', 'azure', 'office'],
            'googl': ['google', 'alphabet', 'youtube', 'android'],
            'amzn': ['amazon', 'aws', 'alexa'],
            'tsla': ['tesla', 'elon musk', 'electric vehicle'],
            'meta': ['meta', 'facebook', 'instagram', 'whatsapp'],
            'nvda': ['nvidia', 'ai chip', 'gpu'],
            'cvs': ['cvs health', 'pharmacy', 'aetna']
        }

        relevant_terms = company_names.get(ticker_lower, [ticker_lower])

        return any(term in title_lower for term in relevant_terms)

    def _clean_title(self, title: str) -> str:
        """Clean article title"""
        if not title:
            return ''

        # Remove common prefixes/suffixes
        title = re.sub(r'^.*?-\s*', '', title)  # Remove "Source - " prefix
        title = re.sub(r'\s*-\s*.*$', '', title)  # Remove " - Source" suffix

        return title.strip()

    def _clean_description(self, description: str) -> str:
        """Clean article description"""
        if not description:
            return ''

        # Remove HTML tags
        description = re.sub(r'<[^>]+>', '', description)

        # Remove Google News formatting
        description = re.sub(r'^.*?&nbsp;', '', description)

        return description.strip()

    def _extract_source(self, description: str, url: str) -> str:
        """Extract news source from description or URL"""
        # Try to extract from URL
        if 'reuters.com' in url:
            return 'Reuters'
        elif 'bloomberg.com' in url:
            return 'Bloomberg'
        elif 'wsj.com' in url:
            return 'Wall Street Journal'
        elif 'marketwatch.com' in url:
            return 'MarketWatch'
        elif 'cnbc.com' in url:
            return 'CNBC'
        elif 'yahoo.com' in url:
            return 'Yahoo Finance'
        elif 'seekingalpha.com' in url:
            return 'Seeking Alpha'
        elif 'benzinga.com' in url:
            return 'Benzinga'
        elif 'fool.com' in url:
            return 'Motley Fool'

        # Try to extract from description
        if description:
            # Look for source patterns
            source_patterns = [
                r'([A-Z][a-zA-Z\s&]+)\s*-\s*',
                r'By\s+([A-Z][a-zA-Z\s&]+)',
                r'Source:\s*([A-Z][a-zA-Z\s&]+)'
            ]

            for pattern in source_patterns:
                match = re.search(pattern, description)
                if match:
                    return match.group(1).strip()

        return 'Google News'

    def _get_company_name(self, ticker: str) -> str:
        """Get company name for search"""
        mapping = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft',
            'GOOGL': 'Alphabet Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta Facebook',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan Chase',
            'CVS': 'CVS Health',
            'JNJ': 'Johnson Johnson',
            'V': 'Visa',
            'PG': 'Procter Gamble',
            'UNH': 'UnitedHealth'
        }
        return mapping.get(ticker.upper(), ticker)

    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles"""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []

        for article in articles:
            url = article.get('url', '')
            title = article.get('headline', '').lower().strip()

            # Skip if we've seen this URL or very similar title
            if url in seen_urls or title in seen_titles:
                continue

            # Skip if title is too short or generic
            if len(title) < 10:
                continue

            seen_urls.add(url)
            seen_titles.add(title)
            unique_articles.append(article)

        return unique_articles


def test_google_news():
    """Test Google News provider"""
    provider = GoogleNewsProvider()

    # Test with different timeframes
    for days in [3, 7, 14, 30]:
        print(f"\n{'='*60}")
        print(f"Testing {days}-day Google News coverage for AAPL")
        print('='*60)

        headlines = provider.fetch_google_news('AAPL', days_back=days, max_articles=30)

        if headlines:
            # Show date distribution
            dates = {}
            for headline in headlines:
                try:
                    pub_date = datetime.fromisoformat(headline['published_at'].replace('Z', '+00:00'))
                    date_key = pub_date.strftime('%Y-%m-%d')
                    dates[date_key] = dates.get(date_key, 0) + 1
                except:
                    continue

            print(f"\n📅 Date distribution ({len(headlines)} total articles):")
            for date, count in sorted(dates.items(), reverse=True)[:10]:  # Show top 10 dates
                print(f"  {date}: {count} articles")

            print(f"\n📰 Sample headlines:")
            for i, headline in enumerate(headlines[:5]):
                date = headline['published_at'][:10]
                source = headline.get('publisher', 'Unknown')
                title = headline['headline'][:70]
                print(f"  {i+1}. [{source}] {date} - {title}...")

        else:
            print("❌ No articles found")

        time.sleep(2)  # Rate limiting between tests


if __name__ == "__main__":
    test_google_news()