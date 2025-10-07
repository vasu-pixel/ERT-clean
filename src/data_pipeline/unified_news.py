"""
Unified news provider combining multiple sources for comprehensive coverage
Integrates Google News RSS, Yahoo Finance, and original scrapers
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline import sources  # Original scraper
from data_pipeline.google_news import GoogleNewsProvider
from data_pipeline.enhanced_news import EnhancedNewsProvider

logger = logging.getLogger(__name__)

class UnifiedNewsProvider:
    """Unified news provider with comprehensive historical coverage"""

    def __init__(self):
        self.google_provider = GoogleNewsProvider()
        self.enhanced_provider = EnhancedNewsProvider()

    def get_comprehensive_news(self, ticker: str, days_back: int = 7, max_articles: int = 50) -> List[Dict]:
        """Get comprehensive news from all available sources"""

        print(f"🔄 Fetching comprehensive {days_back}-day news for {ticker}...")
        all_headlines = []

        # 1. Google News RSS (best historical coverage)
        try:
            print("📰 Fetching from Google News RSS...")
            google_news = self.google_provider.fetch_google_news(ticker, days_back, max_articles//2)
            all_headlines.extend(google_news)
            print(f"   ✅ Google News: {len(google_news)} articles")
        except Exception as e:
            print(f"   ❌ Google News failed: {e}")

        # 2. Original scrapers (Nasdaq, Reuters, etc.)
        try:
            print("📰 Fetching from original scrapers...")
            original_news = sources.fetch_recent_headlines(ticker, limit=max_articles//4)

            # Convert to unified format
            for item in original_news:
                all_headlines.append({
                    'ticker': ticker.upper(),
                    'headline': item.get('title', item.get('headline', '')),
                    'summary': item.get('summary', ''),
                    'publisher': item.get('source', item.get('publisher', 'Unknown')),
                    'url': item.get('url', ''),
                    'published_at': item.get('published_at', datetime.now().isoformat()),
                    'source': 'original_scrapers',
                    'sentiment_score': item.get('sentiment_score', 0.0),
                    'sentiment_label': item.get('sentiment_label', 'neutral')
                })
            print(f"   ✅ Original scrapers: {len(original_news)} articles")
        except Exception as e:
            print(f"   ❌ Original scrapers failed: {e}")

        # 3. Enhanced Yahoo Finance (fallback)
        try:
            print("📰 Fetching from Enhanced Yahoo Finance...")
            yahoo_news = self.enhanced_provider.get_yahoo_finance_news(ticker, days_back)
            all_headlines.extend(yahoo_news)
            print(f"   ✅ Yahoo Finance: {len(yahoo_news)} articles")
        except Exception as e:
            print(f"   ❌ Yahoo Finance failed: {e}")

        # Deduplicate and process
        unique_headlines = self._process_headlines(all_headlines, max_articles)

        # Add sentiment scores to articles that don't have them
        self._add_sentiment_scores(unique_headlines)

        print(f"🎯 Final result: {len(unique_headlines)} unique articles")
        self._print_coverage_summary(unique_headlines)

        return unique_headlines

    def _process_headlines(self, headlines: List[Dict], max_articles: int) -> List[Dict]:
        """Process and deduplicate headlines"""

        # Remove duplicates by URL and similar titles
        seen_urls = set()
        seen_titles = set()
        unique_headlines = []

        for headline in headlines:
            url = headline.get('url', '')
            title = headline.get('headline', '').lower().strip()

            # Skip empty or very short titles
            if len(title) < 10:
                continue

            # Check for duplicates
            title_key = self._normalize_title(title)
            if url in seen_urls or title_key in seen_titles:
                continue

            seen_urls.add(url)
            seen_titles.add(title_key)
            unique_headlines.append(headline)

        # Sort by publication date (newest first)
        unique_headlines.sort(key=lambda x: x.get('published_at', ''), reverse=True)

        # Limit results
        return unique_headlines[:max_articles]

    def _normalize_title(self, title: str) -> str:
        """Normalize title for duplicate detection"""
        import re

        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = ' '.join(normalized.split())

        # Remove common words that might vary
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = [w for w in normalized.split() if w not in stop_words]

        return ' '.join(words)

    def _add_sentiment_scores(self, headlines: List[Dict]):
        """Add sentiment scores to headlines that don't have them"""
        for headline in headlines:
            if 'sentiment_score' not in headline or headline.get('sentiment_score') is None:
                # Use the sentiment scoring from original sources module
                text = f"{headline.get('headline', '')} {headline.get('summary', '')}"
                headline['sentiment_score'] = sources._score_sentiment(text)
                headline['sentiment_label'] = sources._label_sentiment(headline['sentiment_score'])

    def _print_coverage_summary(self, headlines: List[Dict]):
        """Print coverage summary"""
        if not headlines:
            print("   ⚠️ No articles found")
            return

        # Date distribution
        dates = {}
        sources_count = {}

        for headline in headlines:
            try:
                pub_date = datetime.fromisoformat(headline['published_at'].replace('Z', '+00:00'))
                date_key = pub_date.strftime('%Y-%m-%d')
                dates[date_key] = dates.get(date_key, 0) + 1
            except:
                pass

            source = headline.get('source', 'unknown')
            sources_count[source] = sources_count.get(source, 0) + 1

        print(f"\n📅 Date coverage:")
        for date, count in sorted(dates.items(), reverse=True)[:7]:
            print(f"   {date}: {count} articles")

        print(f"\n📊 Source breakdown:")
        for source, count in sorted(sources_count.items(), key=lambda x: x[1], reverse=True):
            print(f"   {source}: {count} articles")

        # Sentiment summary
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        for headline in headlines:
            label = headline.get('sentiment_label', 'neutral')
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

        print(f"\n😊 Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(headlines)) * 100
            print(f"   {sentiment}: {count} articles ({percentage:.1f}%)")


def test_unified_provider():
    """Test the unified news provider"""
    provider = UnifiedNewsProvider()

    # Test with AAPL
    print("=" * 80)
    print("TESTING UNIFIED NEWS PROVIDER")
    print("=" * 80)

    for days in [3, 7, 14]:
        print(f"\n{'='*60}")
        print(f"Testing {days}-day comprehensive coverage for AAPL")
        print('='*60)

        headlines = provider.get_comprehensive_news('AAPL', days_back=days, max_articles=30)

        if headlines:
            print(f"\n📰 Sample headlines:")
            for i, headline in enumerate(headlines[:5]):
                date = headline.get('published_at', '')[:10]
                source = headline.get('publisher', 'Unknown')
                title = headline.get('headline', 'No title')[:60]
                sentiment = headline.get('sentiment_label', 'neutral')
                score = headline.get('sentiment_score', 0.0)

                print(f"  {i+1}. [{source}] {date} - {title}...")
                print(f"     Sentiment: {sentiment} ({score:.2f})")

        time.sleep(2)  # Rate limiting


if __name__ == "__main__":
    test_unified_provider()