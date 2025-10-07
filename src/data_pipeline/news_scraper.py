"""Lightweight headline scrapers for reputable finance outlets.

These scrapers rely on public HTML pages and should be treated as best-effort
fallbacks when structured APIs are unavailable. They intentionally avoid using
third-party dependencies beyond ``requests`` and ``BeautifulSoup``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional

import logging
import re
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
TIMEOUT = 10


@dataclass
class ScrapedHeadline:
    ticker: str
    headline: str
    summary: Optional[str]
    publisher: str
    url: str
    published_at: Optional[str]

    def as_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


def _fetch_html(url: str) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.debug("Unable to fetch %s: %s", url, exc)
        return None


def _parse_reuters(ticker: str) -> List[ScrapedHeadline]:
    url = f"https://www.reuters.com/markets/companies/{ticker.upper()}"
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    articles = soup.select("article.story") or soup.select("article[data-testid='MarketStory']")

    headlines: List[ScrapedHeadline] = []
    for article in articles[:10]:
        title_tag = article.find("h3") or article.find("h2")
        if not title_tag or not title_tag.text:
            continue
        headline = title_tag.text.strip()

        link_tag = article.find("a", href=True)
        if link_tag:
            href = link_tag["href"]
            url_resolved = href if href.startswith("http") else f"https://www.reuters.com{href}"
        else:
            url_resolved = url

        time_tag = article.find("time")
        published_at = None
        if time_tag and time_tag.has_attr("datetime"):
            published_at = time_tag["datetime"]

        summary_tag = article.find("p")
        summary = summary_tag.text.strip() if summary_tag and summary_tag.text else None

        headlines.append(
            ScrapedHeadline(
                ticker=ticker.upper(),
                headline=headline,
                summary=summary,
                publisher="Reuters",
                url=url_resolved,
                published_at=published_at,
            )
        )

    return headlines


def _parse_marketwatch(ticker: str) -> List[ScrapedHeadline]:
    url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    articles = soup.select("div.article__content")

    headlines: List[ScrapedHeadline] = []
    for block in articles[:10]:
        title_tag = block.find("a", class_=re.compile("link"))
        if not title_tag or not title_tag.text:
            continue
        headline = title_tag.text.strip()
        href = title_tag["href"] if title_tag.has_attr("href") else url

        meta = block.find("span", class_=re.compile("timestamp"))
        published_at = meta.text.strip() if meta else None

        summary_tag = block.find("p")
        summary = summary_tag.text.strip() if summary_tag and summary_tag.text else None

        headlines.append(
            ScrapedHeadline(
                ticker=ticker.upper(),
                headline=headline,
                summary=summary,
                publisher="MarketWatch",
                url=href,
                published_at=published_at,
            )
        )

    return headlines


def _parse_generic_rss(url: str, ticker: str, publisher: str) -> List[ScrapedHeadline]:
    xml = _fetch_html(url)
    if not xml:
        return []

    soup = BeautifulSoup(xml, "xml")
    items = soup.find_all("item")
    headlines: List[ScrapedHeadline] = []

    for item in items[:10]:
        title_tag = item.find("title")
        if not title_tag or not title_tag.text:
            continue
        title = title_tag.text.strip()

        link_tag = item.find("link")
        link = link_tag.text.strip() if link_tag and link_tag.text else url

        description_tag = item.find("description") or item.find("summary")
        summary = description_tag.text.strip() if description_tag and description_tag.text else None

        pub_tag = item.find("pubDate") or item.find("published") or item.find("updated")
        published_at = pub_tag.text.strip() if pub_tag and pub_tag.text else None

        headlines.append(
            ScrapedHeadline(
                ticker=ticker.upper(),
                headline=title,
                summary=summary,
                publisher=publisher,
                url=link,
                published_at=published_at,
            )
        )

    return headlines


def _parse_yahoo_finance(ticker: str) -> List[ScrapedHeadline]:
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s="
        f"{quote_plus(ticker)}&region=US&lang=en-US"
    )
    return _parse_generic_rss(url, ticker, "Yahoo Finance")


def _parse_nasdaq(ticker: str) -> List[ScrapedHeadline]:
    url = f"https://www.nasdaq.com/feed/rssoutbound?symbol={quote_plus(ticker.upper())}"
    return _parse_generic_rss(url, ticker, "Nasdaq")


def _parse_cnbc(ticker: str) -> List[ScrapedHeadline]:
    url = f"https://www.cnbc.com/quotes/{quote_plus(ticker.upper())}?tab=news"
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    articles = soup.select("div.QuoteNewsFeed-story")

    headlines: List[ScrapedHeadline] = []
    for article in articles[:10]:
        title_tag = article.find("a", class_=re.compile("QuoteNewsFeed-link"))
        if not title_tag or not title_tag.text:
            continue
        headline = title_tag.text.strip()
        link = title_tag.get("href", url)
        if link.startswith("//"):
            link = f"https:{link}"

        summary_tag = article.find("div", class_=re.compile("QuoteNewsFeed-description"))
        summary = summary_tag.text.strip() if summary_tag and summary_tag.text else None

        time_tag = article.find("time")
        published_at = time_tag.get("datetime") if time_tag and time_tag.has_attr("datetime") else None

        headlines.append(
            ScrapedHeadline(
                ticker=ticker.upper(),
                headline=headline,
                summary=summary,
                publisher="CNBC",
                url=link,
                published_at=published_at,
            )
        )

    return headlines


def _parse_seeking_alpha(ticker: str) -> List[ScrapedHeadline]:
    url = f"https://seekingalpha.com/api/sa/combined/{quote_plus(ticker.lower())}.xml"
    headlines = _parse_generic_rss(url, ticker, "Seeking Alpha")
    return headlines


def _parse_bizwire(ticker: str) -> List[ScrapedHeadline]:
    base_url = (
        "https://www.businesswire.com/portal/site/home/template.NWS/"
        "?javax.portlet.prp_e652b8350a0a47ee8bde58d54feede13=view-all&newsLang=en&searchType=all"
    )
    url = f"{base_url}&keywords={quote_plus(ticker)}"
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    articles = soup.select("div.release")
    headlines: List[ScrapedHeadline] = []

    for article in articles[:10]:
        title_tag = article.find("a", class_="bwTitleLink")
        if not title_tag or not title_tag.text:
            continue
        headline = title_tag.text.strip()
        link = title_tag.get("href", url)
        if not link.startswith("http"):
            link = f"https://www.businesswire.com{link}"

        summary_tag = article.find("div", class_="bwReleaseSummary")
        summary = summary_tag.text.strip() if summary_tag and summary_tag.text else None

        date_tag = article.find("span", class_="bwTimeStmp")
        published_at = date_tag.text.strip() if date_tag and date_tag.text else None

        headlines.append(
            ScrapedHeadline(
                ticker=ticker.upper(),
                headline=headline,
                summary=summary,
                publisher="Business Wire",
                url=link,
                published_at=published_at,
            )
        )

    return headlines


SCRAPERS: Dict[str, Callable[[str], List[ScrapedHeadline]]] = {
    "reuters": _parse_reuters,
    "marketwatch": _parse_marketwatch,
    "yahoo": _parse_yahoo_finance,
    "nasdaq": _parse_nasdaq,
    "cnbc": _parse_cnbc,
    "seekingalpha": _parse_seeking_alpha,
    "businesswire": _parse_bizwire,
}


def scrape_headlines(ticker: str, sources: Optional[Iterable[str]] = None, limit: int = 12) -> List[Dict[str, Optional[str]]]:
    """Scrape headlines from configured publishers.

    Args:
        ticker: The equity ticker.
        sources: Iterable of scraper keys to use. Defaults to all registered.
        limit: Maximum combined headlines to return.
    """
    sources = list(sources) if sources else list(SCRAPERS.keys())
    aggregated: List[ScrapedHeadline] = []

    for source in sources:
        scraper = SCRAPERS.get(source)
        if not scraper:
            continue
        try:
            headlines = scraper(ticker)
            aggregated.extend(headlines)
        except Exception as exc:
            logger.debug("Scraper %s failed for %s: %s", source, ticker, exc)

        if len(aggregated) >= limit:
            break

    # De-duplicate by URL while preserving order
    deduped: Dict[str, ScrapedHeadline] = {}
    for item in aggregated:
        key = item.url
        if key not in deduped:
            deduped[key] = item

    aggregated_unique = list(deduped.values())

    def _sort_key(section: ScrapedHeadline) -> str:
        return section.published_at or ""

    aggregated_unique.sort(key=_sort_key, reverse=True)
    return [item.as_dict() for item in aggregated_unique[:limit]]


def demo(ticker: str = "CVS") -> None:
    """CLI demo for quick manual testing."""
    headlines = scrape_headlines(ticker)
    print(f"Scraped {len(headlines)} headlines for {ticker}")
    for idx, item in enumerate(headlines, 1):
        print(f"[{idx}] {item['headline']} ({item['publisher']})")
        print(f"     {item['url']}")
        print(f"     Sentiment label: {item.get('sentiment_label')}")
        print()


if __name__ == "__main__":
    demo()
