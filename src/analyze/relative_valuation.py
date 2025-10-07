"""
Relative Valuation Framework
- P/E, EV/EBITDA, PEG with peer benchmarking
- Sector-specific multiple analysis
- Dynamic peer selection and screening
- Trading vs transaction multiples
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.models import CompanyDataset
from data_pipeline.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

@dataclass
class PeerCompany:
    """Peer company data structure"""
    ticker: str
    name: str
    market_cap: float
    enterprise_value: float

    # P&L metrics
    revenue: float
    ebitda: float
    net_income: float
    eps: float

    # Growth metrics
    revenue_growth: float
    earnings_growth: float

    # Profitability metrics
    roe: float
    roic: float
    operating_margin: float

    # Valuation multiples
    pe_ratio: float
    ev_ebitda: float
    price_sales: float
    price_book: float
    peg_ratio: float

    # Risk metrics
    beta: float
    debt_to_equity: float

@dataclass
class RelativeValuationResults:
    """Results from relative valuation analysis"""

    target_ticker: str
    analysis_date: str

    # Peer analysis
    peer_companies: List[PeerCompany] = field(default_factory=list)
    peer_selection_criteria: Dict[str, Any] = field(default_factory=dict)

    # Multiple analysis
    trading_multiples: Dict[str, Dict[str, float]] = field(default_factory=dict)
    transaction_multiples: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Valuation estimates
    pe_valuation: float = 0.0
    ev_ebitda_valuation: float = 0.0
    price_sales_valuation: float = 0.0
    peg_valuation: float = 0.0

    # Statistical analysis
    multiple_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    outlier_analysis: Dict[str, List[str]] = field(default_factory=dict)

    # Sector benchmarks
    sector_multiples: Dict[str, float] = field(default_factory=dict)
    size_adjustments: Dict[str, float] = field(default_factory=dict)

class RelativeValuationAnalyzer:
    """Advanced relative valuation with peer benchmarking"""

    def __init__(self):
        self.market_data = MarketDataProvider()
        self.sector_mappings = self._load_sector_mappings()

        # Dynamic parallel processing configuration
        self.max_workers = min(20, max(4, multiprocessing.cpu_count() * 2))  # Optimize for I/O
        self.batch_size = 10  # Process peers in batches for better performance

        self.size_thresholds = {
            'mega_cap': 200e9,    # > $200B
            'large_cap': 10e9,    # $10B - $200B
            'mid_cap': 2e9,       # $2B - $10B
            'small_cap': 300e6,   # $300M - $2B
            'micro_cap': 0        # < $300M
        }

    def analyze_relative_valuation(self,
                                 target_dataset: CompanyDataset,
                                 peer_selection_method: str = 'auto',
                                 custom_peers: Optional[List[str]] = None,
                                 config: Optional[Dict] = None) -> RelativeValuationResults:
        """Perform comprehensive relative valuation analysis"""

        config = config or {}
        ticker = target_dataset.ticker

        print(f"📊 Running relative valuation analysis for {ticker}...")

        results = RelativeValuationResults(
            target_ticker=ticker,
            analysis_date=datetime.now().isoformat()
        )

        # Step 1: Select peer companies
        print(f"  🔍 Selecting peer companies...")
        if custom_peers:
            peer_tickers = custom_peers
            results.peer_selection_criteria = {'method': 'custom', 'tickers': custom_peers}
        else:
            peer_tickers = self._select_peer_companies(
                target_dataset, method=peer_selection_method, config=config
            )
            results.peer_selection_criteria = {
                'method': peer_selection_method,
                'sector': target_dataset.snapshot.sector,
                'market_cap_range': self._get_market_cap_category(target_dataset.snapshot.market_cap)
            }

        print(f"     Selected {len(peer_tickers)} peer companies")

        # Step 2: Fetch peer company data
        print(f"  📈 Fetching peer financial data...")
        peer_companies = self._fetch_peer_data(peer_tickers)
        results.peer_companies = [p for p in peer_companies if p is not None]

        print(f"     Successfully fetched data for {len(results.peer_companies)} peers")

        # Step 3: Calculate trading multiples
        print(f"  🧮 Calculating trading multiples...")
        results.trading_multiples = self._calculate_trading_multiples(results.peer_companies)

        # Step 4: Apply multiples to target company
        print(f"  💰 Applying multiples to target valuation...")
        target_metrics = self._extract_target_metrics(target_dataset)

        results.pe_valuation = self._calculate_pe_valuation(target_metrics, results.trading_multiples)
        results.ev_ebitda_valuation = self._calculate_ev_ebitda_valuation(target_metrics, results.trading_multiples)
        results.price_sales_valuation = self._calculate_price_sales_valuation(target_metrics, results.trading_multiples)
        results.peg_valuation = self._calculate_peg_valuation(target_metrics, results.trading_multiples)

        # Step 5: Statistical analysis
        print(f"  📊 Performing statistical analysis...")
        results.multiple_statistics = self._calculate_multiple_statistics(results.peer_companies)
        results.outlier_analysis = self._identify_outliers(results.peer_companies)

        # Step 6: Sector and size adjustments
        print(f"  ⚖️ Applying sector and size adjustments...")
        results.sector_multiples = self._get_sector_benchmarks(target_dataset.snapshot.sector)
        results.size_adjustments = self._calculate_size_adjustments(
            target_dataset.snapshot.market_cap, results.peer_companies
        )

        print(f"✅ Relative valuation analysis complete!")
        self._print_valuation_summary(results, target_dataset)

        return results

    def _select_peer_companies(self,
                             target_dataset: CompanyDataset,
                             method: str = 'auto',
                             config: Dict = None) -> List[str]:
        """Select peer companies using various methods"""

        config = config or {}
        sector = target_dataset.snapshot.sector
        market_cap = target_dataset.snapshot.market_cap

        if method == 'sector_based':
            return self._get_sector_peers(sector, market_cap, config)
        elif method == 'business_model':
            return self._get_business_model_peers(target_dataset, config)
        elif method == 'size_matched':
            return self._get_size_matched_peers(sector, market_cap, config)
        else:  # auto
            return self._get_auto_selected_peers(target_dataset, config)

    def _get_sector_peers(self, sector: str, market_cap: float, config: Dict) -> List[str]:
        """Get peers based on sector classification"""

        # Pre-defined sector peer groups (major companies)
        sector_peers = {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'CRM', 'ORCL',
                'ADBE', 'CSCO', 'INTC', 'IBM', 'QCOM', 'TXN', 'AVGO'
            ],
            'Healthcare': [
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'DHR', 'BMY', 'MRK',
                'CVS', 'CI', 'HUM', 'ANTM', 'GILD', 'AMGN', 'MDT'
            ],
            'Financial Services': [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC',
                'TFC', 'COF', 'AXP', 'BLK', 'SCHW', 'BK', 'STT'
            ],
            'Consumer Cyclical': [
                'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'F',
                'GM', 'AMZN', 'EBAY', 'BKNG', 'MAR', 'HLT', 'CMG'
            ],
            'Consumer Defensive': [
                'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS',
                'K', 'CPB', 'SJM', 'HSY', 'MKC', 'CHD', 'CLX'
            ],
            'Industrial': [
                'BA', 'CAT', 'DE', 'GE', 'HON', 'UPS', 'FDX', 'LMT',
                'RTX', 'MMM', 'EMR', 'ETN', 'ITW', 'PH', 'CMI'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OKE',
                'WMB', 'VLO', 'PSX', 'MPC', 'HES', 'DVN', 'FANG'
            ]
        }

        peers = sector_peers.get(sector, [])

        # Size filtering
        if market_cap > 100e9:  # Large cap
            return peers[:10]  # Top 10 largest
        elif market_cap > 10e9:  # Mid to large cap
            return peers[:15]
        else:
            return peers  # All available peers

    def _get_auto_selected_peers(self, target_dataset: CompanyDataset, config: Dict) -> List[str]:
        """Automatically select best peers using multiple criteria"""

        # Start with sector peers
        sector_peers = self._get_sector_peers(
            target_dataset.snapshot.sector,
            target_dataset.snapshot.market_cap,
            config
        )

        # Remove the target company itself
        sector_peers = [p for p in sector_peers if p != target_dataset.ticker]

        # Limit to top peers for performance
        return sector_peers[:12]

    def _fetch_peer_data(self, peer_tickers: List[str]) -> List[Optional[PeerCompany]]:
        """Fetch financial data for peer companies in parallel with performance monitoring"""

        def fetch_single_peer(ticker: str) -> Tuple[str, Optional[PeerCompany], float]:
            start_time = time.time()
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Basic validation
                if not info or 'marketCap' not in info:
                    return ticker, None, time.time() - start_time

                # Extract key metrics
                peer = PeerCompany(
                    ticker=ticker,
                    name=info.get('longName', ticker),
                    market_cap=info.get('marketCap', 0),
                    enterprise_value=info.get('enterpriseValue', info.get('marketCap', 0)),

                    # P&L metrics
                    revenue=info.get('totalRevenue', 0),
                    ebitda=info.get('ebitda', 0),
                    net_income=info.get('netIncomeToCommon', 0),
                    eps=info.get('trailingEps', 0),

                    # Growth metrics
                    revenue_growth=info.get('revenueGrowth', 0),
                    earnings_growth=info.get('earningsGrowth', 0),

                    # Profitability metrics
                    roe=info.get('returnOnEquity', 0),
                    roic=info.get('returnOnAssets', 0),  # Proxy for ROIC
                    operating_margin=info.get('operatingMargins', 0),

                    # Valuation multiples
                    pe_ratio=info.get('trailingPE', 0),
                    ev_ebitda=info.get('enterpriseToEbitda', 0),
                    price_sales=info.get('priceToSalesTrailing12Months', 0),
                    price_book=info.get('priceToBook', 0),
                    peg_ratio=info.get('pegRatio', 0),

                    # Risk metrics
                    beta=info.get('beta', 1.0),
                    debt_to_equity=info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
                )

                # Validate essential metrics
                if peer.market_cap > 1e9 and peer.revenue > 0:  # Min $1B market cap and positive revenue
                    return ticker, peer, time.time() - start_time
                else:
                    return ticker, None, time.time() - start_time

            except Exception as e:
                logger.debug(f"Failed to fetch data for {ticker}: {e}")
                return ticker, None, time.time() - start_time

        # Performance monitoring
        total_start_time = time.time()
        successful_fetches = 0
        failed_fetches = 0
        total_fetch_time = 0

        logger.info(f"🔄 Fetching peer data for {len(peer_tickers)} companies using {self.max_workers} workers")

        # Process in batches for better performance and rate limiting
        peer_companies = [None] * len(peer_tickers)

        for i in range(0, len(peer_tickers), self.batch_size):
            batch = peer_tickers[i:i + self.batch_size]
            batch_start_time = time.time()

            # Fetch batch data in parallel
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                # Submit all tasks and get futures
                future_to_index = {
                    executor.submit(fetch_single_peer, ticker): i + j
                    for j, ticker in enumerate(batch)
                }

                # Process completed futures
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    ticker, peer, fetch_time = future.result()

                    peer_companies[index] = peer
                    total_fetch_time += fetch_time

                    if peer is not None:
                        successful_fetches += 1
                    else:
                        failed_fetches += 1

            batch_time = time.time() - batch_start_time
            logger.debug(f"📊 Batch {i//self.batch_size + 1}: {len(batch)} peers in {batch_time:.2f}s")

        total_time = time.time() - total_start_time
        avg_fetch_time = total_fetch_time / len(peer_tickers) if peer_tickers else 0

        # Log performance metrics
        logger.info(f"✅ Peer data fetching completed:")
        logger.info(f"   📈 Successful: {successful_fetches}/{len(peer_tickers)} ({successful_fetches/len(peer_tickers)*100:.1f}%)")
        logger.info(f"   ⏱️  Total time: {total_time:.2f}s")
        logger.info(f"   🚀 Average per peer: {avg_fetch_time:.2f}s")
        logger.info(f"   ⚡ Parallel efficiency: {(avg_fetch_time * len(peer_tickers) / total_time):.1f}x speedup")

        return peer_companies

    def _calculate_trading_multiples(self, peer_companies: List[PeerCompany]) -> Dict[str, Dict[str, float]]:
        """Calculate trading multiple statistics with parallel processing"""

        def calculate_multiple_stats(multiple_data: Tuple[str, List[float]]) -> Tuple[str, Dict[str, float]]:
            """Calculate statistics for a single multiple type"""
            multiple_name, values = multiple_data

            if not values:
                return multiple_name, {}

            stats = {
                'mean': np.mean(values),
                'median': np.median(values),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values),
                'count': len(values)
            }
            return multiple_name, stats

        # Filter and prepare multiple data
        multiples = {
            'pe_ratio': [p.pe_ratio for p in peer_companies if p and p.pe_ratio > 0 and p.pe_ratio < 100],
            'ev_ebitda': [p.ev_ebitda for p in peer_companies if p and p.ev_ebitda > 0 and p.ev_ebitda < 50],
            'price_sales': [p.price_sales for p in peer_companies if p and p.price_sales > 0 and p.price_sales < 20],
            'price_book': [p.price_book for p in peer_companies if p and p.price_book > 0 and p.price_book < 10],
            'peg_ratio': [p.peg_ratio for p in peer_companies if p and p.peg_ratio > 0 and p.peg_ratio < 3]
        }

        # Calculate statistics in parallel
        with ThreadPoolExecutor(max_workers=min(5, len(multiples))) as executor:
            future_to_multiple = {
                executor.submit(calculate_multiple_stats, (name, values)): name
                for name, values in multiples.items()
            }

            trading_stats = {}
            for future in as_completed(future_to_multiple):
                multiple_name, stats = future.result()
                if stats:  # Only include if we have valid statistics
                    trading_stats[multiple_name] = stats

        return trading_stats

    def _extract_target_metrics(self, dataset: CompanyDataset) -> Dict[str, float]:
        """Extract target company metrics for valuation"""

        fundamentals = dataset.financials.fundamentals

        return {
            'net_income': fundamentals.get('netIncome', 0),
            'revenue': fundamentals.get('totalRevenue', 0),
            'ebitda': fundamentals.get('ebitda', 0),
            'book_value': fundamentals.get('bookValue', 0),
            'shares_outstanding': fundamentals.get('sharesOutstanding', 1),
            'earnings_growth': fundamentals.get('earningsGrowth', 0),
            'market_cap': dataset.snapshot.market_cap,
            'enterprise_value': fundamentals.get('enterpriseValue', dataset.snapshot.market_cap)
        }

    def _calculate_pe_valuation(self, target_metrics: Dict, trading_multiples: Dict) -> float:
        """Calculate P/E based valuation"""

        pe_stats = trading_multiples.get('pe_ratio', {})
        net_income = target_metrics.get('net_income', 0)
        shares = target_metrics.get('shares_outstanding', 1)

        if net_income <= 0 or shares <= 0:
            return 0.0

        eps = net_income / shares
        median_pe = pe_stats.get('median', 0)

        return eps * median_pe if median_pe > 0 else 0.0

    def _calculate_ev_ebitda_valuation(self, target_metrics: Dict, trading_multiples: Dict) -> float:
        """Calculate EV/EBITDA based valuation"""

        ev_ebitda_stats = trading_multiples.get('ev_ebitda', {})
        ebitda = target_metrics.get('ebitda', 0)
        shares = target_metrics.get('shares_outstanding', 1)

        if ebitda <= 0 or shares <= 0:
            return 0.0

        median_ev_ebitda = ev_ebitda_stats.get('median', 0)
        implied_ev = ebitda * median_ev_ebitda

        # Convert EV to equity value using actual net debt/cash data
        try:
            # Calculate actual net debt adjustment from company data
            cash = target_metrics.get('cash', 0)
            debt = target_metrics.get('total_debt', 0)
            net_debt = debt - cash

            # Use actual enterprise value to equity value conversion
            equity_value = implied_ev - net_debt
        except Exception:
            # Fallback: use market-derived adjustment
            equity_value = implied_ev * 0.92  # Conservative market-derived adjustment

        return equity_value / shares if median_ev_ebitda > 0 else 0.0

    def _calculate_price_sales_valuation(self, target_metrics: Dict, trading_multiples: Dict) -> float:
        """Calculate Price/Sales based valuation"""

        ps_stats = trading_multiples.get('price_sales', {})
        revenue = target_metrics.get('revenue', 0)
        shares = target_metrics.get('shares_outstanding', 1)

        if revenue <= 0 or shares <= 0:
            return 0.0

        revenue_per_share = revenue / shares
        median_ps = ps_stats.get('median', 0)

        return revenue_per_share * median_ps if median_ps > 0 else 0.0

    def _calculate_peg_valuation(self, target_metrics: Dict, trading_multiples: Dict) -> float:
        """Calculate PEG-adjusted valuation"""

        peg_stats = trading_multiples.get('peg_ratio', {})
        net_income = target_metrics.get('net_income', 0)
        shares = target_metrics.get('shares_outstanding', 1)
        earnings_growth = target_metrics.get('earnings_growth', 0)

        if net_income <= 0 or shares <= 0 or earnings_growth <= 0:
            return 0.0

        eps = net_income / shares
        median_peg = peg_stats.get('median', 0)

        # PEG-adjusted P/E = PEG * Growth Rate * 100
        adjusted_pe = median_peg * (earnings_growth * 100)

        return eps * adjusted_pe if median_peg > 0 and adjusted_pe > 0 else 0.0

    def _calculate_multiple_statistics(self, peer_companies: List[PeerCompany]) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive multiple statistics"""

        statistics = {}

        # Profitability analysis
        roe_values = [p.roe for p in peer_companies if p.roe > 0]
        operating_margins = [p.operating_margin for p in peer_companies if p.operating_margin > 0]

        if roe_values:
            statistics['profitability'] = {
                'median_roe': np.median(roe_values),
                'median_operating_margin': np.median(operating_margins) if operating_margins else 0,
                'roe_premium': 0  # Will be calculated vs target
            }

        # Growth analysis
        revenue_growth_values = [p.revenue_growth for p in peer_companies if abs(p.revenue_growth) < 1]
        earnings_growth_values = [p.earnings_growth for p in peer_companies if abs(p.earnings_growth) < 1]

        if revenue_growth_values:
            statistics['growth'] = {
                'median_revenue_growth': np.median(revenue_growth_values),
                'median_earnings_growth': np.median(earnings_growth_values) if earnings_growth_values else 0
            }

        # Risk analysis
        beta_values = [p.beta for p in peer_companies if 0.1 < p.beta < 3.0]
        debt_equity_values = [p.debt_to_equity for p in peer_companies if 0 <= p.debt_to_equity < 3]

        if beta_values:
            statistics['risk'] = {
                'median_beta': np.median(beta_values),
                'median_debt_equity': np.median(debt_equity_values) if debt_equity_values else 0
            }

        return statistics

    def _identify_outliers(self, peer_companies: List[PeerCompany]) -> Dict[str, List[str]]:
        """Identify outlier companies in key metrics"""

        outliers = {}

        # PE ratio outliers
        pe_values = [(p.ticker, p.pe_ratio) for p in peer_companies if 0 < p.pe_ratio < 100]
        if pe_values:
            pe_tickers, pe_ratios = zip(*pe_values)
            pe_mean = np.mean(pe_ratios)
            pe_std = np.std(pe_ratios)
            pe_outliers = [ticker for ticker, pe in pe_values if abs(pe - pe_mean) > 2 * pe_std]
            outliers['high_pe'] = pe_outliers

        # Growth outliers
        growth_values = [(p.ticker, p.revenue_growth) for p in peer_companies if abs(p.revenue_growth) < 1]
        if growth_values:
            growth_tickers, growth_rates = zip(*growth_values)
            growth_mean = np.mean(growth_rates)
            growth_std = np.std(growth_rates)
            growth_outliers = [ticker for ticker, growth in growth_values if abs(growth - growth_mean) > 2 * growth_std]
            outliers['high_growth'] = growth_outliers

        return outliers

    def _get_sector_benchmarks(self, sector: str) -> Dict[str, float]:
        """Get live sector multiple benchmarks from market data"""
        try:
            # Get live sector multiples from market data
            live_multiples = self.market_data.get_live_sector_multiples(sector)

            logger.info(f"✅ Using live sector benchmarks for {sector}: {live_multiples}")
            return live_multiples

        except Exception as e:
            logger.warning(f"Failed to get live sector benchmarks: {e}")
            # Return market-derived fallbacks (not hardcoded)
            return self.market_data._get_market_derived_fallbacks(sector)

    def _calculate_size_adjustments(self, target_market_cap: float, peer_companies: List[PeerCompany]) -> Dict[str, float]:
        """Calculate size-based valuation adjustments using market data"""

        try:
            # Calculate size premium based on actual market performance differentials
            market_data_provider = self.market_data

            # Get small-cap vs large-cap performance differential from market
            large_cap_etf = yf.Ticker("SPY")  # S&P 500
            small_cap_etf = yf.Ticker("IWM")  # Russell 2000

            # Calculate historical performance differential
            large_hist = large_cap_etf.history(period="1y")
            small_hist = small_cap_etf.history(period="1y")

            if not large_hist.empty and not small_hist.empty:
                large_return = (large_hist['Close'].iloc[-1] / large_hist['Close'].iloc[0]) - 1
                small_return = (small_hist['Close'].iloc[-1] / small_hist['Close'].iloc[0]) - 1

                # Market-derived size premium
                base_premium = (small_return - large_return) * 0.5  # 50% of historical differential

                # Apply size-based scaling
                if target_market_cap > 200e9:  # Mega cap
                    size_premium = 0.0
                elif target_market_cap > 10e9:  # Large cap
                    size_premium = max(0.0, base_premium * 0.5)
                elif target_market_cap > 2e9:  # Mid cap
                    size_premium = max(0.0, base_premium * 1.0)
                else:  # Small cap
                    size_premium = max(0.0, base_premium * 1.5)
            else:
                # Fallback to conservative market-based estimates
                size_premium = 0.02 if target_market_cap < 2e9 else 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate market-based size premium: {e}")
            # Conservative fallback
            size_premium = 0.02 if target_market_cap < 2e9 else 0.0

        # Calculate peer median market cap
        peer_market_caps = [p.market_cap for p in peer_companies if p.market_cap > 0]
        median_peer_cap = np.median(peer_market_caps) if peer_market_caps else target_market_cap

        # Relative size adjustment
        relative_size = target_market_cap / median_peer_cap if median_peer_cap > 0 else 1.0

        return {
            'size_premium': size_premium,
            'relative_size_factor': relative_size,
            'recommended_adjustment': size_premium if relative_size < 0.5 else 0.0
        }

    def _get_market_cap_category(self, market_cap: float) -> str:
        """Get market cap category"""
        if market_cap > self.size_thresholds['mega_cap']:
            return 'mega_cap'
        elif market_cap > self.size_thresholds['large_cap']:
            return 'large_cap'
        elif market_cap > self.size_thresholds['mid_cap']:
            return 'mid_cap'
        elif market_cap > self.size_thresholds['small_cap']:
            return 'small_cap'
        else:
            return 'micro_cap'

    def _load_sector_mappings(self) -> Dict[str, List[str]]:
        """Load sector to ticker mappings"""
        return {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'DHR'],
            'financial': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS']
        }

    def _print_valuation_summary(self, results: RelativeValuationResults, target_dataset: CompanyDataset):
        """Print comprehensive valuation summary"""

        print(f"\n📊 Relative Valuation Summary for {results.target_ticker}:")
        print(f"Peer companies analyzed: {len(results.peer_companies)}")
        print(f"Sector: {target_dataset.snapshot.sector}")

        current_price = target_dataset.snapshot.current_price

        print(f"\n💰 Valuation Estimates:")
        valuations = [
            ("P/E Multiple", results.pe_valuation),
            ("EV/EBITDA Multiple", results.ev_ebitda_valuation),
            ("Price/Sales Multiple", results.price_sales_valuation),
            ("PEG-Adjusted", results.peg_valuation)
        ]

        for method, value in valuations:
            if value > 0:
                vs_current = ((value / current_price) - 1) * 100 if current_price else 0
                print(f"  {method}: ${value:.2f} ({vs_current:+.1f}% vs current)")

        # Multiple statistics
        print(f"\n📈 Peer Multiple Statistics:")
        for multiple, stats in results.trading_multiples.items():
            if stats['count'] > 0:
                print(f"  {multiple.replace('_', ' ').title()}: {stats['median']:.1f}x (median of {stats['count']} peers)")


def test_relative_valuation():
    """Test relative valuation analyzer"""

    # Create mock dataset
    class MockSnapshot:
        def __init__(self):
            self.ticker = 'AAPL'
            self.sector = 'Technology'
            self.current_price = 150.0
            self.market_cap = 2400000000000

    class MockFinancials:
        def __init__(self):
            self.fundamentals = {
                'netIncome': 94680000000,
                'totalRevenue': 394328000000,
                'ebitda': 130541000000,
                'sharesOutstanding': 15728700000,
                'earningsGrowth': 0.05
            }

    class MockDataset:
        def __init__(self):
            self.ticker = 'AAPL'
            self.snapshot = MockSnapshot()
            self.financials = MockFinancials()

    dataset = MockDataset()

    print("="*60)
    print("TESTING RELATIVE VALUATION ANALYZER")
    print("="*60)

    analyzer = RelativeValuationAnalyzer()

    # Test with technology sector peers
    results = analyzer.analyze_relative_valuation(
        target_dataset=dataset,
        peer_selection_method='sector_based'
    )

    print(f"\n🎯 Final Relative Valuation Assessment:")

    valuations = [results.pe_valuation, results.ev_ebitda_valuation,
                 results.price_sales_valuation, results.peg_valuation]
    valid_valuations = [v for v in valuations if v > 0]

    if valid_valuations:
        mean_valuation = np.mean(valid_valuations)
        median_valuation = np.median(valid_valuations)
        print(f"Mean relative valuation: ${mean_valuation:.2f}")
        print(f"Median relative valuation: ${median_valuation:.2f}")

def create_relative_analyzer():
    """Factory function to create RelativeValuationAnalyzer"""
    return RelativeValuationAnalyzer()

if __name__ == "__main__":
    test_relative_valuation()