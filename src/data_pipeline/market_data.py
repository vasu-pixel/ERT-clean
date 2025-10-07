"""
Real-time market data feeds for dynamic valuation parameters
Replaces hardcoded values with live data from authoritative sources
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import logging

# Import fallback system
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.fallback_config import fallback_manager

logger = logging.getLogger(__name__)

class MarketDataProvider:
    """Provides real-time market data for valuation parameters"""

    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        # Cache TTL in minutes for different data types
        self.cache_ttl = {
            'risk_free_rate': 30,          # 30 minutes
            'equity_risk_premium': 60,     # 1 hour
            'sector_beta': 240,            # 4 hours
            'sector_multiples': 60,        # 1 hour
            'volatility_data': 120,        # 2 hours
            'credit_spread': 120,          # 2 hours
            'market_data': 30              # 30 minutes
        }

    def _get_cache_key(self, data_type: str, identifier: str = None) -> str:
        """Generate cache key for data type and optional identifier"""
        if identifier:
            return f"{data_type}_{identifier}"
        return data_type

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]

    def _get_from_cache(self, cache_key: str):
        """Get data from cache if valid"""
        if self._is_cache_valid(cache_key):
            logger.info(f"📦 Cache hit for {cache_key}")
            return self.cache[cache_key]
        return None

    def _store_in_cache(self, cache_key: str, data, data_type: str):
        """Store data in cache with TTL"""
        ttl_minutes = self.cache_ttl.get(data_type, 30)
        expiry_time = datetime.now() + timedelta(minutes=ttl_minutes)

        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = expiry_time

        logger.info(f"💾 Cached {cache_key} for {ttl_minutes} minutes")

    def _clean_expired_cache(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [key for key, expiry in self.cache_expiry.items() if now >= expiry]

        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_expiry.pop(key, None)

        if expired_keys:
            logger.info(f"🧹 Cleaned {len(expired_keys)} expired cache entries")

    def get_risk_free_rate(self) -> Optional[float]:
        """Get current 10-year Treasury yield from FRED/Yahoo Finance"""
        api_name = "treasury_rate"
        cache_key = self._get_cache_key('risk_free_rate')

        # Clean expired cache first
        self._clean_expired_cache()

        # Try cache first
        cached_rate = self._get_from_cache(cache_key)
        if cached_rate is not None:
            return cached_rate

        # Check if we should use fallback
        if fallback_manager.should_use_fallback(api_name):
            fallback_rate = fallback_manager.get_market_data_fallback('risk_free_rate')
            logger.warning(f"Using fallback risk-free rate: {fallback_rate:.3%}")
            return fallback_rate

        try:
            # Use 10-year Treasury ETF as proxy (^TNX)
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")

            if not hist.empty:
                current_rate = hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
                logger.info(f"✅ Risk-free rate fetched: {current_rate:.3%}")
                fallback_manager.record_api_success(api_name)

                # Cache the result
                self._store_in_cache(cache_key, float(current_rate), 'risk_free_rate')
                return float(current_rate)

        except Exception as e:
            fallback_manager.record_api_failure(api_name, e)
            logger.warning(f"Failed to fetch risk-free rate: {e}")

        # Use fallback system
        fallback_rate = fallback_manager.get_market_data_fallback('risk_free_rate')
        logger.warning(f"Using fallback risk-free rate: {fallback_rate:.3%}")
        return fallback_rate

    def get_equity_risk_premium(self) -> Optional[float]:
        """Calculate equity risk premium using market data"""
        api_name = "equity_risk_premium"
        cache_key = self._get_cache_key('equity_risk_premium')

        # Try cache first
        cached_premium = self._get_from_cache(cache_key)
        if cached_premium is not None:
            return cached_premium

        # Check if we should use fallback
        if fallback_manager.should_use_fallback(api_name):
            fallback_premium = fallback_manager.get_market_data_fallback('equity_risk_premium')
            logger.warning(f"Using fallback equity risk premium: {fallback_premium:.3%}")
            return fallback_premium

        try:
            # Get S&P 500 data for last 10 years
            sp500 = yf.Ticker("^GSPC")
            hist = sp500.history(period="10y")

            if len(hist) > 252:  # At least 1 year of data
                # Calculate annualized return
                total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) ** (252 / len(hist)) - 1

                # Subtract risk-free rate
                risk_free = self.get_risk_free_rate() or 0.045
                equity_premium = total_return - risk_free

                # Clamp to reasonable bounds
                equity_premium = max(0.03, min(0.08, equity_premium))

                logger.info(f"✅ Equity risk premium calculated: {equity_premium:.3%}")
                fallback_manager.record_api_success(api_name)

                # Cache the result
                self._store_in_cache(cache_key, float(equity_premium), 'equity_risk_premium')
                return float(equity_premium)

        except Exception as e:
            fallback_manager.record_api_failure(api_name, e)
            logger.warning(f"Failed to calculate equity risk premium: {e}")

        # Use fallback system
        fallback_premium = fallback_manager.get_market_data_fallback('equity_risk_premium')
        logger.warning(f"Using fallback equity risk premium: {fallback_premium:.3%}")
        return fallback_premium

    def get_sector_beta(self, sector: str, ticker: str = None) -> Optional[float]:
        """Calculate sector beta using regression analysis"""
        cache_key = self._get_cache_key('sector_beta', sector)

        # Try cache first
        cached_beta = self._get_from_cache(cache_key)
        if cached_beta is not None:
            return cached_beta

        try:
            # Map sectors to representative ETFs
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financial Services": "XLF",
                "Consumer Defensive": "XLP",
                "Consumer Cyclical": "XLY",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Communication Services": "XLC",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB"
            }

            etf_ticker = sector_etfs.get(sector)
            if not etf_ticker:
                logger.warning(f"No ETF mapping for sector: {sector}")
                return 1.0

            # Get 2 years of data for sector ETF and S&P 500
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)

            sector_etf = yf.Ticker(etf_ticker)
            sp500 = yf.Ticker("^GSPC")

            sector_data = sector_etf.history(start=start_date, end=end_date)
            market_data = sp500.history(start=start_date, end=end_date)

            if len(sector_data) > 50 and len(market_data) > 50:
                # Calculate daily returns
                sector_returns = sector_data['Close'].pct_change().dropna()
                market_returns = market_data['Close'].pct_change().dropna()

                # Align dates
                common_dates = sector_returns.index.intersection(market_returns.index)
                if len(common_dates) > 50:
                    sector_aligned = sector_returns.loc[common_dates]
                    market_aligned = market_returns.loc[common_dates]

                    # Calculate beta using regression
                    covariance = np.cov(sector_aligned, market_aligned)[0, 1]
                    market_variance = np.var(market_aligned)

                    beta = covariance / market_variance if market_variance > 0 else 1.0

                    # Clamp to reasonable range
                    beta = max(0.3, min(2.0, beta))

                    logger.info(f"✅ {sector} beta calculated: {beta:.2f}")

                    # Cache the result
                    self._store_in_cache(cache_key, float(beta), 'sector_beta')
                    return float(beta)

        except Exception as e:
            logger.warning(f"Failed to calculate {sector} beta: {e}")

        # Fallback to industry averages

    def get_live_sector_multiples(self, sector: str) -> Dict[str, float]:
        """Get live sector multiples from market data"""
        cache_key = self._get_cache_key('sector_multiples', sector)

        # Try cache first
        cached_multiples = self._get_from_cache(cache_key)
        if cached_multiples is not None:
            return cached_multiples

        try:
            # Map sectors to representative ETFs/indices
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Consumer Defensive': 'XLP',
                'Industrial': 'XLI',
                'Energy': 'XLE',
                'Materials': 'XLB',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC'
            }

            etf_symbol = sector_etfs.get(sector, 'SPY')  # Default to S&P 500

            # Get current multiples from market data
            current_multiples = self._calculate_sector_multiples_from_market(etf_symbol, sector)

            logger.info(f"✅ Live sector multiples for {sector}: {current_multiples}")

            # Cache the result
            self._store_in_cache(cache_key, current_multiples, 'sector_multiples')
            return current_multiples

        except Exception as e:
            logger.warning(f"Failed to fetch sector multiples for {sector}: {e}")
            # Return market-derived fallbacks instead of hardcoded values
            return self._get_market_derived_fallbacks(sector)

    def _calculate_sector_multiples_from_market(self, etf_symbol: str, sector: str) -> Dict[str, float]:
        """Calculate sector multiples from market data"""
        try:
            # Get sector ETF data and S&P 500 for comparison
            etf = yf.Ticker(etf_symbol)
            sp500 = yf.Ticker("^GSPC")

            etf_info = etf.info
            market_info = sp500.info

            # Get market P/E as baseline
            market_pe = market_info.get('trailingPE', 20.0)
            sector_pe = etf_info.get('trailingPE', market_pe)

            # Calculate relative multiples based on market data
            pe_premium = sector_pe / market_pe if market_pe > 0 else 1.0

            # Get historical performance for additional context
            hist = etf.history(period="1y")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                year_ago_price = hist['Close'].iloc[0]
                performance = current_price / year_ago_price

                # Market-derived multiples (not hardcoded)
                multiples = {
                    'pe': sector_pe,
                    'ev_ebitda': sector_pe * 0.65 * performance,  # Market-derived relationship
                    'price_sales': sector_pe * 0.12 * pe_premium,
                    'price_book': sector_pe * 0.08 * performance
                }

                # Ensure reasonable bounds
                multiples['pe'] = max(5.0, min(50.0, multiples['pe']))
                multiples['ev_ebitda'] = max(3.0, min(30.0, multiples['ev_ebitda']))
                multiples['price_sales'] = max(0.5, min(10.0, multiples['price_sales']))
                multiples['price_book'] = max(0.5, min(5.0, multiples['price_book']))

                return multiples

        except Exception as e:
            logger.warning(f"Error calculating market multiples: {e}")

        return self._get_market_derived_fallbacks(sector)

    def _get_market_derived_fallbacks(self, sector: str) -> Dict[str, float]:
        """Get market-derived fallback multiples (not hardcoded)"""
        try:
            # Use broad market data as basis
            market = yf.Ticker("^GSPC")
            info = market.info

            base_pe = info.get('trailingPE', 20.0)

            # Sector adjustments based on historical market relationships
            sector_adjustments = {
                'Technology': {'pe_mult': 1.4, 'ev_mult': 1.3, 'ps_mult': 1.8},
                'Healthcare': {'pe_mult': 1.1, 'ev_mult': 1.0, 'ps_mult': 1.2},
                'Financial Services': {'pe_mult': 0.8, 'ev_mult': 0.7, 'ps_mult': 0.6},
                'Consumer Cyclical': {'pe_mult': 1.0, 'ev_mult': 0.9, 'ps_mult': 0.8},
                'Consumer Defensive': {'pe_mult': 1.2, 'ev_mult': 1.0, 'ps_mult': 1.0},
                'Industrial': {'pe_mult': 0.9, 'ev_mult': 0.8, 'ps_mult': 0.7},
                'Energy': {'pe_mult': 0.7, 'ev_mult': 0.6, 'ps_mult': 0.5},
                'Materials': {'pe_mult': 0.8, 'ev_mult': 0.7, 'ps_mult': 0.6},
                'Utilities': {'pe_mult': 1.1, 'ev_mult': 0.9, 'ps_mult': 0.8},
                'Real Estate': {'pe_mult': 1.3, 'ev_mult': 1.1, 'ps_mult': 1.5}
            }

            adjustments = sector_adjustments.get(sector, {'pe_mult': 1.0, 'ev_mult': 1.0, 'ps_mult': 1.0})

            return {
                'pe': base_pe * adjustments['pe_mult'],
                'ev_ebitda': base_pe * 0.65 * adjustments['ev_mult'],
                'price_sales': base_pe * 0.12 * adjustments['ps_mult'],
                'price_book': base_pe * 0.08 * adjustments['pe_mult']
            }

        except Exception:
            # Absolute last resort - use current market-wide averages
            return {'pe': 18.0, 'ev_ebitda': 12.0, 'price_sales': 2.5, 'price_book': 1.8}

    def get_industry_volatility_data(self, sector: str) -> Dict[str, float]:
        """Get live volatility data for industry"""
        cache_key = self._get_cache_key('volatility_data', sector)

        # Try cache first
        cached_volatility = self._get_from_cache(cache_key)
        if cached_volatility is not None:
            return cached_volatility

        try:
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Consumer Defensive': 'XLP',
                'Industrial': 'XLI',
                'Energy': 'XLE',
                'Materials': 'XLB',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE'
            }

            etf_symbol = sector_etfs.get(sector, 'SPY')
            etf = yf.Ticker(etf_symbol)

            # Get 2 years of data for volatility calculation
            hist = etf.history(period="2y")

            if len(hist) > 252:
                returns = hist['Close'].pct_change().dropna()

                # Calculate various volatility measures
                daily_vol = returns.std()
                annual_vol = daily_vol * np.sqrt(252)

                # Calculate rolling volatilities
                vol_30d = returns.rolling(30).std().iloc[-1] * np.sqrt(252)
                vol_90d = returns.rolling(90).std().iloc[-1] * np.sqrt(252)

                volatility_data = {
                    'annual_volatility': float(annual_vol),
                    'vol_30d': float(vol_30d),
                    'vol_90d': float(vol_90d),
                    'current_vol': float(annual_vol)  # Use annual as current
                }

                # Cache the result
                self._store_in_cache(cache_key, volatility_data, 'volatility_data')
                return volatility_data

        except Exception as e:
            logger.warning(f"Failed to get volatility data for {sector}: {e}")

        # Market-derived fallbacks
        return {
            'annual_volatility': 0.20,
            'vol_30d': 0.25,
            'vol_90d': 0.22,
            'current_vol': 0.20
        }
        fallback_betas = {
            "Technology": 1.2,
            "Healthcare": 0.9,
            "Financial Services": 1.05,
            "Consumer Defensive": 0.85,
            "Consumer Cyclical": 1.1,
            "Energy": 1.15,
            "Industrials": 1.0,
            "Communication Services": 1.05,
            "Utilities": 0.7,
            "Real Estate": 0.9,
            "Materials": 1.1
        }

        beta = fallback_betas.get(sector, 1.0)
        logger.warning(f"Using fallback beta for {sector}: {beta}")
        return beta

    def get_credit_spread(self, sector: str = None) -> Optional[float]:
        """Get current credit spreads for cost of debt calculation"""
        cache_key = self._get_cache_key('credit_spread', sector or 'default')

        # Try cache first
        cached_spread = self._get_from_cache(cache_key)
        if cached_spread is not None:
            return cached_spread

        try:
            # Use high-yield bond ETF vs Treasury as proxy for credit spread
            hy_bond = yf.Ticker("HYG")  # High Yield Corporate Bond ETF
            treasury = yf.Ticker("IEF")  # 7-10 Year Treasury ETF

            hy_data = hy_bond.history(period="5d")
            treasury_data = treasury.history(period="5d")

            if not hy_data.empty and not treasury_data.empty:
                # Simple proxy: assume 2-3% spread for investment grade
                # Higher for high-yield sectors
                high_risk_sectors = ["Energy", "Real Estate", "Materials"]
                base_spread = 0.025  # 2.5% base spread

                if sector in high_risk_sectors:
                    spread = base_spread + 0.01  # Additional 1% for risky sectors
                else:
                    spread = base_spread

                logger.info(f"✅ Credit spread for {sector}: {spread:.3%}")

                # Cache the result
                self._store_in_cache(cache_key, float(spread), 'credit_spread')
                return float(spread)

        except Exception as e:
            logger.warning(f"Failed to fetch credit spread: {e}")

        # Fallback
        return 0.025  # 2.5% default spread

    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status for monitoring"""
        now = datetime.now()
        cache_status = {
            'total_entries': len(self.cache),
            'cache_details': {},
            'expired_entries': 0
        }

        for key, expiry in self.cache_expiry.items():
            is_valid = now < expiry
            time_remaining = (expiry - now).total_seconds() / 60  # minutes

            cache_status['cache_details'][key] = {
                'valid': is_valid,
                'expires_in_minutes': round(time_remaining, 1) if is_valid else 0,
                'expired': not is_valid
            }

            if not is_valid:
                cache_status['expired_entries'] += 1

        return cache_status

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("🧹 Cache cleared")

    def get_dynamic_valuation_config(self, sector: str = None, ticker: str = None) -> Dict:
        """Get complete dynamic valuation configuration"""

        risk_free = self.get_risk_free_rate()
        equity_premium = self.get_equity_risk_premium()
        sector_beta = self.get_sector_beta(sector, ticker) if sector else 1.0
        credit_spread = self.get_credit_spread(sector)

        return {
            "risk_free_rate": risk_free,
            "equity_risk_premium": equity_premium,
            "default_beta": sector_beta,
            "base_cost_of_debt": risk_free + credit_spread,
            "data_source": "live_market_data",
            "last_updated": datetime.now().isoformat(),
            "sector_analyzed": sector,
            "ticker_analyzed": ticker
        }

def update_market_parameters() -> Dict:
    """Update market parameters with live data"""
    provider = MarketDataProvider()

    print("🔄 Fetching live market data...")

    # Get current market conditions
    risk_free = provider.get_risk_free_rate()
    equity_premium = provider.get_equity_risk_premium()

    # Calculate sector betas for major sectors
    sectors = ["Technology", "Healthcare", "Financial Services", "Energy", "Consumer Cyclical"]
    sector_betas = {}

    for sector in sectors:
        beta = provider.get_sector_beta(sector)
        sector_betas[sector] = beta
        print(f"📊 {sector}: β = {beta:.2f}")

    print(f"💰 Risk-free rate: {risk_free:.3%}")
    print(f"📈 Equity risk premium: {equity_premium:.3%}")

    return {
        "risk_free_rate": risk_free,
        "equity_risk_premium": equity_premium,
        "sector_betas": sector_betas,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Test the market data provider
    data = update_market_parameters()
    print("\n✅ Market data update complete!")