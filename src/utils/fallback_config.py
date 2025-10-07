"""
Graceful Fallback Configuration System
Provides robust fallbacks when APIs or data sources are unavailable
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FallbackConfigManager:
    """Manages fallback configurations and graceful degradation"""

    def __init__(self):
        self.fallback_data = self._initialize_fallback_data()
        self.api_health_status = {}
        self.last_successful_calls = {}

    def _initialize_fallback_data(self) -> Dict[str, Any]:
        """Initialize comprehensive fallback data"""
        return {
            # Market Data Fallbacks
            'risk_free_rate': 0.045,  # 4.5% - Based on long-term Treasury average
            'equity_risk_premium': 0.055,  # 5.5% - Historical market premium
            'market_volatility': 0.20,  # 20% - Long-term market volatility

            # Sector-specific fallbacks (market-derived)
            'sector_betas': {
                'Technology': 1.25,
                'Healthcare': 0.90,
                'Financial Services': 1.15,
                'Consumer Defensive': 0.70,
                'Consumer Cyclical': 1.10,
                'Energy': 1.20,
                'Industrials': 1.05,
                'Communication Services': 1.00,
                'Utilities': 0.65,
                'Real Estate': 1.00,
                'Materials': 1.15,
                'Default': 1.00
            },

            # Valuation multiples fallbacks
            'sector_multiples': {
                'Technology': {'pe': 25.0, 'ev_ebitda': 18.0, 'ps': 6.0, 'pb': 4.5},
                'Healthcare': {'pe': 22.0, 'ev_ebitda': 15.0, 'ps': 4.5, 'pb': 3.0},
                'Financial Services': {'pe': 12.0, 'ev_ebitda': 8.0, 'ps': 2.5, 'pb': 1.2},
                'Consumer Defensive': {'pe': 18.0, 'ev_ebitda': 12.0, 'ps': 2.0, 'pb': 2.5},
                'Consumer Cyclical': {'pe': 20.0, 'ev_ebitda': 14.0, 'ps': 1.8, 'pb': 2.8},
                'Energy': {'pe': 15.0, 'ev_ebitda': 8.0, 'ps': 1.5, 'pb': 1.8},
                'Industrials': {'pe': 19.0, 'ev_ebitda': 13.0, 'ps': 2.2, 'pb': 2.5},
                'Utilities': {'pe': 16.0, 'ev_ebitda': 10.0, 'ps': 2.8, 'pb': 1.5},
                'Real Estate': {'pe': 20.0, 'ev_ebitda': 12.0, 'ps': 8.0, 'pb': 1.8},
                'Materials': {'pe': 17.0, 'ev_ebitda': 11.0, 'ps': 1.6, 'pb': 2.2},
                'Default': {'pe': 18.0, 'ev_ebitda': 12.0, 'ps': 3.0, 'pb': 2.5}
            },

            # Credit spread fallbacks
            'credit_spreads': {
                'AAA': 0.005,  # 0.5%
                'AA': 0.010,   # 1.0%
                'A': 0.015,    # 1.5%
                'BBB': 0.025,  # 2.5%
                'BB': 0.045,   # 4.5%
                'B': 0.065,    # 6.5%
                'CCC': 0.100,  # 10.0%
                'Default': 0.025
            },

            # Growth rate fallbacks
            'growth_rates': {
                'terminal_growth': 0.025,  # 2.5% long-term GDP growth
                'revenue_growth': {
                    'Technology': 0.12,
                    'Healthcare': 0.08,
                    'Consumer Defensive': 0.05,
                    'Utilities': 0.04,
                    'Default': 0.06
                }
            }
        }

    def record_api_failure(self, api_name: str, error: Exception):
        """Record API failure for fallback decision making"""
        self.api_health_status[api_name] = {
            'status': 'failed',
            'last_error': str(error),
            'error_time': datetime.now()
        }
        logger.warning(f"API failure recorded for {api_name}: {error}")

    def record_api_success(self, api_name: str):
        """Record API success"""
        self.api_health_status[api_name] = {
            'status': 'healthy',
            'last_success': datetime.now()
        }
        self.last_successful_calls[api_name] = datetime.now()

    def is_api_healthy(self, api_name: str, max_failure_age_minutes: int = 30) -> bool:
        """Check if API is considered healthy"""
        if api_name not in self.api_health_status:
            return True  # Assume healthy if no history

        status = self.api_health_status[api_name]

        if status['status'] == 'healthy':
            return True

        # Check if failure is recent
        if 'error_time' in status:
            time_since_failure = datetime.now() - status['error_time']
            if time_since_failure > timedelta(minutes=max_failure_age_minutes):
                return True  # Try again after cooldown period

        return False

    def get_fallback_value(self, category: str, key: str, sector: str = None) -> Any:
        """Get fallback value for a specific parameter"""
        try:
            if category in self.fallback_data:
                data = self.fallback_data[category]

                # Handle sector-specific lookups
                if isinstance(data, dict) and sector:
                    if sector in data:
                        return data[sector]
                    elif 'Default' in data:
                        return data['Default']

                # Handle direct key lookup
                if key in data:
                    return data[key]

                # Return the data itself if it's a simple value
                if not isinstance(data, dict):
                    return data

            logger.warning(f"No fallback found for {category}.{key} (sector: {sector})")
            return None

        except Exception as e:
            logger.error(f"Error getting fallback value: {e}")
            return None

    def get_market_data_fallback(self, data_type: str, sector: str = None) -> Any:
        """Get market data fallbacks"""
        fallback_map = {
            'risk_free_rate': self.fallback_data['risk_free_rate'],
            'equity_risk_premium': self.fallback_data['equity_risk_premium'],
            'volatility': self.fallback_data['market_volatility'],
            'beta': self.get_fallback_value('sector_betas', '', sector),
            'credit_spread': self.fallback_data['credit_spreads']['Default']
        }

        return fallback_map.get(data_type)

    def get_valuation_multiples_fallback(self, sector: str) -> Dict[str, float]:
        """Get valuation multiples fallback for a sector"""
        return self.get_fallback_value('sector_multiples', '', sector) or \
               self.fallback_data['sector_multiples']['Default']

    def should_use_fallback(self, api_name: str, force_fallback: bool = False) -> bool:
        """Determine if fallback should be used"""
        if force_fallback:
            return True

        return not self.is_api_healthy(api_name)

    def get_data_freshness_status(self) -> Dict[str, str]:
        """Get status of data freshness"""
        status = {}
        current_time = datetime.now()

        for api_name, last_success in self.last_successful_calls.items():
            age = current_time - last_success
            if age < timedelta(hours=1):
                status[api_name] = "Fresh"
            elif age < timedelta(hours=6):
                status[api_name] = "Stale"
            else:
                status[api_name] = "Very Stale"

        return status

    def create_degraded_mode_config(self) -> Dict[str, Any]:
        """Create configuration for degraded mode operation"""
        return {
            'use_cached_data_only': True,
            'skip_live_market_data': True,
            'use_fallback_multiples': True,
            'reduced_monte_carlo_runs': 1000,  # Reduce from 10,000
            'skip_real_options_valuation': True,
            'basic_peer_selection_only': True,
            'message': "Running in degraded mode due to API unavailability"
        }

# Global instance
fallback_manager = FallbackConfigManager()