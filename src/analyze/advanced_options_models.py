"""
Advanced Options Modeling Module
American-style options, path-dependent options, and multi-factor models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AdvancedOptionResults:
    """Results from advanced options modeling"""
    american_call_value: float
    american_put_value: float
    european_call_value: float
    european_put_value: float
    barrier_option_values: Dict[str, float]
    asian_option_values: Dict[str, float]
    lookback_option_values: Dict[str, float]
    compound_option_values: Dict[str, float]
    rainbow_option_values: Dict[str, float]
    early_exercise_premium: float
    path_dependency_analysis: Dict[str, Any]
    greeks_analysis: Dict[str, float]
    optimal_exercise_boundary: List[float]
    monte_carlo_confidence: Dict[str, float]
    methodology_notes: List[str]

class AdvancedOptionsModeler:
    """
    Advanced options pricing for real options valuation
    """

    def __init__(self):
        self.simulation_paths = 50000
        self.time_steps = 252  # Daily steps for 1 year

    def price_advanced_options(self, underlying_value: float,
                             strike_prices: List[float],
                             time_to_expiry: float,
                             volatility: float,
                             risk_free_rate: float,
                             dividend_yield: float = 0.0) -> AdvancedOptionResults:
        """
        Price advanced options using multiple models
        """
        try:
            print("🎯 Pricing advanced real options...")

            # Step 1: American Options
            american_call = self._price_american_call(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            american_put = self._price_american_put(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            # Step 2: European Options (benchmark)
            european_call = self._black_scholes_call(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            european_put = self._black_scholes_put(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            # Step 3: Barrier Options
            barrier_values = self._price_barrier_options(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            # Step 4: Asian Options
            asian_values = self._price_asian_options(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            # Step 5: Lookback Options
            lookback_values = self._price_lookback_options(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            # Step 6: Compound Options
            compound_values = self._price_compound_options(
                underlying_value, strike_prices, time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            # Step 7: Rainbow Options (multi-asset)
            rainbow_values = self._price_rainbow_options(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate
            )

            # Step 8: Calculate early exercise premium
            early_exercise_premium = max(0, american_call - european_call)

            # Step 9: Path dependency analysis
            path_analysis = self._analyze_path_dependency(
                underlying_value, volatility, time_to_expiry
            )

            # Step 10: Greeks calculation
            greeks = self._calculate_greeks(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate, dividend_yield
            )

            # Step 11: Optimal exercise boundary
            exercise_boundary = self._calculate_optimal_exercise_boundary(
                strike_prices[0], time_to_expiry, volatility, risk_free_rate
            )

            # Step 12: Monte Carlo confidence intervals
            mc_confidence = self._calculate_monte_carlo_confidence(
                underlying_value, strike_prices[0], time_to_expiry,
                volatility, risk_free_rate
            )

            # Step 13: Generate methodology notes
            methodology_notes = self._generate_methodology_notes()

            return AdvancedOptionResults(
                american_call_value=american_call,
                american_put_value=american_put,
                european_call_value=european_call,
                european_put_value=european_put,
                barrier_option_values=barrier_values,
                asian_option_values=asian_values,
                lookback_option_values=lookback_values,
                compound_option_values=compound_values,
                rainbow_option_values=rainbow_values,
                early_exercise_premium=early_exercise_premium,
                path_dependency_analysis=path_analysis,
                greeks_analysis=greeks,
                optimal_exercise_boundary=exercise_boundary,
                monte_carlo_confidence=mc_confidence,
                methodology_notes=methodology_notes
            )

        except Exception as e:
            logger.error(f"Error in advanced options modeling: {e}")
            return self._create_empty_results()

    def _price_american_call(self, S: float, K: float, T: float,
                           sigma: float, r: float, q: float) -> float:
        """Price American call option using binomial tree"""
        try:
            n = 100  # Number of time steps
            dt = T / n
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp((r - q) * dt) - d) / (u - d)

            # Initialize asset prices
            asset_prices = np.zeros((n + 1, n + 1))
            for i in range(n + 1):
                for j in range(i + 1):
                    asset_prices[j, i] = S * (u ** (i - j)) * (d ** j)

            # Initialize option values at expiration
            option_values = np.zeros((n + 1, n + 1))
            for j in range(n + 1):
                option_values[j, n] = max(0, asset_prices[j, n] - K)

            # Backward induction with early exercise check
            for i in range(n - 1, -1, -1):
                for j in range(i + 1):
                    # European option value
                    european_value = np.exp(-r * dt) * (
                        p * option_values[j, i + 1] +
                        (1 - p) * option_values[j + 1, i + 1]
                    )
                    # Early exercise value
                    early_exercise_value = max(0, asset_prices[j, i] - K)
                    # American option value
                    option_values[j, i] = max(european_value, early_exercise_value)

            return option_values[0, 0]

        except Exception as e:
            logger.warning(f"Error pricing American call: {e}")
            return 0.0

    def _price_american_put(self, S: float, K: float, T: float,
                          sigma: float, r: float, q: float) -> float:
        """Price American put option using binomial tree"""
        try:
            n = 100
            dt = T / n
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp((r - q) * dt) - d) / (u - d)

            # Initialize asset prices
            asset_prices = np.zeros((n + 1, n + 1))
            for i in range(n + 1):
                for j in range(i + 1):
                    asset_prices[j, i] = S * (u ** (i - j)) * (d ** j)

            # Initialize option values at expiration
            option_values = np.zeros((n + 1, n + 1))
            for j in range(n + 1):
                option_values[j, n] = max(0, K - asset_prices[j, n])

            # Backward induction with early exercise check
            for i in range(n - 1, -1, -1):
                for j in range(i + 1):
                    # European option value
                    european_value = np.exp(-r * dt) * (
                        p * option_values[j, i + 1] +
                        (1 - p) * option_values[j + 1, i + 1]
                    )
                    # Early exercise value
                    early_exercise_value = max(0, K - asset_prices[j, i])
                    # American option value
                    option_values[j, i] = max(european_value, early_exercise_value)

            return option_values[0, 0]

        except Exception as e:
            logger.warning(f"Error pricing American put: {e}")
            return 0.0

    def _black_scholes_call(self, S: float, K: float, T: float,
                          sigma: float, r: float, q: float) -> float:
        """Black-Scholes European call option price"""
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            call_price = (S * np.exp(-q * T) * norm.cdf(d1) -
                         K * np.exp(-r * T) * norm.cdf(d2))

            return max(0, call_price)

        except Exception:
            return 0.0

    def _black_scholes_put(self, S: float, K: float, T: float,
                         sigma: float, r: float, q: float) -> float:
        """Black-Scholes European put option price"""
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            put_price = (K * np.exp(-r * T) * norm.cdf(-d2) -
                        S * np.exp(-q * T) * norm.cdf(-d1))

            return max(0, put_price)

        except Exception:
            return 0.0

    def _price_barrier_options(self, S: float, K: float, T: float,
                             sigma: float, r: float, q: float) -> Dict[str, float]:
        """Price barrier options using Monte Carlo"""
        try:
            # Set barrier levels
            barrier_up = S * 1.3    # Up-and-out barrier
            barrier_down = S * 0.7  # Down-and-out barrier

            # Monte Carlo simulation
            np.random.seed(42)
            dt = T / self.time_steps
            paths = self.simulation_paths

            # Generate price paths
            Z = np.random.standard_normal((paths, self.time_steps))
            price_paths = np.zeros((paths, self.time_steps + 1))
            price_paths[:, 0] = S

            for t in range(1, self.time_steps + 1):
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
                )

            # Check barrier conditions
            max_prices = np.max(price_paths, axis=1)
            min_prices = np.min(price_paths, axis=1)
            final_prices = price_paths[:, -1]

            # Up-and-out call
            up_out_call_payoffs = np.where(
                max_prices <= barrier_up,
                np.maximum(final_prices - K, 0),
                0
            )

            # Down-and-out put
            down_out_put_payoffs = np.where(
                min_prices >= barrier_down,
                np.maximum(K - final_prices, 0),
                0
            )

            # Up-and-in call
            up_in_call_payoffs = np.where(
                max_prices > barrier_up,
                np.maximum(final_prices - K, 0),
                0
            )

            # Discount and average
            discount_factor = np.exp(-r * T)

            return {
                'up_and_out_call': np.mean(up_out_call_payoffs) * discount_factor,
                'down_and_out_put': np.mean(down_out_put_payoffs) * discount_factor,
                'up_and_in_call': np.mean(up_in_call_payoffs) * discount_factor,
                'barrier_up_level': barrier_up,
                'barrier_down_level': barrier_down
            }

        except Exception as e:
            logger.warning(f"Error pricing barrier options: {e}")
            return {}

    def _price_asian_options(self, S: float, K: float, T: float,
                           sigma: float, r: float, q: float) -> Dict[str, float]:
        """Price Asian (average price) options"""
        try:
            # Monte Carlo simulation
            np.random.seed(42)
            dt = T / self.time_steps
            paths = self.simulation_paths

            # Generate price paths
            Z = np.random.standard_normal((paths, self.time_steps))
            price_paths = np.zeros((paths, self.time_steps + 1))
            price_paths[:, 0] = S

            for t in range(1, self.time_steps + 1):
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
                )

            # Calculate averages
            arithmetic_averages = np.mean(price_paths, axis=1)
            geometric_averages = np.exp(np.mean(np.log(price_paths), axis=1))

            # Calculate payoffs
            asian_call_arithmetic = np.maximum(arithmetic_averages - K, 0)
            asian_put_arithmetic = np.maximum(K - arithmetic_averages, 0)
            asian_call_geometric = np.maximum(geometric_averages - K, 0)
            asian_put_geometric = np.maximum(K - geometric_averages, 0)

            # Discount and average
            discount_factor = np.exp(-r * T)

            return {
                'arithmetic_average_call': np.mean(asian_call_arithmetic) * discount_factor,
                'arithmetic_average_put': np.mean(asian_put_arithmetic) * discount_factor,
                'geometric_average_call': np.mean(asian_call_geometric) * discount_factor,
                'geometric_average_put': np.mean(asian_put_geometric) * discount_factor
            }

        except Exception as e:
            logger.warning(f"Error pricing Asian options: {e}")
            return {}

    def _price_lookback_options(self, S: float, K: float, T: float,
                              sigma: float, r: float, q: float) -> Dict[str, float]:
        """Price lookback options"""
        try:
            # Monte Carlo simulation
            np.random.seed(42)
            dt = T / self.time_steps
            paths = self.simulation_paths

            # Generate price paths
            Z = np.random.standard_normal((paths, self.time_steps))
            price_paths = np.zeros((paths, self.time_steps + 1))
            price_paths[:, 0] = S

            for t in range(1, self.time_steps + 1):
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
                )

            # Calculate lookback payoffs
            max_prices = np.max(price_paths, axis=1)
            min_prices = np.min(price_paths, axis=1)
            final_prices = price_paths[:, -1]

            # Fixed strike lookback
            lookback_call_fixed = np.maximum(max_prices - K, 0)
            lookback_put_fixed = np.maximum(K - min_prices, 0)

            # Floating strike lookback
            lookback_call_floating = final_prices - min_prices
            lookback_put_floating = max_prices - final_prices

            # Discount and average
            discount_factor = np.exp(-r * T)

            return {
                'fixed_strike_call': np.mean(lookback_call_fixed) * discount_factor,
                'fixed_strike_put': np.mean(lookback_put_fixed) * discount_factor,
                'floating_strike_call': np.mean(lookback_call_floating) * discount_factor,
                'floating_strike_put': np.mean(lookback_put_floating) * discount_factor
            }

        except Exception as e:
            logger.warning(f"Error pricing lookback options: {e}")
            return {}

    def _price_compound_options(self, S: float, strike_prices: List[float],
                              T: float, sigma: float, r: float, q: float) -> Dict[str, float]:
        """Price compound options (options on options)"""
        try:
            if len(strike_prices) < 2:
                return {}

            K1 = strike_prices[0]  # Strike of underlying option
            K2 = strike_prices[1] if len(strike_prices) > 1 else K1 * 1.1  # Strike of compound option

            T1 = T * 0.5  # Time to expiry of compound option
            T2 = T       # Time to expiry of underlying option

            # Call on call
            call_on_call = self._geske_compound_option(S, K1, K2, T1, T2, sigma, r, q, 'call_on_call')

            # Put on call
            put_on_call = self._geske_compound_option(S, K1, K2, T1, T2, sigma, r, q, 'put_on_call')

            # Call on put
            call_on_put = self._geske_compound_option(S, K1, K2, T1, T2, sigma, r, q, 'call_on_put')

            # Put on put
            put_on_put = self._geske_compound_option(S, K1, K2, T1, T2, sigma, r, q, 'put_on_put')

            return {
                'call_on_call': call_on_call,
                'put_on_call': put_on_call,
                'call_on_put': call_on_put,
                'put_on_put': put_on_put,
                'compound_strike': K2,
                'underlying_strike': K1
            }

        except Exception as e:
            logger.warning(f"Error pricing compound options: {e}")
            return {}

    def _geske_compound_option(self, S: float, K1: float, K2: float,
                             T1: float, T2: float, sigma: float, r: float, q: float,
                             option_type: str) -> float:
        """Geske model for compound options (simplified)"""
        try:
            # Simplified compound option pricing using Monte Carlo
            np.random.seed(42)
            paths = 10000

            # Simulate to T1
            Z1 = np.random.standard_normal(paths)
            S_T1 = S * np.exp((r - q - 0.5 * sigma**2) * T1 + sigma * np.sqrt(T1) * Z1)

            # Calculate option values at T1
            option_values_T1 = []
            for s_t1 in S_T1:
                if option_type == 'call_on_call':
                    underlying_option_value = self._black_scholes_call(s_t1, K1, T2 - T1, sigma, r, q)
                    compound_payoff = max(0, underlying_option_value - K2)
                elif option_type == 'put_on_call':
                    underlying_option_value = self._black_scholes_call(s_t1, K1, T2 - T1, sigma, r, q)
                    compound_payoff = max(0, K2 - underlying_option_value)
                elif option_type == 'call_on_put':
                    underlying_option_value = self._black_scholes_put(s_t1, K1, T2 - T1, sigma, r, q)
                    compound_payoff = max(0, underlying_option_value - K2)
                else:  # put_on_put
                    underlying_option_value = self._black_scholes_put(s_t1, K1, T2 - T1, sigma, r, q)
                    compound_payoff = max(0, K2 - underlying_option_value)

                option_values_T1.append(compound_payoff)

            # Discount back to present
            compound_option_value = np.mean(option_values_T1) * np.exp(-r * T1)
            return compound_option_value

        except Exception:
            return 0.0

    def _price_rainbow_options(self, S: float, K: float, T: float,
                             sigma: float, r: float) -> Dict[str, float]:
        """Price rainbow (multi-asset) options"""
        try:
            # Simulate two correlated assets
            np.random.seed(42)
            dt = T / self.time_steps
            paths = self.simulation_paths
            correlation = 0.5

            # Generate correlated random numbers
            Z1 = np.random.standard_normal((paths, self.time_steps))
            Z2 = correlation * Z1 + np.sqrt(1 - correlation**2) * np.random.standard_normal((paths, self.time_steps))

            # Asset 1 (original asset)
            S1_paths = np.zeros((paths, self.time_steps + 1))
            S1_paths[:, 0] = S

            # Asset 2 (related asset, e.g., sector ETF)
            S2_paths = np.zeros((paths, self.time_steps + 1))
            S2_paths[:, 0] = S * 0.8  # Different starting price

            sigma2 = sigma * 1.2  # Different volatility

            for t in range(1, self.time_steps + 1):
                S1_paths[:, t] = S1_paths[:, t-1] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z1[:, t-1]
                )
                S2_paths[:, t] = S2_paths[:, t-1] * np.exp(
                    (r - 0.5 * sigma2**2) * dt + sigma2 * np.sqrt(dt) * Z2[:, t-1]
                )

            final_S1 = S1_paths[:, -1]
            final_S2 = S2_paths[:, -1]

            # Rainbow option payoffs
            max_of_two = np.maximum(final_S1, final_S2)
            min_of_two = np.minimum(final_S1, final_S2)

            # Different rainbow options
            call_on_max = np.maximum(max_of_two - K, 0)
            call_on_min = np.maximum(min_of_two - K, 0)
            outperformance = np.maximum(final_S1 - final_S2, 0)

            # Discount and average
            discount_factor = np.exp(-r * T)

            return {
                'call_on_maximum': np.mean(call_on_max) * discount_factor,
                'call_on_minimum': np.mean(call_on_min) * discount_factor,
                'outperformance_option': np.mean(outperformance) * discount_factor,
                'asset_correlation': correlation
            }

        except Exception as e:
            logger.warning(f"Error pricing rainbow options: {e}")
            return {}

    def _analyze_path_dependency(self, S: float, sigma: float, T: float) -> Dict[str, Any]:
        """Analyze path dependency effects"""
        try:
            # Compare path-dependent vs path-independent option values
            # This is a simplified analysis
            path_dependent_premium = sigma * np.sqrt(T) * S * 0.1  # Rough estimate

            return {
                'path_dependency_premium': path_dependent_premium,
                'volatility_impact': sigma,
                'time_impact': T,
                'path_types_analyzed': ['barrier', 'asian', 'lookback'],
                'early_exercise_impact': 'moderate' if sigma > 0.3 else 'low'
            }

        except Exception:
            return {}

    def _calculate_greeks(self, S: float, K: float, T: float,
                        sigma: float, r: float, q: float) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            h = 0.01  # Small increment for numerical differentiation

            # Base option price
            base_price = self._black_scholes_call(S, K, T, sigma, r, q)

            # Delta (price sensitivity)
            delta = (self._black_scholes_call(S + h, K, T, sigma, r, q) - base_price) / h

            # Gamma (delta sensitivity)
            gamma = (self._black_scholes_call(S + h, K, T, sigma, r, q) +
                    self._black_scholes_call(S - h, K, T, sigma, r, q) - 2 * base_price) / (h**2)

            # Theta (time decay)
            theta = (self._black_scholes_call(S, K, T - h/365, sigma, r, q) - base_price) / (h/365)

            # Vega (volatility sensitivity)
            vega = (self._black_scholes_call(S, K, T, sigma + h, r, q) - base_price) / h

            # Rho (interest rate sensitivity)
            rho = (self._black_scholes_call(S, K, T, sigma, r + h, q) - base_price) / h

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }

        except Exception:
            return {}

    def _calculate_optimal_exercise_boundary(self, K: float, T: float,
                                           sigma: float, r: float) -> List[float]:
        """Calculate optimal exercise boundary for American options"""
        try:
            # Simplified exercise boundary calculation
            time_points = np.linspace(0, T, 20)
            exercise_boundary = []

            for t in time_points:
                # Critical stock price for early exercise (simplified)
                # This is a rough approximation
                if t < T:
                    critical_price = K * (1 + r * (T - t) + sigma * np.sqrt(T - t))
                else:
                    critical_price = K

                exercise_boundary.append(critical_price)

            return exercise_boundary

        except Exception:
            return []

    def _calculate_monte_carlo_confidence(self, S: float, K: float, T: float,
                                        sigma: float, r: float) -> Dict[str, float]:
        """Calculate confidence intervals for Monte Carlo prices"""
        try:
            # Run multiple Monte Carlo simulations
            simulation_results = []

            for _ in range(10):  # 10 independent simulations
                np.random.seed(None)  # Random seed each time
                call_price = self._monte_carlo_european_call(S, K, T, sigma, r)
                simulation_results.append(call_price)

            mean_price = np.mean(simulation_results)
            std_price = np.std(simulation_results)

            return {
                'mean_price': mean_price,
                'standard_error': std_price,
                '95_confidence_lower': mean_price - 1.96 * std_price,
                '95_confidence_upper': mean_price + 1.96 * std_price,
                'confidence_width': 3.92 * std_price
            }

        except Exception:
            return {}

    def _monte_carlo_european_call(self, S: float, K: float, T: float,
                                 sigma: float, r: float) -> float:
        """Simple Monte Carlo European call pricing"""
        try:
            paths = 10000
            Z = np.random.standard_normal(paths)
            S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            payoffs = np.maximum(S_T - K, 0)
            return np.mean(payoffs) * np.exp(-r * T)

        except Exception:
            return 0.0

    def _generate_methodology_notes(self) -> List[str]:
        """Generate methodology notes"""
        return [
            "Advanced options pricing using multiple numerical methods",
            "American options priced using binomial trees with early exercise optimization",
            "Barrier options priced using Monte Carlo simulation with path monitoring",
            "Asian options use both arithmetic and geometric averaging",
            "Lookback options include both fixed and floating strike variants",
            "Compound options use Geske-style nested option valuation",
            "Rainbow options model correlation between multiple underlying assets",
            "Greeks calculated using numerical differentiation",
            "Monte Carlo confidence intervals based on multiple simulation runs",
            "Optimal exercise boundaries approximate using analytical methods"
        ]

    def _create_empty_results(self) -> AdvancedOptionResults:
        """Create empty results for error cases"""
        return AdvancedOptionResults(
            american_call_value=0.0,
            american_put_value=0.0,
            european_call_value=0.0,
            european_put_value=0.0,
            barrier_option_values={},
            asian_option_values={},
            lookback_option_values={},
            compound_option_values={},
            rainbow_option_values={},
            early_exercise_premium=0.0,
            path_dependency_analysis={},
            greeks_analysis={},
            optimal_exercise_boundary=[],
            monte_carlo_confidence={},
            methodology_notes=["Advanced options modeling could not be completed"]
        )

def create_advanced_options_modeler():
    """Factory function to create AdvancedOptionsModeler"""
    return AdvancedOptionsModeler()