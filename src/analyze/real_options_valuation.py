"""
Real Options Valuation Module
For growth companies with high uncertainty and strategic flexibility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy.stats import norm
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

@dataclass
class RealOptionResults:
    """Results from real options valuation analysis"""
    black_scholes_value: float
    binomial_tree_value: float
    monte_carlo_value: float
    expansion_option_value: float
    abandonment_option_value: float
    switching_option_value: float
    timing_option_value: float
    option_breakdown: Dict[str, float]
    volatility_analysis: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    confidence_score: float
    methodology_notes: List[str]

class RealOptionsValuationAnalyzer:
    """
    Comprehensive real options valuation for growth companies
    """

    def __init__(self):
        self.market_data = MarketDataProvider()
        self.risk_free_rate = 0.045  # Will be updated with live data
        self.option_types = [
            'expansion',
            'abandonment',
            'switching',
            'timing'
        ]

    def analyze_real_options(self, dataset) -> RealOptionResults:
        """
        Perform comprehensive real options valuation analysis
        """
        try:
            # Extract financial data
            financials = dataset.financials
            snapshot = dataset.snapshot

            # Update risk-free rate from market data
            self._update_risk_free_rate()

            # Calculate company volatility
            volatility_analysis = self._calculate_volatility(dataset)

            # Identify and value real options
            option_breakdown = self._identify_real_options(dataset)

            # Calculate option values using different methods
            bs_value = self._calculate_black_scholes_value(dataset, volatility_analysis)
            binomial_value = self._calculate_binomial_tree_value(dataset, volatility_analysis)
            mc_value = self._calculate_monte_carlo_value(dataset, volatility_analysis)

            # Calculate specific option types
            expansion_value = self._calculate_expansion_option(dataset, volatility_analysis)
            abandonment_value = self._calculate_abandonment_option(dataset, volatility_analysis)
            switching_value = self._calculate_switching_option(dataset, volatility_analysis)
            timing_value = self._calculate_timing_option(dataset, volatility_analysis)

            # Sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(
                dataset, volatility_analysis
            )

            # Assess confidence
            confidence_score = self._assess_confidence(dataset, volatility_analysis)

            # Generate methodology notes
            methodology_notes = self._generate_methodology_notes(
                option_breakdown, volatility_analysis
            )

            return RealOptionResults(
                black_scholes_value=bs_value,
                binomial_tree_value=binomial_value,
                monte_carlo_value=mc_value,
                expansion_option_value=expansion_value,
                abandonment_option_value=abandonment_value,
                switching_option_value=switching_value,
                timing_option_value=timing_value,
                option_breakdown=option_breakdown,
                volatility_analysis=volatility_analysis,
                sensitivity_analysis=sensitivity_analysis,
                confidence_score=confidence_score,
                methodology_notes=methodology_notes
            )

        except Exception as e:
            logger.error(f"Error in real options valuation: {e}")
            return self._create_empty_results()

    def _update_risk_free_rate(self):
        """Update risk-free rate from live market data"""
        try:
            # Get live risk-free rate from market data
            live_rate = self.market_data.get_risk_free_rate()
            if live_rate:
                self.risk_free_rate = live_rate
                logger.info(f"✅ Updated risk-free rate to live data: {live_rate:.3%}")
            else:
                self.risk_free_rate = 0.045  # Conservative fallback
        except Exception as e:
            logger.warning(f"Failed to update risk-free rate: {e}")
            self.risk_free_rate = 0.045

    def _calculate_volatility(self, dataset) -> Dict[str, float]:
        """Calculate various volatility measures"""
        volatility_data = {}

        try:
            # Get sector-specific volatility from market data
            sector = dataset.snapshot.sector if hasattr(dataset.snapshot, 'sector') else 'Technology'
            market_volatility = self.market_data.get_industry_volatility_data(sector)

            # Historical stock price volatility
            if hasattr(dataset, 'historical_data') and dataset.historical_data is not None:
                prices = dataset.historical_data['Close']
                returns = prices.pct_change().dropna()

                # Annualized volatility
                volatility_data['stock_volatility'] = returns.std() * np.sqrt(252)
            else:
                # Use market-derived sector volatility instead of hardcoded default
                volatility_data['stock_volatility'] = market_volatility.get('annual_volatility', 0.30)

            # Revenue volatility
            income_statement = dataset.financials.income_statement
            if income_statement is not None and not income_statement.empty:
                revenue = income_statement.get('Total Revenue', pd.Series())
                if len(revenue) > 3:
                    revenue_growth = revenue.pct_change().dropna()
                    volatility_data['revenue_volatility'] = revenue_growth.std()
                else:
                    # Use market-derived sector volatility scaled for revenue
                    volatility_data['revenue_volatility'] = market_volatility.get('vol_90d', 0.25) * 0.8

            # EBITDA volatility
            if income_statement is not None and not income_statement.empty:
                ebitda = income_statement.get('EBITDA', pd.Series())
                if len(ebitda) > 3:
                    ebitda_growth = ebitda.pct_change().dropna()
                    volatility_data['ebitda_volatility'] = ebitda_growth.std()
                else:
                    # EBITDA typically more volatile than revenue
                    volatility_data['ebitda_volatility'] = market_volatility.get('vol_30d', 0.35) * 1.2

            # Cash flow volatility
            cash_flow = dataset.financials.cash_flow
            if cash_flow is not None and not cash_flow.empty:
                fcf = cash_flow.get('Free Cash Flow', pd.Series())
                if len(fcf) > 3:
                    fcf_growth = fcf.pct_change().dropna()
                    volatility_data['fcf_volatility'] = fcf_growth.std()
                else:
                    # FCF typically most volatile metric
                    volatility_data['fcf_volatility'] = market_volatility.get('annual_volatility', 0.40) * 1.5

            # Composite volatility for option pricing
            volatilities = [v for v in volatility_data.values() if v > 0]
            volatility_data['composite_volatility'] = np.mean(volatilities) if volatilities else market_volatility.get('current_vol', 0.30)

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            # Use market-derived fallbacks instead of hardcoded values
            try:
                sector = dataset.snapshot.sector if hasattr(dataset.snapshot, 'sector') else 'Technology'
                market_volatility = self.market_data.get_industry_volatility_data(sector)
                volatility_data = {
                    'stock_volatility': market_volatility.get('annual_volatility', 0.30),
                    'revenue_volatility': market_volatility.get('vol_90d', 0.25),
                    'ebitda_volatility': market_volatility.get('vol_30d', 0.35),
                    'fcf_volatility': market_volatility.get('annual_volatility', 0.40) * 1.2,
                    'composite_volatility': market_volatility.get('current_vol', 0.30)
                }
            except Exception:
                # Absolute last resort
                volatility_data = {
                    'stock_volatility': 0.30,
                    'revenue_volatility': 0.25,
                    'ebitda_volatility': 0.35,
                    'fcf_volatility': 0.40,
                    'composite_volatility': 0.30
                }

        return volatility_data

    def _identify_real_options(self, dataset) -> Dict[str, float]:
        """Identify potential real options in the business"""
        options = {}

        try:
            # Extract key metrics
            balance_sheet = dataset.financials.balance_sheet
            income_statement = dataset.financials.income_statement
            cash_flow = dataset.financials.cash_flow

            if balance_sheet is not None and not balance_sheet.empty:
                latest_bs = balance_sheet.iloc[-1]
                total_assets = latest_bs.get('Total Assets', 0)
                ppe = latest_bs.get('Property Plant Equipment Net', 0)
                cash = latest_bs.get('Cash And Cash Equivalents', 0)

                # Expansion options (based on growth capex and R&D)
                if cash_flow is not None and not cash_flow.empty:
                    latest_cf = cash_flow.iloc[-1]
                    capex = abs(latest_cf.get('Capital Expenditure', 0))
                    options['expansion_capex_base'] = capex * 2  # Future expansion potential

                # R&D as expansion options
                if income_statement is not None and not income_statement.empty:
                    latest_is = income_statement.iloc[-1]
                    rd_expense = latest_is.get('Research And Development', 0)
                    options['rd_option_value'] = rd_expense * 5  # R&D multiplier

                # Asset utilization options
                if total_assets > 0:
                    revenue = latest_is.get('Total Revenue', 0)
                    asset_turnover = revenue / total_assets
                    if asset_turnover < 1.0:  # Underutilized assets
                        options['asset_utilization'] = (1.0 - asset_turnover) * total_assets * 0.1

                # Cash optionality
                options['cash_deployment'] = cash * 0.15  # Cash deployment options

                # Market expansion (international, new products)
                options['market_expansion'] = revenue * 0.2 if revenue > 0 else 0

        except Exception as e:
            logger.error(f"Error identifying real options: {e}")

        return options

    def _calculate_black_scholes_value(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Calculate option value using Black-Scholes model"""
        try:
            # Current value (underlying asset)
            current_value = dataset.snapshot.market_cap or dataset.snapshot.enterprise_value

            # Strike price (investment required)
            balance_sheet = dataset.financials.balance_sheet
            cash_flow = dataset.financials.cash_flow

            if cash_flow is not None and not cash_flow.empty:
                latest_cf = cash_flow.iloc[-1]
                strike_price = abs(latest_cf.get('Capital Expenditure', 0)) * 3
            else:
                strike_price = current_value * 0.5

            # Time to expiration (years)
            time_to_expiry = 3.0  # Typical strategic planning horizon

            # Volatility
            volatility = volatility_analysis.get('composite_volatility', 0.35)

            # Black-Scholes formula for call option
            if current_value > 0 and strike_price > 0:
                d1 = (np.log(current_value / strike_price) +
                      (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                     (volatility * np.sqrt(time_to_expiry))

                d2 = d1 - volatility * np.sqrt(time_to_expiry)

                option_value = (current_value * norm.cdf(d1) -
                               strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))

                return max(option_value, 0)

        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")

        return 0.0

    def _calculate_binomial_tree_value(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Calculate option value using binomial tree model"""
        try:
            # Parameters
            current_value = dataset.snapshot.market_cap or dataset.snapshot.enterprise_value
            volatility = volatility_analysis.get('composite_volatility', 0.35)
            time_to_expiry = 3.0
            steps = 50

            # Calculate up and down factors
            dt = time_to_expiry / steps
            u = np.exp(volatility * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(self.risk_free_rate * dt) - d) / (u - d)

            # Strike price
            cash_flow = dataset.financials.cash_flow
            if cash_flow is not None and not cash_flow.empty:
                latest_cf = cash_flow.iloc[-1]
                strike_price = abs(latest_cf.get('Capital Expenditure', 0)) * 3
            else:
                strike_price = current_value * 0.5

            # Build binomial tree
            option_values = np.zeros((steps + 1, steps + 1))

            # Calculate terminal option values
            for i in range(steps + 1):
                asset_price = current_value * (u ** (steps - i)) * (d ** i)
                option_values[i, steps] = max(asset_price - strike_price, 0)

            # Backward induction
            for j in range(steps - 1, -1, -1):
                for i in range(j + 1):
                    option_values[i, j] = np.exp(-self.risk_free_rate * dt) * \
                                         (p * option_values[i, j + 1] +
                                          (1 - p) * option_values[i + 1, j + 1])

            return option_values[0, 0]

        except Exception as e:
            logger.error(f"Error in binomial tree calculation: {e}")
            return 0.0

    def _calculate_monte_carlo_value(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Calculate option value using Monte Carlo simulation"""
        try:
            # Parameters
            current_value = dataset.snapshot.market_cap or dataset.snapshot.enterprise_value
            volatility = volatility_analysis.get('composite_volatility', 0.35)
            time_to_expiry = 3.0
            num_simulations = 10000

            # Strike price
            cash_flow = dataset.financials.cash_flow
            if cash_flow is not None and not cash_flow.empty:
                latest_cf = cash_flow.iloc[-1]
                strike_price = abs(latest_cf.get('Capital Expenditure', 0)) * 3
            else:
                strike_price = current_value * 0.5

            # Monte Carlo simulation
            np.random.seed(42)
            z = np.random.standard_normal(num_simulations)

            # Terminal asset values
            terminal_values = current_value * np.exp(
                (self.risk_free_rate - 0.5 * volatility**2) * time_to_expiry +
                volatility * np.sqrt(time_to_expiry) * z
            )

            # Option payoffs
            option_payoffs = np.maximum(terminal_values - strike_price, 0)

            # Discounted expected payoff
            option_value = np.exp(-self.risk_free_rate * time_to_expiry) * np.mean(option_payoffs)

            return option_value

        except Exception as e:
            logger.error(f"Error in Monte Carlo calculation: {e}")
            return 0.0

    def _calculate_expansion_option(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Calculate expansion option value"""
        try:
            income_statement = dataset.financials.income_statement
            if income_statement is None or income_statement.empty:
                return 0.0

            # Current revenue as base
            latest_is = income_statement.iloc[-1]
            current_revenue = latest_is.get('Total Revenue', 0)

            # Expansion potential (doubling revenue)
            expanded_revenue = current_revenue * 2
            expansion_value = expanded_revenue - current_revenue

            # Apply option model with expansion-specific parameters
            volatility = volatility_analysis.get('revenue_volatility', 0.25)
            time_to_expiry = 5.0  # Longer horizon for expansion

            # Investment required for expansion
            cash_flow = dataset.financials.cash_flow
            if cash_flow is not None and not cash_flow.empty:
                latest_cf = cash_flow.iloc[-1]
                expansion_cost = abs(latest_cf.get('Capital Expenditure', 0)) * 4
            else:
                expansion_cost = expansion_value * 0.6

            # Simplified Black-Scholes for expansion
            if expansion_value > 0 and expansion_cost > 0:
                d1 = (np.log(expansion_value / expansion_cost) +
                      (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                     (volatility * np.sqrt(time_to_expiry))

                d2 = d1 - volatility * np.sqrt(time_to_expiry)

                option_value = (expansion_value * norm.cdf(d1) -
                               expansion_cost * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))

                return max(option_value, 0)

        except Exception as e:
            logger.error(f"Error calculating expansion option: {e}")

        return 0.0

    def _calculate_abandonment_option(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Calculate abandonment option value (put option)"""
        try:
            balance_sheet = dataset.financials.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                return 0.0

            # Liquidation value as strike price
            latest_bs = balance_sheet.iloc[-1]
            total_assets = latest_bs.get('Total Assets', 0)
            liquidation_value = total_assets * 0.6  # Liquidation discount

            # Current enterprise value
            current_value = dataset.snapshot.market_cap or liquidation_value

            # Put option parameters
            volatility = volatility_analysis.get('composite_volatility', 0.35)
            time_to_expiry = 2.0  # Shorter horizon for abandonment decision

            # Black-Scholes put option
            if current_value > 0 and liquidation_value > 0:
                d1 = (np.log(current_value / liquidation_value) +
                      (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                     (volatility * np.sqrt(time_to_expiry))

                d2 = d1 - volatility * np.sqrt(time_to_expiry)

                put_value = (liquidation_value * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                            current_value * norm.cdf(-d1))

                return max(put_value, 0)

        except Exception as e:
            logger.error(f"Error calculating abandonment option: {e}")

        return 0.0

    def _calculate_switching_option(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Calculate switching option value (flexibility to change strategy)"""
        try:
            # Switching value based on revenue diversification potential
            income_statement = dataset.financials.income_statement
            if income_statement is None or income_statement.empty:
                return 0.0

            latest_is = income_statement.iloc[-1]
            current_revenue = latest_is.get('Total Revenue', 0)

            # Switching to alternative business model value
            switching_upside = current_revenue * 0.3  # 30% revenue upside
            switching_cost = current_revenue * 0.1    # 10% revenue cost

            # Option parameters
            volatility = volatility_analysis.get('revenue_volatility', 0.25)
            time_to_expiry = 4.0

            # Simplified switching option as spread option
            if switching_upside > switching_cost:
                net_switching_value = switching_upside - switching_cost
                return net_switching_value * 0.4  # Probability-adjusted value

        except Exception as e:
            logger.error(f"Error calculating switching option: {e}")

        return 0.0

    def _calculate_timing_option(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Calculate timing option value (when to invest)"""
        try:
            # Value of waiting to invest
            cash_flow = dataset.financials.cash_flow
            if cash_flow is None or cash_flow.empty:
                return 0.0

            latest_cf = cash_flow.iloc[-1]
            potential_investment = abs(latest_cf.get('Capital Expenditure', 0)) * 2

            # Value of flexibility to time investment
            volatility = volatility_analysis.get('composite_volatility', 0.35)

            # Higher volatility increases timing option value
            timing_premium = potential_investment * volatility * 0.2

            return timing_premium

        except Exception as e:
            logger.error(f"Error calculating timing option: {e}")

        return 0.0

    def _perform_sensitivity_analysis(self, dataset, volatility_analysis: Dict[str, float]) -> Dict[str, float]:
        """Perform sensitivity analysis on key parameters"""
        sensitivities = {}

        try:
            base_value = self._calculate_black_scholes_value(dataset, volatility_analysis)
            base_volatility = volatility_analysis.get('composite_volatility', 0.35)

            # Volatility sensitivity
            high_vol = volatility_analysis.copy()
            high_vol['composite_volatility'] = base_volatility * 1.2
            high_vol_value = self._calculate_black_scholes_value(dataset, high_vol)

            low_vol = volatility_analysis.copy()
            low_vol['composite_volatility'] = base_volatility * 0.8
            low_vol_value = self._calculate_black_scholes_value(dataset, low_vol)

            sensitivities['volatility_sensitivity'] = (high_vol_value - low_vol_value) / (0.4 * base_volatility)

            # Risk-free rate sensitivity
            original_rate = self.risk_free_rate

            self.risk_free_rate = original_rate * 1.2
            high_rate_value = self._calculate_black_scholes_value(dataset, volatility_analysis)

            self.risk_free_rate = original_rate * 0.8
            low_rate_value = self._calculate_black_scholes_value(dataset, volatility_analysis)

            # Restore original rate
            self.risk_free_rate = original_rate

            sensitivities['rate_sensitivity'] = (high_rate_value - low_rate_value) / (0.4 * original_rate)

        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")

        return sensitivities

    def _assess_confidence(self, dataset, volatility_analysis: Dict[str, float]) -> float:
        """Assess confidence in real options valuation"""
        try:
            confidence_factors = []

            # Data availability
            has_historicals = hasattr(dataset, 'historical_data') and dataset.historical_data is not None
            confidence_factors.append(0.8 if has_historicals else 0.5)

            # Volatility reliability
            vol = volatility_analysis.get('composite_volatility', 0)
            if 0.2 <= vol <= 0.8:  # Reasonable volatility range
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)

            # Company stage (real options more applicable to growth companies)
            income_statement = dataset.financials.income_statement
            if income_statement is not None and not income_statement.empty and len(income_statement) >= 3:
                revenue_growth = income_statement['Total Revenue'].pct_change().mean()
                if revenue_growth > 0.15:  # High growth
                    confidence_factors.append(0.8)
                elif revenue_growth > 0.05:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.4)
            else:
                confidence_factors.append(0.5)

            return np.mean(confidence_factors)

        except Exception:
            return 0.5

    def _generate_methodology_notes(self, option_breakdown: Dict[str, float],
                                  volatility_analysis: Dict[str, float]) -> List[str]:
        """Generate methodology notes"""
        notes = [
            "Real options valuation using Black-Scholes, binomial tree, and Monte Carlo methods",
            f"Composite volatility: {volatility_analysis.get('composite_volatility', 0):.1%}",
            f"Risk-free rate: {self.risk_free_rate:.1%}"
        ]

        if option_breakdown:
            notes.append(f"Identified {len(option_breakdown)} potential real options")

        notes.extend([
            "Expansion options valued based on growth capital requirements",
            "Abandonment options calculated as liquidation put options",
            "Switching options represent strategic flexibility value",
            "Timing options capture investment timing flexibility",
            "Higher volatility increases option values (convexity benefit)"
        ])

        return notes

    def _create_empty_results(self) -> RealOptionResults:
        """Create empty results for error cases"""
        return RealOptionResults(
            black_scholes_value=0.0,
            binomial_tree_value=0.0,
            monte_carlo_value=0.0,
            expansion_option_value=0.0,
            abandonment_option_value=0.0,
            switching_option_value=0.0,
            timing_option_value=0.0,
            option_breakdown={},
            volatility_analysis={},
            sensitivity_analysis={},
            confidence_score=0.0,
            methodology_notes=["Real options valuation could not be calculated"]
        )

def create_real_options_analyzer():
    """Factory function to create RealOptionsValuationAnalyzer"""
    return RealOptionsValuationAnalyzer()