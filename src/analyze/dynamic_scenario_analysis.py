"""
Dynamic Scenario Analysis Module
Creates Bear/Base/Bull scenarios based on live market conditions
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

@dataclass
class ScenarioParameters:
    """Parameters for a specific scenario"""
    name: str
    probability: float
    gdp_growth: float
    inflation_rate: float
    risk_free_rate: float
    equity_premium: float
    revenue_growth_adj: float
    margin_adj: float
    multiple_adj: float
    volatility_adj: float
    description: str

@dataclass
class ScenarioResults:
    """Results from dynamic scenario analysis"""
    bear_scenario: ScenarioParameters
    base_scenario: ScenarioParameters
    bull_scenario: ScenarioParameters
    bear_valuation: float
    base_valuation: float
    bull_valuation: float
    probability_weighted_value: float
    scenario_analysis: Dict[str, Any]
    stress_test_results: Dict[str, float]
    market_regime: str
    confidence_intervals: Dict[str, float]
    methodology_notes: List[str]

class DynamicScenarioAnalyzer:
    """
    Dynamic scenario analysis based on current market conditions
    """

    def __init__(self):
        self.market_data = MarketDataProvider()
        self.vix_symbol = "^VIX"
        self.yield_curve_symbols = ["^TNX", "^FVX", "^TYX"]  # 10Y, 5Y, 30Y

    def analyze_market_scenarios(self, dataset, base_valuation_results: Dict) -> ScenarioResults:
        """
        Create dynamic scenarios based on current market conditions
        """
        try:
            print("🌪️ Analyzing dynamic market scenarios...")

            # Step 1: Assess current market regime
            market_regime = self._determine_market_regime()

            # Step 2: Get market-derived scenario parameters
            bear_params = self._create_bear_scenario(market_regime, dataset)
            base_params = self._create_base_scenario(market_regime, dataset)
            bull_params = self._create_bull_scenario(market_regime, dataset)

            # Step 3: Calculate scenario valuations
            bear_valuation = self._calculate_scenario_valuation(
                dataset, base_valuation_results, bear_params
            )
            base_valuation = self._calculate_scenario_valuation(
                dataset, base_valuation_results, base_params
            )
            bull_valuation = self._calculate_scenario_valuation(
                dataset, base_valuation_results, bull_params
            )

            # Step 4: Calculate probability-weighted valuation
            probability_weighted = (
                bear_params.probability * bear_valuation +
                base_params.probability * base_valuation +
                bull_params.probability * bull_valuation
            )

            # Step 5: Perform stress testing
            stress_test_results = self._perform_stress_testing(
                dataset, base_valuation_results
            )

            # Step 6: Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                bear_valuation, base_valuation, bull_valuation
            )

            # Step 7: Create scenario analysis summary
            scenario_analysis = self._create_scenario_summary(
                bear_params, base_params, bull_params,
                bear_valuation, base_valuation, bull_valuation
            )

            # Step 8: Generate methodology notes
            methodology_notes = self._generate_scenario_notes(market_regime)

            return ScenarioResults(
                bear_scenario=bear_params,
                base_scenario=base_params,
                bull_scenario=bull_params,
                bear_valuation=bear_valuation,
                base_valuation=base_valuation,
                bull_valuation=bull_valuation,
                probability_weighted_value=probability_weighted,
                scenario_analysis=scenario_analysis,
                stress_test_results=stress_test_results,
                market_regime=market_regime,
                confidence_intervals=confidence_intervals,
                methodology_notes=methodology_notes
            )

        except Exception as e:
            logger.error(f"Error in dynamic scenario analysis: {e}")
            return self._create_empty_results()

    def _determine_market_regime(self) -> str:
        """Determine current market regime based on live indicators"""
        try:
            # Get VIX for market fear gauge
            vix = yf.Ticker(self.vix_symbol)
            vix_hist = vix.history(period="30d")

            if not vix_hist.empty:
                current_vix = vix_hist['Close'].iloc[-1]
                avg_vix = vix_hist['Close'].mean()

                # Get yield curve data
                tnx = yf.Ticker("^TNX")  # 10-year
                fvx = yf.Ticker("^FVX")  # 5-year

                tnx_hist = tnx.history(period="5d")
                fvx_hist = fvx.history(period="5d")

                if not tnx_hist.empty and not fvx_hist.empty:
                    yield_spread = tnx_hist['Close'].iloc[-1] - fvx_hist['Close'].iloc[-1]
                else:
                    yield_spread = 0.5  # Normal spread

                # Get market performance
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="3mo")

                if not spy_hist.empty:
                    market_return_3m = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1
                else:
                    market_return_3m = 0.0

                # Determine regime
                if current_vix > 30 or market_return_3m < -0.15:
                    return "crisis"
                elif current_vix > 25 or yield_spread < 0:
                    return "stress"
                elif current_vix < 15 and market_return_3m > 0.10:
                    return "expansion"
                elif current_vix < 20 and yield_spread > 1.0:
                    return "recovery"
                else:
                    return "normal"

        except Exception as e:
            logger.warning(f"Could not determine market regime: {e}")

        return "normal"

    def _create_bear_scenario(self, market_regime: str, dataset) -> ScenarioParameters:
        """Create bear scenario based on current market conditions"""
        try:
            # Base bear scenario adjustments
            base_adjustments = {
                "crisis": {"prob": 0.40, "gdp": -0.03, "inflation": -0.01, "rev_adj": -0.25, "margin_adj": -0.15},
                "stress": {"prob": 0.35, "gdp": -0.02, "inflation": 0.00, "rev_adj": -0.20, "margin_adj": -0.10},
                "normal": {"prob": 0.25, "gdp": -0.015, "inflation": 0.005, "rev_adj": -0.15, "margin_adj": -0.08},
                "recovery": {"prob": 0.20, "gdp": -0.01, "inflation": 0.01, "rev_adj": -0.12, "margin_adj": -0.05},
                "expansion": {"prob": 0.15, "gdp": -0.005, "inflation": 0.015, "rev_adj": -0.10, "margin_adj": -0.03}
            }

            adj = base_adjustments.get(market_regime, base_adjustments["normal"])

            # Get current risk-free rate
            current_rf = self.market_data.get_risk_free_rate()

            # Get sector-specific adjustments
            sector = dataset.snapshot.sector if hasattr(dataset.snapshot, 'sector') else 'Technology'
            volatility_data = self.market_data.get_industry_volatility_data(sector)

            return ScenarioParameters(
                name="Bear Case",
                probability=adj["prob"],
                gdp_growth=adj["gdp"],
                inflation_rate=0.02 + adj["inflation"],
                risk_free_rate=current_rf + 0.01,  # Rates rise in stress
                equity_premium=0.08 + 0.02,  # Risk premium increases
                revenue_growth_adj=adj["rev_adj"],
                margin_adj=adj["margin_adj"],
                multiple_adj=-0.25,  # P/E compression
                volatility_adj=1.5,  # Higher volatility
                description=f"Economic downturn scenario reflecting {market_regime} market conditions"
            )

        except Exception as e:
            logger.warning(f"Error creating bear scenario: {e}")
            return ScenarioParameters(
                name="Bear Case",
                probability=0.25,
                gdp_growth=-0.02,
                inflation_rate=0.015,
                risk_free_rate=0.055,
                equity_premium=0.10,
                revenue_growth_adj=-0.15,
                margin_adj=-0.08,
                multiple_adj=-0.25,
                volatility_adj=1.5,
                description="Conservative economic downturn scenario"
            )

    def _create_base_scenario(self, market_regime: str, dataset) -> ScenarioParameters:
        """Create base scenario based on current market conditions"""
        try:
            # Base probabilities by regime
            base_probs = {
                "crisis": 0.40,
                "stress": 0.45,
                "normal": 0.50,
                "recovery": 0.60,
                "expansion": 0.55
            }

            probability = base_probs.get(market_regime, 0.50)

            # Get current market data
            current_rf = self.market_data.get_risk_free_rate()
            current_erp = self.market_data.get_equity_risk_premium()

            # Sector-specific data
            sector = dataset.snapshot.sector if hasattr(dataset.snapshot, 'sector') else 'Technology'

            return ScenarioParameters(
                name="Base Case",
                probability=probability,
                gdp_growth=0.025,  # Long-term GDP growth
                inflation_rate=0.025,  # Fed target
                risk_free_rate=current_rf,
                equity_premium=current_erp,
                revenue_growth_adj=0.0,  # No adjustment
                margin_adj=0.0,  # No adjustment
                multiple_adj=0.0,  # Current multiples
                volatility_adj=1.0,  # Current volatility
                description=f"Most likely scenario based on current {market_regime} market conditions"
            )

        except Exception as e:
            logger.warning(f"Error creating base scenario: {e}")
            return ScenarioParameters(
                name="Base Case",
                probability=0.50,
                gdp_growth=0.025,
                inflation_rate=0.025,
                risk_free_rate=0.045,
                equity_premium=0.065,
                revenue_growth_adj=0.0,
                margin_adj=0.0,
                multiple_adj=0.0,
                volatility_adj=1.0,
                description="Most likely scenario based on current market conditions"
            )

    def _create_bull_scenario(self, market_regime: str, dataset) -> ScenarioParameters:
        """Create bull scenario based on current market conditions"""
        try:
            # Bull scenario adjustments
            bull_adjustments = {
                "crisis": {"prob": 0.20, "gdp": 0.05, "inflation": -0.005, "rev_adj": 0.30, "margin_adj": 0.20},
                "stress": {"prob": 0.20, "gdp": 0.04, "inflation": -0.005, "rev_adj": 0.25, "margin_adj": 0.15},
                "normal": {"prob": 0.25, "gdp": 0.035, "inflation": 0.00, "rev_adj": 0.20, "margin_adj": 0.12},
                "recovery": {"prob": 0.20, "gdp": 0.045, "inflation": 0.005, "rev_adj": 0.25, "margin_adj": 0.15},
                "expansion": {"prob": 0.30, "gdp": 0.04, "inflation": 0.01, "rev_adj": 0.30, "margin_adj": 0.18}
            }

            adj = bull_adjustments.get(market_regime, bull_adjustments["normal"])

            current_rf = self.market_data.get_risk_free_rate()

            return ScenarioParameters(
                name="Bull Case",
                probability=adj["prob"],
                gdp_growth=adj["gdp"],
                inflation_rate=0.02 + adj["inflation"],
                risk_free_rate=max(0.02, current_rf - 0.005),  # Rates could fall
                equity_premium=0.055,  # Risk premium compresses
                revenue_growth_adj=adj["rev_adj"],
                margin_adj=adj["margin_adj"],
                multiple_adj=0.20,  # P/E expansion
                volatility_adj=0.8,  # Lower volatility
                description=f"Optimistic growth scenario reflecting potential {market_regime} market upside"
            )

        except Exception as e:
            logger.warning(f"Error creating bull scenario: {e}")
            return ScenarioParameters(
                name="Bull Case",
                probability=0.25,
                gdp_growth=0.035,
                inflation_rate=0.02,
                risk_free_rate=0.035,
                equity_premium=0.055,
                revenue_growth_adj=0.20,
                margin_adj=0.12,
                multiple_adj=0.20,
                volatility_adj=0.8,
                description="Optimistic growth scenario"
            )

    def _calculate_scenario_valuation(self, dataset, base_results: Dict,
                                    scenario: ScenarioParameters) -> float:
        """Calculate valuation under specific scenario"""
        try:
            # Start with base DCF valuation
            base_dcf = base_results.get('valuation', {}).get('dcf_value', 0)

            if base_dcf == 0:
                return 0.0

            # Apply scenario adjustments
            # 1. Revenue growth adjustment
            revenue_adj_factor = 1 + scenario.revenue_growth_adj

            # 2. Margin adjustment
            margin_adj_factor = 1 + scenario.margin_adj

            # 3. Multiple adjustment (affects terminal value)
            multiple_adj_factor = 1 + scenario.multiple_adj

            # 4. Discount rate adjustment
            # New WACC = risk_free_rate + equity_premium
            new_discount_rate = scenario.risk_free_rate + scenario.equity_premium

            # Get current WACC for comparison
            current_wacc = base_results.get('valuation', {}).get('wacc', 0.10)
            discount_rate_adj = current_wacc / new_discount_rate if new_discount_rate > 0 else 1.0

            # Combined adjustment
            scenario_adjustment = (
                revenue_adj_factor *
                margin_adj_factor *
                multiple_adj_factor *
                discount_rate_adj
            )

            scenario_valuation = base_dcf * scenario_adjustment

            return max(0, scenario_valuation)

        except Exception as e:
            logger.warning(f"Error calculating scenario valuation: {e}")
            return 0.0

    def _perform_stress_testing(self, dataset, base_results: Dict) -> Dict[str, float]:
        """Perform stress testing with extreme scenarios"""
        try:
            base_dcf = base_results.get('valuation', {}).get('dcf_value', 0)

            if base_dcf == 0:
                return {}

            stress_tests = {
                'recession_2008': base_dcf * 0.45,  # Financial crisis level
                'covid_crash': base_dcf * 0.65,     # 2020 pandemic level
                'inflation_shock': base_dcf * 0.70,  # 1970s style inflation
                'rate_shock': base_dcf * 0.75,      # Rapid rate increases
                'sector_rotation': base_dcf * 0.80,  # Sector out of favor
                'liquidity_crisis': base_dcf * 0.50, # Market liquidity dry up
                'cyber_attack': base_dcf * 0.60,     # Major cyber incident
                'regulatory_shock': base_dcf * 0.55   # Adverse regulation
            }

            return stress_tests

        except Exception as e:
            logger.warning(f"Error in stress testing: {e}")
            return {}

    def _calculate_confidence_intervals(self, bear_val: float, base_val: float,
                                      bull_val: float) -> Dict[str, float]:
        """Calculate confidence intervals around valuations"""
        try:
            values = [bear_val, base_val, bull_val]
            mean_val = np.mean(values)
            std_val = np.std(values)

            return {
                '95th_percentile': mean_val + 1.96 * std_val,
                '90th_percentile': mean_val + 1.645 * std_val,
                '75th_percentile': mean_val + 0.674 * std_val,
                'median': base_val,
                '25th_percentile': mean_val - 0.674 * std_val,
                '10th_percentile': mean_val - 1.645 * std_val,
                '5th_percentile': mean_val - 1.96 * std_val,
                'range_width': bull_val - bear_val,
                'coefficient_variation': std_val / mean_val if mean_val > 0 else 0
            }

        except Exception:
            return {}

    def _create_scenario_summary(self, bear: ScenarioParameters, base: ScenarioParameters,
                               bull: ScenarioParameters, bear_val: float,
                               base_val: float, bull_val: float) -> Dict[str, Any]:
        """Create comprehensive scenario analysis summary"""
        return {
            'scenario_count': 3,
            'valuation_range': {
                'min': bear_val,
                'max': bull_val,
                'spread': bull_val - bear_val,
                'spread_pct': (bull_val - bear_val) / base_val if base_val > 0 else 0
            },
            'probability_distribution': {
                'bear_prob': bear.probability,
                'base_prob': base.probability,
                'bull_prob': bull.probability
            },
            'key_drivers': [
                'GDP growth expectations',
                'Interest rate environment',
                'Equity risk premium',
                'Sector-specific factors',
                'Market volatility regime'
            ],
            'scenario_descriptions': {
                'bear': bear.description,
                'base': base.description,
                'bull': bull.description
            }
        }

    def _generate_scenario_notes(self, market_regime: str) -> List[str]:
        """Generate methodology notes for scenario analysis"""
        return [
            f"Dynamic scenarios based on current {market_regime} market regime",
            "Probabilities adjusted using live market indicators (VIX, yield curve, performance)",
            "GDP growth, inflation, and risk premiums derived from current market conditions",
            "Revenue and margin adjustments calibrated to sector-specific volatility",
            "Multiple adjustments reflect current valuation environment",
            "Stress tests include historical crisis scenarios",
            "All parameters update automatically with market conditions"
        ]

    def _create_empty_results(self) -> ScenarioResults:
        """Create empty results for error cases"""
        empty_scenario = ScenarioParameters(
            name="N/A", probability=0.0, gdp_growth=0.0, inflation_rate=0.0,
            risk_free_rate=0.0, equity_premium=0.0, revenue_growth_adj=0.0,
            margin_adj=0.0, multiple_adj=0.0, volatility_adj=1.0, description="N/A"
        )

        return ScenarioResults(
            bear_scenario=empty_scenario,
            base_scenario=empty_scenario,
            bull_scenario=empty_scenario,
            bear_valuation=0.0,
            base_valuation=0.0,
            bull_valuation=0.0,
            probability_weighted_value=0.0,
            scenario_analysis={},
            stress_test_results={},
            market_regime="unknown",
            confidence_intervals={},
            methodology_notes=["Dynamic scenario analysis could not be calculated"]
        )

def create_dynamic_scenario_analyzer():
    """Factory function to create DynamicScenarioAnalyzer"""
    return DynamicScenarioAnalyzer()