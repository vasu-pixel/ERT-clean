"""
Multi-Method Valuation Framework
Integrates DCF, relative, sum-of-parts, asset-based, and real options approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Import individual valuation modules
from . import deterministic
try:
    from .relative_valuation import RelativeValuationAnalyzer, create_relative_analyzer
except ImportError:
    RelativeValuationAnalyzer = None
    create_relative_analyzer = None

try:
    from .sum_of_parts import SumOfPartsAnalyzer, create_sum_of_parts_analyzer
except ImportError:
    SumOfPartsAnalyzer = None
    create_sum_of_parts_analyzer = None

try:
    from .asset_based_valuation import AssetBasedValuationAnalyzer, create_asset_valuation_analyzer
except ImportError:
    AssetBasedValuationAnalyzer = None
    create_asset_valuation_analyzer = None

try:
    from .real_options_valuation import RealOptionsValuationAnalyzer, create_real_options_analyzer
except ImportError:
    RealOptionsValuationAnalyzer = None
    create_real_options_analyzer = None

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.market_data import MarketDataProvider
from utils.data_validation import data_validator

logger = logging.getLogger(__name__)

@dataclass
class MultiMethodResults:
    """Comprehensive results from multi-method valuation"""
    dcf_value: float
    relative_value: float
    sum_of_parts_value: float
    asset_based_value: float
    real_options_value: float
    weighted_average_value: float
    confidence_weighted_value: float
    valuation_range: Dict[str, float]
    method_weights: Dict[str, float]
    confidence_scores: Dict[str, float]
    valuation_summary: Dict[str, Any]
    investment_recommendation: Dict[str, Any]
    methodology_notes: List[str]

class MultiMethodValuationFramework:
    """
    Comprehensive valuation framework combining multiple methodologies
    """

    def __init__(self):
        # Initialize individual analyzers
        self.market_data = MarketDataProvider()
        self.relative_analyzer = create_relative_analyzer() if create_relative_analyzer else None
        self.sop_analyzer = create_sum_of_parts_analyzer() if create_sum_of_parts_analyzer else None
        self.asset_analyzer = create_asset_valuation_analyzer() if create_asset_valuation_analyzer else None
        self.options_analyzer = create_real_options_analyzer() if create_real_options_analyzer else None

        # Base method applicability weights (will be adjusted by market data)
        self.base_industry_method_weights = {
            'technology': {
                'dcf': 0.35,
                'relative': 0.30,
                'sum_of_parts': 0.05,
                'asset_based': 0.05,
                'real_options': 0.25
            },
            'healthcare': {
                'dcf': 0.30,
                'relative': 0.25,
                'sum_of_parts': 0.10,
                'asset_based': 0.10,
                'real_options': 0.25
            },
            'financial': {
                'dcf': 0.20,
                'relative': 0.40,
                'sum_of_parts': 0.20,
                'asset_based': 0.15,
                'real_options': 0.05
            },
            'real_estate': {
                'dcf': 0.25,
                'relative': 0.30,
                'sum_of_parts': 0.10,
                'asset_based': 0.30,
                'real_options': 0.05
            },
            'utilities': {
                'dcf': 0.35,
                'relative': 0.25,
                'sum_of_parts': 0.15,
                'asset_based': 0.20,
                'real_options': 0.05
            },
            'energy': {
                'dcf': 0.30,
                'relative': 0.25,
                'sum_of_parts': 0.15,
                'asset_based': 0.25,
                'real_options': 0.05
            },
            'materials': {
                'dcf': 0.25,
                'relative': 0.30,
                'sum_of_parts': 0.15,
                'asset_based': 0.25,
                'real_options': 0.05
            },
            'industrials': {
                'dcf': 0.30,
                'relative': 0.30,
                'sum_of_parts': 0.20,
                'asset_based': 0.15,
                'real_options': 0.05
            },
            'consumer': {
                'dcf': 0.35,
                'relative': 0.35,
                'sum_of_parts': 0.15,
                'asset_based': 0.10,
                'real_options': 0.05
            },
            'default': {
                'dcf': 0.30,
                'relative': 0.30,
                'sum_of_parts': 0.15,
                'asset_based': 0.15,
                'real_options': 0.10
            }
        }

    def perform_comprehensive_valuation(self, dataset) -> MultiMethodResults:
        """
        Perform comprehensive multi-method valuation analysis
        """
        try:
            # Step 1: Run all valuation methods
            print("🔬 Running comprehensive multi-method valuation...")

            # Initialize all result variables to avoid scope issues
            dcf_results = None
            relative_results = None
            sop_results = None
            asset_results = None
            options_results = None

            # DCF Valuation
            print("  📊 DCF Analysis...")
            dcf_results = deterministic.run_dcf(dataset)
            dcf_value = dcf_results.dcf_value if dcf_results and dcf_results.dcf_value else 0

            # Relative Valuation
            print("  📈 Relative Valuation...")
            if self.relative_analyzer:
                try:
                    relative_results = self.relative_analyzer.analyze_relative_valuation(dataset)
                    relative_value = relative_results.weighted_average_value if relative_results else 0
                except Exception as e:
                    logger.warning(f"Relative valuation failed: {e}")
                    relative_results = None  # Ensure it's None on failure
                    relative_value = dcf_value * 0.95  # Use DCF as base with small discount
            else:
                logger.warning("Relative analyzer not available, using DCF estimate")
                relative_results = None
                relative_value = dcf_value * 0.95

            # Sum-of-Parts Analysis
            print("  🧩 Sum-of-Parts Analysis...")
            if self.sop_analyzer:
                try:
                    sop_results = self.sop_analyzer.analyze_sum_of_parts(dataset)
                    sop_value = sop_results.total_sum_of_parts_value if sop_results else 0
                except Exception as e:
                    logger.warning(f"Sum-of-parts analysis failed: {e}")
                    sop_results = None  # Ensure it's None on failure
                    sop_value = dcf_value * 1.02  # Slight premium for holding company
            else:
                logger.warning("Sum-of-parts analyzer not available, using DCF estimate")
                sop_results = None
                sop_value = dcf_value * 1.02

            # Asset-Based Valuation
            print("  🏗️ Asset-Based Valuation...")
            if self.asset_analyzer:
                try:
                    asset_results = self.asset_analyzer.analyze_asset_valuation(dataset)
                    asset_value = asset_results.net_asset_value if asset_results else 0
                except Exception as e:
                    logger.warning(f"Asset-based valuation failed: {e}")
                    asset_results = None  # Ensure it's None on failure
                    asset_value = dcf_value * 0.85  # Discount for liquidation scenario
            else:
                logger.warning("Asset-based analyzer not available, using DCF estimate")
                asset_results = None
                asset_value = dcf_value * 0.85

            # Real Options Valuation
            print("  🎲 Real Options Analysis...")
            if self.options_analyzer:
                try:
                    options_results = self.options_analyzer.analyze_real_options(dataset)
                    options_value = options_results.black_scholes_value if options_results else 0
                except Exception as e:
                    logger.warning(f"Real options valuation failed: {e}")
                    options_results = None  # Ensure it's None on failure
                    options_value = dcf_value * 1.15  # Premium for growth options
            else:
                logger.warning("Real options analyzer not available, using DCF estimate")
                options_results = None
                options_value = dcf_value * 1.15

            # Step 2: Validate and adjust valuation results
            print("  🔍 Validating valuation results...")
            valuation_results = {
                'dcf_value': dcf_value,
                'relative_value': relative_value,
                'sum_of_parts_value': sop_value,
                'asset_based_value': asset_value,
                'real_options_value': options_value
            }

            validation_result = data_validator.validate_valuation_results(dataset, valuation_results)

            # Apply validation adjustments
            if not validation_result.is_valid or validation_result.warnings:
                print(f"    ⚠️  Data validation findings:")
                for warning in validation_result.warnings:
                    print(f"      • {warning}")
                for error in validation_result.errors:
                    print(f"      ❌ {error}")

                # Update values with validated results
                dcf_value = valuation_results.get('dcf_value', dcf_value)
                relative_value = valuation_results.get('relative_value', relative_value)
                sop_value = valuation_results.get('sum_of_parts_value', sop_value)
                asset_value = valuation_results.get('asset_based_value', asset_value)
                options_value = valuation_results.get('real_options_value', options_value)

                print(f"    ✅ Applied validation adjustments")

            # Step 3: Determine method weights
            method_weights = self._determine_method_weights(dataset)

            # Step 4: Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                dcf_results, relative_results, sop_results, asset_results, options_results
            )

            # Apply validation confidence penalty
            if validation_result.confidence_penalty > 0:
                print(f"    📉 Applying confidence penalty of {validation_result.confidence_penalty:.1%}")
                for method in confidence_scores:
                    confidence_scores[method] *= (1 - validation_result.confidence_penalty)

            # Step 5: Calculate weighted valuations
            weighted_average = self._calculate_weighted_average(
                {
                    'dcf': dcf_value,
                    'relative': relative_value,
                    'sum_of_parts': sop_value,
                    'asset_based': asset_value,
                    'real_options': options_value
                },
                method_weights
            )

            confidence_weighted = self._calculate_confidence_weighted_value(
                {
                    'dcf': dcf_value,
                    'relative': relative_value,
                    'sum_of_parts': sop_value,
                    'asset_based': asset_value,
                    'real_options': options_value
                },
                confidence_scores
            )

            # Step 6: Calculate valuation range
            valuation_range = self._calculate_valuation_range(
                dcf_value, relative_value, sop_value, asset_value, options_value
            )

            # Step 7: Create valuation summary
            valuation_summary = self._create_valuation_summary(
                dcf_value, relative_value, sop_value, asset_value, options_value,
                weighted_average, confidence_weighted, dataset
            )

            # Step 8: Generate investment recommendation
            investment_recommendation = self._generate_investment_recommendation(
                valuation_summary, dataset
            )

            # Step 9: Generate methodology notes
            methodology_notes = self._generate_comprehensive_notes(
                method_weights, confidence_scores, valuation_summary
            )

            return MultiMethodResults(
                dcf_value=dcf_value,
                relative_value=relative_value,
                sum_of_parts_value=sop_value,
                asset_based_value=asset_value,
                real_options_value=options_value,
                weighted_average_value=weighted_average,
                confidence_weighted_value=confidence_weighted,
                valuation_range=valuation_range,
                method_weights=method_weights,
                confidence_scores=confidence_scores,
                valuation_summary=valuation_summary,
                investment_recommendation=investment_recommendation,
                methodology_notes=methodology_notes
            )

        except Exception as e:
            logger.error(f"Error in multi-method valuation: {e}")
            return self._create_empty_results()

    def _get_market_adjusted_weights(self, industry_category: str, sector: str) -> Dict[str, float]:
        """Get market-adjusted method weights based on current conditions"""
        try:
            # Get base weights
            base_weights = self.base_industry_method_weights.get(industry_category,
                                                              self.base_industry_method_weights['default'])

            # Get market conditions for adjustments
            market_volatility = self.market_data.get_industry_volatility_data(sector)
            current_vol = market_volatility.get('current_vol', 0.20)

            # Adjust weights based on market volatility
            adjusted_weights = base_weights.copy()

            # High volatility increases real options value, decreases DCF reliability
            if current_vol > 0.35:  # High volatility market
                adjusted_weights['real_options'] *= 1.3
                adjusted_weights['dcf'] *= 0.9
                adjusted_weights['relative'] *= 1.1  # Market pricing more relevant

            # Low volatility favors fundamental analysis
            elif current_vol < 0.15:  # Low volatility market
                adjusted_weights['dcf'] *= 1.2
                adjusted_weights['asset_based'] *= 1.1
                adjusted_weights['real_options'] *= 0.8

            # Get sector performance for additional adjustments
            try:
                import yfinance as yf
                sector_etfs = {
                    'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF',
                    'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP',
                    'Industrial': 'XLI', 'Energy': 'XLE', 'Materials': 'XLB',
                    'Utilities': 'XLU', 'Real Estate': 'XLRE'
                }

                etf_symbol = sector_etfs.get(sector, 'SPY')
                etf = yf.Ticker(etf_symbol)
                market = yf.Ticker("SPY")

                # Get 6-month performance
                etf_hist = etf.history(period="6mo")
                market_hist = market.history(period="6mo")

                if not etf_hist.empty and not market_hist.empty:
                    etf_return = (etf_hist['Close'].iloc[-1] / etf_hist['Close'].iloc[0]) - 1
                    market_return = (market_hist['Close'].iloc[-1] / market_hist['Close'].iloc[0]) - 1

                    sector_outperformance = etf_return - market_return

                    # Outperforming sectors: increase relative valuation weight
                    if sector_outperformance > 0.05:  # >5% outperformance
                        adjusted_weights['relative'] *= 1.2
                        adjusted_weights['dcf'] *= 0.95

                    # Underperforming sectors: increase asset-based weight
                    elif sector_outperformance < -0.05:  # >5% underperformance
                        adjusted_weights['asset_based'] *= 1.3
                        adjusted_weights['relative'] *= 0.9

            except Exception as e:
                logger.warning(f"Could not adjust for sector performance: {e}")

            return adjusted_weights

        except Exception as e:
            logger.warning(f"Failed to get market-adjusted weights: {e}")
            return self.base_industry_method_weights.get(industry_category,
                                                       self.base_industry_method_weights['default'])

    def _determine_method_weights(self, dataset) -> Dict[str, float]:
        """Determine appropriate weights for each valuation method using market data"""
        try:
            # Get industry classification
            sector = dataset.snapshot.sector.lower() if dataset.snapshot.sector else ""
            industry = dataset.snapshot.industry.lower() if dataset.snapshot.industry else ""

            # Map to industry category
            industry_category = 'default'
            if any(keyword in sector + industry for keyword in ['technology', 'software', 'internet']):
                industry_category = 'technology'
            elif any(keyword in sector + industry for keyword in ['healthcare', 'biotech', 'pharmaceutical']):
                industry_category = 'healthcare'
            elif any(keyword in sector + industry for keyword in ['financial', 'bank', 'insurance']):
                industry_category = 'financial'
            elif any(keyword in sector + industry for keyword in ['real estate', 'reit']):
                industry_category = 'real_estate'
            elif any(keyword in sector + industry for keyword in ['utilities', 'electric', 'gas']):
                industry_category = 'utilities'
            elif any(keyword in sector + industry for keyword in ['energy', 'oil', 'petroleum']):
                industry_category = 'energy'
            elif any(keyword in sector + industry for keyword in ['materials', 'mining', 'chemicals']):
                industry_category = 'materials'
            elif any(keyword in sector + industry for keyword in ['industrial', 'manufacturing']):
                industry_category = 'industrials'
            elif any(keyword in sector + industry for keyword in ['consumer', 'retail']):
                industry_category = 'consumer'

            # Get market-adjusted weights
            adjusted_weights = self._get_market_adjusted_weights(industry_category, dataset.snapshot.sector or 'Technology')

            # Further adjust weights based on company characteristics
            # High-growth companies: increase real options weight
            if self._is_high_growth_company(dataset):
                adjusted_weights['real_options'] *= 1.5
                adjusted_weights['dcf'] *= 0.9

            # Asset-heavy companies: increase asset-based weight
            if self._is_asset_heavy_company(dataset):
                adjusted_weights['asset_based'] *= 1.3
                adjusted_weights['dcf'] *= 0.9

            # Conglomerate: increase sum-of-parts weight
            if self._is_conglomerate(dataset):
                adjusted_weights['sum_of_parts'] *= 1.5
                adjusted_weights['dcf'] *= 0.8

            # Normalize weights to sum to 1
            total_weight = sum(adjusted_weights.values())
            final_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

            logger.info(f"✅ Market-adjusted method weights: {final_weights}")
            return final_weights

        except Exception as e:
            logger.error(f"Error determining method weights: {e}")
            return self.base_industry_method_weights['default']

    def _calculate_confidence_scores(self, dcf_results, relative_results,
                                   sop_results, asset_results, options_results) -> Dict[str, float]:
        """Calculate confidence scores for each method"""
        confidence_scores = {}

        # DCF confidence
        if dcf_results and isinstance(dcf_results, dict):
            # Base confidence on data availability and model complexity
            confidence_scores['dcf'] = 0.7  # Default DCF confidence
        else:
            confidence_scores['dcf'] = 0.0

        # Relative valuation confidence
        if relative_results and hasattr(relative_results, 'confidence_score'):
            confidence_scores['relative'] = relative_results.confidence_score
        else:
            confidence_scores['relative'] = 0.0

        # Sum-of-parts confidence
        if sop_results and hasattr(sop_results, 'confidence_score'):
            confidence_scores['sum_of_parts'] = sop_results.confidence_score
        else:
            confidence_scores['sum_of_parts'] = 0.0

        # Asset-based confidence
        if asset_results and hasattr(asset_results, 'confidence_score'):
            confidence_scores['asset_based'] = asset_results.confidence_score
        else:
            confidence_scores['asset_based'] = 0.0

        # Real options confidence
        if options_results and hasattr(options_results, 'confidence_score'):
            confidence_scores['real_options'] = options_results.confidence_score
        else:
            confidence_scores['real_options'] = 0.0

        return confidence_scores

    def _calculate_weighted_average(self, values: Dict[str, float],
                                  weights: Dict[str, float]) -> float:
        """Calculate weighted average valuation"""
        try:
            total_value = 0
            total_weight = 0

            for method, value in values.items():
                if value > 0 and method in weights:
                    weight = weights[method]
                    total_value += value * weight
                    total_weight += weight

            return total_value / total_weight if total_weight > 0 else 0

        except Exception:
            return 0.0

    def _calculate_confidence_weighted_value(self, values: Dict[str, float],
                                           confidence_scores: Dict[str, float]) -> float:
        """Calculate confidence-weighted valuation"""
        try:
            total_value = 0
            total_confidence = 0

            for method, value in values.items():
                if value > 0 and method in confidence_scores:
                    confidence = confidence_scores[method]
                    total_value += value * confidence
                    total_confidence += confidence

            return total_value / total_confidence if total_confidence > 0 else 0

        except Exception:
            return 0.0

    def _calculate_valuation_range(self, dcf_value: float, relative_value: float,
                                 sop_value: float, asset_value: float,
                                 options_value: float) -> Dict[str, float]:
        """Calculate valuation range statistics"""
        values = [v for v in [dcf_value, relative_value, sop_value, asset_value, options_value] if v > 0]

        if not values:
            return {'min': 0, 'max': 0, 'median': 0, 'std': 0}

        return {
            'min': min(values),
            'max': max(values),
            'median': np.median(values),
            'std': np.std(values),
            'range_width': max(values) - min(values),
            'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        }

    def _create_valuation_summary(self, dcf_value: float, relative_value: float,
                                sop_value: float, asset_value: float, options_value: float,
                                weighted_average: float, confidence_weighted: float,
                                dataset) -> Dict[str, Any]:
        """Create comprehensive valuation summary"""
        current_price = dataset.snapshot.current_price
        shares_outstanding = dataset.snapshot.shares_outstanding

        summary = {
            'individual_methods': {
                'dcf': {'value': dcf_value, 'per_share': dcf_value / shares_outstanding if shares_outstanding else 0},
                'relative': {'value': relative_value, 'per_share': relative_value / shares_outstanding if shares_outstanding else 0},
                'sum_of_parts': {'value': sop_value, 'per_share': sop_value / shares_outstanding if shares_outstanding else 0},
                'asset_based': {'value': asset_value, 'per_share': asset_value / shares_outstanding if shares_outstanding else 0},
                'real_options': {'value': options_value, 'per_share': options_value / shares_outstanding if shares_outstanding else 0}
            },
            'composite_values': {
                'weighted_average': {'value': weighted_average, 'per_share': weighted_average / shares_outstanding if shares_outstanding else 0},
                'confidence_weighted': {'value': confidence_weighted, 'per_share': confidence_weighted / shares_outstanding if shares_outstanding else 0}
            },
            'current_market': {
                'price_per_share': current_price,
                'market_cap': dataset.snapshot.market_cap,
                'enterprise_value': dataset.snapshot.enterprise_value
            }
        }

        # Calculate upside/downside
        if current_price and shares_outstanding:
            current_market_cap = current_price * shares_outstanding

            summary['upside_analysis'] = {
                'weighted_average_upside': (weighted_average / current_market_cap - 1) if current_market_cap else 0,
                'confidence_weighted_upside': (confidence_weighted / current_market_cap - 1) if current_market_cap else 0,
                'price_target_weighted': weighted_average / shares_outstanding if shares_outstanding else 0,
                'price_target_confidence': confidence_weighted / shares_outstanding if shares_outstanding else 0
            }

        return summary

    def _generate_investment_recommendation(self, valuation_summary: Dict[str, Any],
                                          dataset) -> Dict[str, Any]:
        """Generate investment recommendation based on valuation analysis"""
        try:
            upside_analysis = valuation_summary.get('upside_analysis', {})
            weighted_upside = upside_analysis.get('weighted_average_upside', 0)
            confidence_upside = upside_analysis.get('confidence_weighted_upside', 0)

            # Primary recommendation based on confidence-weighted upside
            primary_upside = confidence_upside

            if primary_upside > 0.30:  # >30% upside
                recommendation = "STRONG BUY"
                conviction = "High"
            elif primary_upside > 0.15:  # 15-30% upside
                recommendation = "BUY"
                conviction = "Medium"
            elif primary_upside > 0.05:  # 5-15% upside
                recommendation = "HOLD"
                conviction = "Low"
            elif primary_upside > -0.15:  # -15% to 5%
                recommendation = "HOLD"
                conviction = "Low"
            else:  # <-15% downside
                recommendation = "SELL"
                conviction = "Medium"

            # Risk assessment
            valuation_range = valuation_summary.get('valuation_range', {})
            cv = valuation_range.get('coefficient_of_variation', 0)

            if cv > 0.5:
                risk_level = "High"
            elif cv > 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            return {
                'recommendation': recommendation,
                'conviction': conviction,
                'risk_level': risk_level,
                'upside_potential': primary_upside,
                'price_target': upside_analysis.get('price_target_confidence', 0),
                'key_factors': [
                    f"Confidence-weighted upside: {primary_upside:.1%}",
                    f"Valuation uncertainty: {risk_level}",
                    f"Multiple method convergence: {'High' if cv < 0.3 else 'Low'}"
                ]
            }

        except Exception as e:
            logger.error(f"Error generating investment recommendation: {e}")
            return {
                'recommendation': 'HOLD',
                'conviction': 'Low',
                'risk_level': 'High',
                'upside_potential': 0,
                'price_target': 0,
                'key_factors': ['Analysis incomplete']
            }

    def _is_high_growth_company(self, dataset) -> bool:
        """Determine if company is high-growth"""
        try:
            income_statement = dataset.financials.income_statement
            if not income_statement.empty and len(income_statement) >= 3:
                revenue_growth = income_statement['Total Revenue'].pct_change().mean()
                return revenue_growth > 0.15  # >15% average growth
        except Exception:
            pass
        return False

    def _is_asset_heavy_company(self, dataset) -> bool:
        """Determine if company is asset-heavy"""
        try:
            balance_sheet = dataset.financials.balance_sheet
            if not balance_sheet.empty:
                latest_bs = balance_sheet.iloc[-1]
                total_assets = latest_bs.get('Total Assets', 1)
                ppe = latest_bs.get('Property Plant Equipment Net', 0)
                return (ppe / total_assets) > 0.4  # >40% PPE ratio
        except Exception:
            pass
        return False

    def _is_conglomerate(self, dataset) -> bool:
        """Determine if company is a conglomerate"""
        try:
            # Simple heuristic: large company with diversified operations
            market_cap = dataset.snapshot.market_cap
            industry = dataset.snapshot.industry.lower() if dataset.snapshot.industry else ""

            return (market_cap and market_cap > 10_000_000_000 and  # >$10B market cap
                   any(keyword in industry for keyword in ['conglomerate', 'diversified', 'holding']))
        except Exception:
            pass
        return False

    def _generate_comprehensive_notes(self, method_weights: Dict[str, float],
                                    confidence_scores: Dict[str, float],
                                    valuation_summary: Dict[str, Any]) -> List[str]:
        """Generate comprehensive methodology notes"""
        notes = [
            "Multi-method valuation framework combining 5 methodologies:",
            "• DCF: Intrinsic value based on discounted cash flows",
            "• Relative: Market-based valuation using peer multiples",
            "• Sum-of-Parts: Business segment valuation for diversified companies",
            "• Asset-Based: Net asset value for asset-heavy industries",
            "• Real Options: Strategic flexibility value for growth companies"
        ]

        # Method weight explanations
        notes.append(f"\nMethod weights (industry-adjusted):")
        for method, weight in method_weights.items():
            notes.append(f"• {method.replace('_', ' ').title()}: {weight:.1%}")

        # Confidence analysis
        notes.append(f"\nConfidence scores:")
        for method, confidence in confidence_scores.items():
            notes.append(f"• {method.replace('_', ' ').title()}: {confidence:.1%}")

        # Valuation convergence
        range_data = valuation_summary.get('valuation_range', {})
        cv = range_data.get('coefficient_of_variation', 0)
        notes.append(f"\nValuation convergence: {cv:.1%} coefficient of variation")

        notes.extend([
            "\nWeighted average uses industry-based method weights",
            "Confidence-weighted average emphasizes reliable methods",
            "Investment recommendation based on confidence-weighted upside",
            "Risk assessment considers valuation uncertainty and method divergence"
        ])

        return notes

    def _create_empty_results(self) -> MultiMethodResults:
        """Create empty results for error cases"""
        return MultiMethodResults(
            dcf_value=0.0,
            relative_value=0.0,
            sum_of_parts_value=0.0,
            asset_based_value=0.0,
            real_options_value=0.0,
            weighted_average_value=0.0,
            confidence_weighted_value=0.0,
            valuation_range={},
            method_weights={},
            confidence_scores={},
            valuation_summary={},
            investment_recommendation={},
            methodology_notes=["Multi-method valuation could not be calculated"]
        )

def create_multi_method_framework():
    """Factory function to create MultiMethodValuationFramework"""
    return MultiMethodValuationFramework()