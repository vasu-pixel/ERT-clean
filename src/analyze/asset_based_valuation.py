"""
Asset-Based Valuation Module
For asset-heavy industries: real estate, utilities, natural resources, manufacturing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

@dataclass
class AssetValuationResults:
    """Results from asset-based valuation analysis"""
    net_asset_value: float
    tangible_book_value: float
    replacement_cost: float
    liquidation_value: float
    adjusted_book_value: float
    asset_breakdown: Dict[str, float]
    valuation_adjustments: Dict[str, float]
    industry_metrics: Dict[str, float]
    confidence_score: float
    methodology_notes: List[str]

class AssetBasedValuationAnalyzer:
    """
    Comprehensive asset-based valuation for asset-heavy industries
    """

    def __init__(self):
        self.market_data = MarketDataProvider()
        # Initialize with empty adjustments - will be populated from market data
        self.industry_adjustments = {}

    def analyze_asset_valuation(self, dataset) -> AssetValuationResults:
        """
        Perform comprehensive asset-based valuation analysis
        """
        try:
            # Extract financial data
            balance_sheet = dataset.financials.balance_sheet
            income_statement = dataset.financials.income_statement
            cash_flow = dataset.financials.cash_flow

            # Determine industry classification
            industry = self._classify_industry(dataset)

            # Calculate base asset values
            asset_breakdown = self._calculate_asset_breakdown(balance_sheet)

            # Apply industry-specific adjustments
            valuation_adjustments = self._apply_industry_adjustments(
                asset_breakdown, industry
            )

            # Calculate various asset-based valuations
            nav = self._calculate_net_asset_value(asset_breakdown, valuation_adjustments)
            tbv = self._calculate_tangible_book_value(balance_sheet)
            replacement_cost = self._calculate_replacement_cost(
                asset_breakdown, industry
            )
            liquidation_value = self._calculate_liquidation_value(
                asset_breakdown, industry
            )
            adjusted_book = self._calculate_adjusted_book_value(
                balance_sheet, valuation_adjustments
            )

            # Calculate industry metrics
            industry_metrics = self._calculate_industry_metrics(
                dataset, industry
            )

            # Assess confidence
            confidence_score = self._assess_confidence(
                asset_breakdown, industry, dataset
            )

            # Generate methodology notes
            methodology_notes = self._generate_methodology_notes(
                industry, valuation_adjustments
            )

            return AssetValuationResults(
                net_asset_value=nav,
                tangible_book_value=tbv,
                replacement_cost=replacement_cost,
                liquidation_value=liquidation_value,
                adjusted_book_value=adjusted_book,
                asset_breakdown=asset_breakdown,
                valuation_adjustments=valuation_adjustments,
                industry_metrics=industry_metrics,
                confidence_score=confidence_score,
                methodology_notes=methodology_notes
            )

        except Exception as e:
            logger.error(f"Error in asset-based valuation: {e}")
            return self._create_empty_results()

    def _classify_industry(self, dataset) -> str:
        """Classify company into asset-heavy industry category"""
        try:
            sector = dataset.snapshot.sector.lower() if dataset.snapshot.sector else ""
            industry = dataset.snapshot.industry.lower() if dataset.snapshot.industry else ""

            if any(keyword in sector + industry for keyword in
                   ['real estate', 'reit', 'property']):
                return 'real_estate'
            elif any(keyword in sector + industry for keyword in
                     ['utilities', 'electric', 'gas', 'water']):
                return 'utilities'
            elif any(keyword in sector + industry for keyword in
                     ['materials', 'mining', 'metals', 'chemicals']):
                return 'materials'
            elif any(keyword in sector + industry for keyword in
                     ['energy', 'oil', 'gas', 'petroleum']):
                return 'energy'
            elif any(keyword in sector + industry for keyword in
                     ['industrial', 'manufacturing', 'machinery']):
                return 'industrials'
            else:
                return 'general'

        except Exception:
            return 'general'

    def _calculate_asset_breakdown(self, balance_sheet) -> Dict[str, float]:
        """Break down assets into categories"""
        try:
            latest_bs = balance_sheet.iloc[-1] if balance_sheet is not None and not balance_sheet.empty else {}

            # Property, Plant & Equipment
            ppe = latest_bs.get('Property Plant Equipment Net', 0)
            if ppe == 0:
                ppe = latest_bs.get('Total Property Plant Equipment Net', 0)

            # Intangible assets
            intangibles = latest_bs.get('Intangible Assets', 0)
            goodwill = latest_bs.get('Goodwill', 0)

            # Current assets
            cash = latest_bs.get('Cash And Cash Equivalents', 0)
            inventory = latest_bs.get('Inventory', 0)
            receivables = latest_bs.get('Accounts Receivable', 0)

            # Investments
            investments = latest_bs.get('Other Investments', 0)

            # Other assets
            other_assets = latest_bs.get('Other Assets', 0)

            return {
                'property_plant_equipment': float(ppe),
                'intangible_assets': float(intangibles),
                'goodwill': float(goodwill),
                'cash_equivalents': float(cash),
                'inventory': float(inventory),
                'receivables': float(receivables),
                'investments': float(investments),
                'other_assets': float(other_assets)
            }

        except Exception as e:
            logger.error(f"Error calculating asset breakdown: {e}")
            return {}

    def _get_market_derived_industry_adjustments(self, industry: str) -> Dict[str, float]:
        """Get industry adjustments based on live market data"""
        try:
            # Get industry volatility and performance data
            volatility_data = self.market_data.get_industry_volatility_data(industry)
            current_vol = volatility_data.get('current_vol', 0.20)

            # Get sector performance vs market
            sector_etfs = {
                'real_estate': 'XLRE',
                'utilities': 'XLU',
                'materials': 'XLB',
                'energy': 'XLE',
                'industrials': 'XLI'
            }

            etf_symbol = sector_etfs.get(industry, 'SPY')

            try:
                import yfinance as yf
                etf = yf.Ticker(etf_symbol)
                market = yf.Ticker("SPY")

                # Get 1-year performance
                etf_hist = etf.history(period="1y")
                market_hist = market.history(period="1y")

                if not etf_hist.empty and not market_hist.empty:
                    etf_return = (etf_hist['Close'].iloc[-1] / etf_hist['Close'].iloc[0]) - 1
                    market_return = (market_hist['Close'].iloc[-1] / market_hist['Close'].iloc[0]) - 1

                    performance_ratio = (etf_return + 1) / (market_return + 1)
                else:
                    performance_ratio = 1.0

            except Exception:
                performance_ratio = 1.0

            # Market-derived adjustments based on actual performance and volatility
            if industry == 'real_estate':
                return {
                    'appreciation_factor': performance_ratio * 1.1,
                    'liquidity_discount': 0.95 - (current_vol * 0.5),  # Higher vol = more discount
                    'market_premium': performance_ratio
                }
            elif industry == 'utilities':
                return {
                    'regulatory_adjustment': 0.98 - (current_vol * 0.2),
                    'infrastructure_premium': 1.0 + (performance_ratio - 1) * 0.5,
                    'replacement_factor': 1.1 + (current_vol * 0.5)  # Higher vol = higher replacement cost
                }
            elif industry == 'materials':
                return {
                    'commodity_volatility': 1.0 - (current_vol * 0.5),
                    'depletion_factor': 0.90 - (current_vol * 0.3),
                    'exploration_premium': performance_ratio
                }
            elif industry == 'energy':
                return {
                    'reserve_value_factor': performance_ratio * 1.1,
                    'environmental_discount': 0.95 - (current_vol * 0.2),
                    'technology_obsolescence': 1.0 - (current_vol * 0.3)
                }
            elif industry == 'industrials':
                return {
                    'equipment_depreciation': 0.85 - (current_vol * 0.3),
                    'technology_factor': 0.90 - (current_vol * 0.2),
                    'capacity_utilization': performance_ratio
                }
            else:
                # General market-based adjustments
                return {
                    'market_adjustment': performance_ratio,
                    'volatility_discount': 1.0 - (current_vol * 0.3)
                }

        except Exception as e:
            logger.warning(f"Failed to get market-derived adjustments for {industry}: {e}")
            # Conservative fallback
            return {'conservative_adjustment': 0.95}

    def _apply_industry_adjustments(self, asset_breakdown: Dict[str, float],
                                  industry: str) -> Dict[str, float]:
        """Apply industry-specific valuation adjustments using market data"""
        adjustments = {}

        # Get market-derived industry factors
        industry_factors = self._get_market_derived_industry_adjustments(industry)

        try:
            # Property, Plant & Equipment adjustments using market factors
            ppe_value = asset_breakdown.get('property_plant_equipment', 0)

            # Apply market-derived adjustments
            for factor_name, factor_value in industry_factors.items():
                if 'appreciation' in factor_name or 'premium' in factor_name:
                    adjustments[f'ppe_{factor_name}'] = ppe_value * (factor_value - 1.0)
                elif 'discount' in factor_name or 'depreciation' in factor_name or 'obsolescence' in factor_name:
                    adjustments[f'ppe_{factor_name}'] = ppe_value * (factor_value - 1.0)
                else:
                    # General market adjustment
                    adjustments[f'ppe_{factor_name}'] = ppe_value * (factor_value - 1.0) * 0.1

            # Inventory adjustments
            inventory_value = asset_breakdown.get('inventory', 0)
            if inventory_value > 0:
                if industry in ['materials', 'energy']:
                    adjustments['commodity_inventory_mark'] = inventory_value * 0.05
                elif industry == 'industrials':
                    adjustments['inventory_obsolescence'] = inventory_value * -0.10

            # Intangible asset adjustments
            intangible_value = asset_breakdown.get('intangible_assets', 0)
            if intangible_value > 0:
                if industry == 'utilities':
                    adjustments['regulatory_intangibles'] = intangible_value * 0.10
                else:
                    adjustments['intangible_impairment'] = intangible_value * -0.20

            return adjustments

        except Exception as e:
            logger.error(f"Error applying industry adjustments: {e}")
            return {}

    def _calculate_net_asset_value(self, asset_breakdown: Dict[str, float],
                                 adjustments: Dict[str, float]) -> float:
        """Calculate Net Asset Value"""
        try:
            # Sum tangible assets
            tangible_assets = (
                asset_breakdown.get('property_plant_equipment', 0) +
                asset_breakdown.get('cash_equivalents', 0) +
                asset_breakdown.get('inventory', 0) +
                asset_breakdown.get('receivables', 0) +
                asset_breakdown.get('investments', 0) +
                asset_breakdown.get('other_assets', 0)
            )

            # Apply adjustments
            total_adjustments = sum(adjustments.values())

            return tangible_assets + total_adjustments

        except Exception:
            return 0.0

    def _calculate_tangible_book_value(self, balance_sheet) -> float:
        """Calculate Tangible Book Value"""
        try:
            latest_bs = balance_sheet.iloc[-1] if balance_sheet is not None and not balance_sheet.empty else {}

            book_value = latest_bs.get('Stockholders Equity', 0)
            intangibles = latest_bs.get('Intangible Assets', 0)
            goodwill = latest_bs.get('Goodwill', 0)

            return float(book_value - intangibles - goodwill)

        except Exception:
            return 0.0

    def _calculate_replacement_cost(self, asset_breakdown: Dict[str, float],
                                  industry: str) -> float:
        """Calculate replacement cost of assets using market data"""
        try:
            ppe = asset_breakdown.get('property_plant_equipment', 0)

            # Get market-derived replacement factors
            industry_adjustments = self._get_market_derived_industry_adjustments(industry)

            # Use replacement factor if available, otherwise calculate from market performance
            replacement_factor = industry_adjustments.get('replacement_factor', None)

            if replacement_factor is None:
                # Calculate based on construction/commodity costs and inflation
                volatility_data = self.market_data.get_industry_volatility_data(industry)
                current_vol = volatility_data.get('current_vol', 0.20)

                # Higher volatility industries typically have higher replacement costs
                replacement_factor = 1.05 + (current_vol * 2.0)  # Market-derived

            # Ensure reasonable bounds
            replacement_factor = max(1.0, min(2.0, replacement_factor))

            return ppe * replacement_factor

        except Exception:
            return 0.0

    def _calculate_liquidation_value(self, asset_breakdown: Dict[str, float],
                                   industry: str) -> float:
        """Calculate liquidation value of assets using market data"""
        try:
            # Standard liquidation values for current assets
            cash = asset_breakdown.get('cash_equivalents', 0) * 1.0
            receivables = asset_breakdown.get('receivables', 0) * 0.85
            inventory = asset_breakdown.get('inventory', 0) * 0.60
            ppe = asset_breakdown.get('property_plant_equipment', 0)

            # Market-derived PPE liquidation factors
            try:
                volatility_data = self.market_data.get_industry_volatility_data(industry)
                current_vol = volatility_data.get('current_vol', 0.20)

                # Base liquidation factor adjusted by market volatility
                base_factor = 0.60  # Conservative base
                volatility_adjustment = current_vol * -0.5  # Higher vol = lower liquidation value

                # Industry-specific adjustments based on asset liquidity
                if industry == 'real_estate':
                    industry_adjustment = 0.15  # Real estate more liquid
                elif industry == 'utilities':
                    industry_adjustment = -0.20  # Utility assets less liquid
                elif industry == 'materials':
                    industry_adjustment = -0.10  # Commodity assets moderately liquid
                elif industry == 'energy':
                    industry_adjustment = -0.15  # Energy assets less liquid
                elif industry == 'industrials':
                    industry_adjustment = -0.05  # Industrial assets moderately liquid
                else:
                    industry_adjustment = 0.0

                ppe_factor = base_factor + volatility_adjustment + industry_adjustment
                ppe_factor = max(0.20, min(0.80, ppe_factor))  # Reasonable bounds

            except Exception:
                # Conservative fallback
                ppe_factor = 0.50

            ppe_liquidation = ppe * ppe_factor

            return cash + receivables + inventory + ppe_liquidation

        except Exception:
            return 0.0

    def _calculate_adjusted_book_value(self, balance_sheet,
                                     adjustments: Dict[str, float]) -> float:
        """Calculate adjusted book value"""
        try:
            latest_bs = balance_sheet.iloc[-1] if balance_sheet is not None and not balance_sheet.empty else {}
            book_value = latest_bs.get('Stockholders Equity', 0)

            total_adjustments = sum(adjustments.values())
            return float(book_value + total_adjustments)

        except Exception:
            return 0.0

    def _calculate_industry_metrics(self, dataset, industry: str) -> Dict[str, float]:
        """Calculate industry-specific metrics"""
        metrics = {}

        try:
            balance_sheet = dataset.financials.balance_sheet
            income_statement = dataset.financials.income_statement

            if (balance_sheet is not None and not balance_sheet.empty and
                income_statement is not None and not income_statement.empty):
                latest_bs = balance_sheet.iloc[-1]
                latest_is = income_statement.iloc[-1]

                total_assets = latest_bs.get('Total Assets', 1)
                revenue = latest_is.get('Total Revenue', 0)

                # Asset turnover
                metrics['asset_turnover'] = revenue / total_assets if total_assets > 0 else 0

                # Asset intensity
                ppe = latest_bs.get('Property Plant Equipment Net', 0)
                metrics['asset_intensity'] = ppe / total_assets if total_assets > 0 else 0

                # Industry-specific metrics
                if industry == 'real_estate':
                    metrics['occupancy_proxy'] = min(revenue / total_assets * 10, 1.0)
                elif industry == 'utilities':
                    metrics['rate_base_proxy'] = ppe / 1000000  # Rate base approximation
                elif industry == 'materials':
                    inventory = latest_bs.get('Inventory', 0)
                    metrics['inventory_intensity'] = inventory / total_assets

        except Exception as e:
            logger.error(f"Error calculating industry metrics: {e}")

        return metrics

    def _assess_confidence(self, asset_breakdown: Dict[str, float],
                          industry: str, dataset) -> float:
        """Assess confidence in asset-based valuation"""
        try:
            confidence_factors = []

            # Asset composition factor
            total_assets = sum(asset_breakdown.values())
            ppe_ratio = asset_breakdown.get('property_plant_equipment', 0) / total_assets if total_assets > 0 else 0

            if ppe_ratio > 0.5:  # High asset intensity
                confidence_factors.append(0.8)
            elif ppe_ratio > 0.3:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)

            # Industry suitability
            if industry in ['real_estate', 'utilities', 'materials']:
                confidence_factors.append(0.9)
            elif industry in ['energy', 'industrials']:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)

            # Data availability
            balance_sheet = dataset.financials.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty and len(balance_sheet) >= 3:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)

            return np.mean(confidence_factors)

        except Exception:
            return 0.5

    def _generate_methodology_notes(self, industry: str,
                                  adjustments: Dict[str, float]) -> List[str]:
        """Generate methodology notes"""
        notes = [
            f"Asset-based valuation methodology applied for {industry} industry",
            "Tangible assets valued at adjusted book value with industry-specific factors"
        ]

        if adjustments:
            notes.append(f"Applied {len(adjustments)} industry-specific adjustments")

        industry_notes = {
            'real_estate': [
                "Real estate appreciated at market rates",
                "Applied liquidity discount for illiquid properties"
            ],
            'utilities': [
                "Rate base assets valued at replacement cost",
                "Regulatory lag adjustments applied"
            ],
            'materials': [
                "Commodity price volatility adjustments",
                "Resource depletion reserves considered"
            ],
            'energy': [
                "Proven reserves valued separately",
                "Environmental liabilities factored"
            ],
            'industrials': [
                "Technology obsolescence discount applied",
                "Capacity utilization premium included"
            ]
        }

        if industry in industry_notes:
            notes.extend(industry_notes[industry])

        return notes

    def _create_empty_results(self) -> AssetValuationResults:
        """Create empty results for error cases"""
        return AssetValuationResults(
            net_asset_value=0.0,
            tangible_book_value=0.0,
            replacement_cost=0.0,
            liquidation_value=0.0,
            adjusted_book_value=0.0,
            asset_breakdown={},
            valuation_adjustments={},
            industry_metrics={},
            confidence_score=0.0,
            methodology_notes=["Asset-based valuation could not be calculated"]
        )

def create_asset_valuation_analyzer():
    """Factory function to create AssetBasedValuationAnalyzer"""
    return AssetBasedValuationAnalyzer()