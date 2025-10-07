"""
Sum-of-Parts Analysis for Conglomerates
- Business segment valuation using different methodologies
- Holding company discount/premium analysis
- Asset allocation and capital structure overlay
- Synergy and dis-synergy adjustments
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from src.data_pipeline.models import CompanyDataset
from src.analyze.relative_valuation import RelativeValuationAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class BusinessSegment:
    """Individual business segment data"""
    segment_name: str
    segment_type: str  # 'operating', 'investment', 'discontinued'

    # Financial metrics
    revenue: float = 0.0
    operating_income: float = 0.0
    ebitda: float = 0.0
    assets: float = 0.0
    invested_capital: float = 0.0

    # Growth and margins
    revenue_growth: float = 0.0
    operating_margin: float = 0.0
    roic: float = 0.0

    # Valuation inputs
    segment_multiple: float = 0.0
    segment_wacc: float = 0.0
    comparable_companies: List[str] = field(default_factory=list)

    # Valuation outputs
    dcf_value: float = 0.0
    multiple_value: float = 0.0
    asset_value: float = 0.0
    recommended_value: float = 0.0

@dataclass
class SumOfPartsResults:
    """Sum-of-parts valuation results"""

    target_ticker: str
    analysis_date: str

    # Segment analysis
    operating_segments: List[BusinessSegment] = field(default_factory=list)
    investment_segments: List[BusinessSegment] = field(default_factory=list)
    discontinued_segments: List[BusinessSegment] = field(default_factory=list)

    # Corporate level adjustments
    corporate_costs: float = 0.0
    net_debt: float = 0.0
    excess_cash: float = 0.0
    pension_obligations: float = 0.0
    other_liabilities: float = 0.0

    # Valuation summary
    gross_sum_of_parts: float = 0.0
    net_sum_of_parts: float = 0.0
    per_share_value: float = 0.0

    # Holding company analysis
    holding_company_discount: float = 0.0
    complexity_discount: float = 0.0
    synergy_premium: float = 0.0

    # Final valuation
    adjusted_sum_of_parts: float = 0.0
    confidence_level: str = 'Medium'

class SumOfPartsAnalyzer:
    """Advanced sum-of-parts analysis for conglomerates"""

    def __init__(self):
        self.relative_analyzer = RelativeValuationAnalyzer()
        self.industry_multiples = self._load_industry_multiples()
        self.holding_company_discounts = self._load_hc_discounts()

    def analyze_sum_of_parts(self,
                           target_dataset: CompanyDataset,
                           segment_data: Optional[Dict] = None,
                           config: Optional[Dict] = None) -> SumOfPartsResults:
        """Perform comprehensive sum-of-parts analysis"""

        config = config or {}
        ticker = target_dataset.ticker

        print(f"🏢 Running sum-of-parts analysis for {ticker}...")

        results = SumOfPartsResults(
            target_ticker=ticker,
            analysis_date=datetime.now().isoformat()
        )

        # Step 1: Extract or use provided segment data
        print(f"  📊 Extracting business segment data...")
        if segment_data:
            segments = self._parse_provided_segments(segment_data)
        else:
            segments = self._extract_segments_from_dataset(target_dataset)

        # Step 2: Classify segments
        print(f"  🏷️ Classifying business segments...")
        results.operating_segments, results.investment_segments, results.discontinued_segments = \
            self._classify_segments(segments)

        print(f"     Operating segments: {len(results.operating_segments)}")
        print(f"     Investment segments: {len(results.investment_segments)}")
        print(f"     Discontinued segments: {len(results.discontinued_segments)}")

        # Step 3: Value each operating segment
        print(f"  💰 Valuing operating segments...")
        for segment in results.operating_segments:
            self._value_operating_segment(segment, target_dataset, config)

        # Step 4: Value investment segments
        print(f"  📈 Valuing investment segments...")
        for segment in results.investment_segments:
            self._value_investment_segment(segment, config)

        # Step 5: Corporate level adjustments
        print(f"  🏛️ Calculating corporate adjustments...")
        results = self._calculate_corporate_adjustments(results, target_dataset)

        # Step 6: Calculate gross sum-of-parts
        print(f"  🧮 Calculating gross sum-of-parts...")
        results.gross_sum_of_parts = self._calculate_gross_sum_of_parts(results)

        # Step 7: Apply holding company adjustments
        print(f"  ⚖️ Applying holding company adjustments...")
        results = self._apply_holding_company_adjustments(results, target_dataset)

        # Step 8: Calculate final valuation
        print(f"  🎯 Finalizing sum-of-parts valuation...")
        results = self._finalize_valuation(results, target_dataset)

        print(f"✅ Sum-of-parts analysis complete!")
        self._print_sum_of_parts_summary(results)

        return results

    def _extract_segments_from_dataset(self, dataset: CompanyDataset) -> List[Dict]:
        """Extract segment data from company dataset"""

        # Try to get segment data from supplemental information
        segment_info = dataset.supplemental.get('segments', {})

        if not segment_info:
            # Create default segments based on company fundamentals
            fundamentals = dataset.financials.fundamentals
            sector = dataset.snapshot.sector

            # Create a single main operating segment
            main_segment = {
                'segment_name': f'{dataset.ticker} Operations',
                'segment_type': 'operating',
                'revenue': fundamentals.get('totalRevenue', 0),
                'operating_income': fundamentals.get('operatingIncome', 0),
                'ebitda': fundamentals.get('ebitda', fundamentals.get('operatingIncome', 0) * 1.2),
                'assets': fundamentals.get('totalAssets', 0) * 0.8,  # Assume 80% operating assets
                'revenue_growth': fundamentals.get('revenueGrowth', 0.05),
                'operating_margin': fundamentals.get('operatingMargins', 0.15)
            }

            return [main_segment]

        return self._parse_segment_data(segment_info)

    def _parse_segment_data(self, segment_info: Dict) -> List[Dict]:
        """Parse segment information into standardized format"""

        segments = []

        # Parse business segments
        business_segments = segment_info.get('business_segments', [])
        for segment in business_segments:
            parsed_segment = {
                'segment_name': segment.get('segment_name', 'Unknown'),
                'segment_type': 'operating',
                'revenue': segment.get('financials', {}).get('revenue', 0),
                'operating_income': segment.get('financials', {}).get('operating_income', 0),
                'ebitda': segment.get('financials', {}).get('ebitda', 0),
                'assets': segment.get('financials', {}).get('assets', 0),
                'revenue_growth': 0.05,  # Default assumption
                'operating_margin': 0.15  # Default assumption
            }
            segments.append(parsed_segment)

        # Parse geographic segments as additional context
        geographic_segments = segment_info.get('geographic_segments', [])

        return segments if segments else self._create_default_segments()

    def _create_default_segments(self) -> List[Dict]:
        """Create default segment structure for single-business companies"""

        return [{
            'segment_name': 'Core Operations',
            'segment_type': 'operating',
            'revenue': 100000000000,  # Will be overridden
            'operating_income': 15000000000,
            'ebitda': 20000000000,
            'assets': 80000000000,
            'revenue_growth': 0.05,
            'operating_margin': 0.15
        }]

    def _classify_segments(self, segments: List[Dict]) -> Tuple[List[BusinessSegment], List[BusinessSegment], List[BusinessSegment]]:
        """Classify segments into operating, investment, and discontinued"""

        operating_segments = []
        investment_segments = []
        discontinued_segments = []

        for segment_data in segments:
            segment = BusinessSegment(
                segment_name=segment_data.get('segment_name', 'Unknown'),
                segment_type=segment_data.get('segment_type', 'operating'),
                revenue=segment_data.get('revenue', 0),
                operating_income=segment_data.get('operating_income', 0),
                ebitda=segment_data.get('ebitda', 0),
                assets=segment_data.get('assets', 0),
                revenue_growth=segment_data.get('revenue_growth', 0.05),
                operating_margin=segment_data.get('operating_margin', 0.15)
            )

            # Calculate derived metrics
            if segment.revenue > 0:
                segment.operating_margin = segment.operating_income / segment.revenue
            if segment.assets > 0:
                segment.roic = segment.operating_income / segment.assets

            # Classify based on characteristics
            if segment.segment_type == 'investment' or 'investment' in segment.segment_name.lower():
                investment_segments.append(segment)
            elif segment.segment_type == 'discontinued' or segment.revenue <= 0:
                discontinued_segments.append(segment)
            else:
                operating_segments.append(segment)

        return operating_segments, investment_segments, discontinued_segments

    def _value_operating_segment(self, segment: BusinessSegment, target_dataset: CompanyDataset, config: Dict):
        """Value an operating segment using multiple approaches"""

        # Method 1: EV/EBITDA Multiple
        segment.multiple_value = self._value_segment_by_multiple(segment, target_dataset)

        # Method 2: DCF (simplified)
        segment.dcf_value = self._value_segment_by_dcf(segment, config)

        # Method 3: Asset-based (for asset-heavy segments)
        segment.asset_value = self._value_segment_by_assets(segment)

        # Recommended value (weighted average)
        if segment.ebitda > 0 and segment.multiple_value > 0:
            # Prefer multiple-based for profitable segments
            segment.recommended_value = (
                segment.multiple_value * 0.6 +
                segment.dcf_value * 0.3 +
                segment.asset_value * 0.1
            )
        elif segment.dcf_value > 0:
            # Fall back to DCF
            segment.recommended_value = segment.dcf_value
        else:
            # Asset-based as last resort
            segment.recommended_value = segment.asset_value

    def _value_segment_by_multiple(self, segment: BusinessSegment, target_dataset: CompanyDataset) -> float:
        """Value segment using industry multiples"""

        if segment.ebitda <= 0:
            return 0.0

        # Get industry multiple based on segment characteristics
        industry_multiple = self._get_segment_multiple(segment, target_dataset.snapshot.sector)

        return segment.ebitda * industry_multiple

    def _value_segment_by_dcf(self, segment: BusinessSegment, config: Dict) -> float:
        """Value segment using simplified DCF"""

        if segment.operating_income <= 0:
            return 0.0

        # Simplified 5-year DCF
        wacc = config.get('default_wacc', 0.08)
        terminal_growth = config.get('terminal_growth', 0.025)
        tax_rate = config.get('tax_rate', 0.25)

        # Project cash flows
        after_tax_income = segment.operating_income * (1 - tax_rate)
        terminal_value = after_tax_income * (1 + terminal_growth) / (wacc - terminal_growth)

        # Present value (simplified)
        pv_terminal = terminal_value / ((1 + wacc) ** 5)
        pv_operations = after_tax_income * 4  # Simplified 4 years of cash flows

        return pv_operations + pv_terminal

    def _value_segment_by_assets(self, segment: BusinessSegment) -> float:
        """Value segment based on asset value"""

        if segment.assets <= 0:
            return 0.0

        # Apply haircut based on asset quality and liquidity
        asset_multiple = 0.8  # 20% haircut for fire sale

        return segment.assets * asset_multiple

    def _value_investment_segment(self, segment: BusinessSegment, config: Dict):
        """Value investment/non-operating segments"""

        # For investment segments, use asset-based approach
        if 'real estate' in segment.segment_name.lower():
            # Real estate typically trades at book value or higher
            segment.recommended_value = segment.assets * 1.1
        elif 'investment' in segment.segment_name.lower():
            # Financial investments at market value
            segment.recommended_value = segment.assets * 0.95
        else:
            # Other investments with conservative haircut
            segment.recommended_value = segment.assets * 0.8

    def _get_segment_multiple(self, segment: BusinessSegment, parent_sector: str) -> float:
        """Get appropriate EV/EBITDA multiple for segment"""

        segment_name_lower = segment.segment_name.lower()

        # Segment-specific multiples
        if any(tech_word in segment_name_lower for tech_word in ['software', 'digital', 'cloud', 'tech']):
            return 15.0  # Technology multiple
        elif any(retail_word in segment_name_lower for retail_word in ['retail', 'store', 'consumer']):
            return 8.0   # Retail multiple
        elif any(industrial_word in segment_name_lower for industrial_word in ['manufacturing', 'industrial', 'equipment']):
            return 10.0  # Industrial multiple
        elif any(service_word in segment_name_lower for service_word in ['service', 'consulting', 'professional']):
            return 12.0  # Services multiple
        else:
            # Default to parent sector multiple
            return self._get_sector_default_multiple(parent_sector)

    def _get_sector_default_multiple(self, sector: str) -> float:
        """Get default multiple for sector"""

        sector_multiples = {
            'Technology': 15.0,
            'Healthcare': 12.0,
            'Financial Services': 8.0,
            'Consumer Cyclical': 9.0,
            'Consumer Defensive': 11.0,
            'Industrial': 10.0,
            'Energy': 6.0,
            'Utilities': 7.0,
            'Real Estate': 14.0,
            'Materials': 8.0
        }

        return sector_multiples.get(sector, 10.0)  # Default to 10x

    def _calculate_corporate_adjustments(self, results: SumOfPartsResults, dataset: CompanyDataset) -> SumOfPartsResults:
        """Calculate corporate-level adjustments"""

        fundamentals = dataset.financials.fundamentals

        # Corporate costs (unallocated overhead)
        total_operating_income = sum(seg.operating_income for seg in results.operating_segments)
        reported_operating_income = fundamentals.get('operatingIncome', 0)
        results.corporate_costs = max(0, total_operating_income - reported_operating_income)

        # Net debt
        total_debt = fundamentals.get('totalDebt', 0)
        cash_and_equivalents = fundamentals.get('totalCashFromOperatingActivities', 0) * 0.1  # Estimate
        results.net_debt = max(0, total_debt - cash_and_equivalents)

        # Excess cash (above operating requirements)
        results.excess_cash = max(0, cash_and_equivalents - dataset.snapshot.market_cap * 0.05)  # 5% of market cap

        # Other liabilities (pension, environmental, etc.)
        results.pension_obligations = fundamentals.get('netTangibleAssets', 0) * 0.02  # Estimate
        results.other_liabilities = 0  # Would require detailed analysis

        return results

    def _calculate_gross_sum_of_parts(self, results: SumOfPartsResults) -> float:
        """Calculate gross sum-of-parts value"""

        operating_value = sum(seg.recommended_value for seg in results.operating_segments)
        investment_value = sum(seg.recommended_value for seg in results.investment_segments)

        return operating_value + investment_value

    def _apply_holding_company_adjustments(self, results: SumOfPartsResults, dataset: CompanyDataset) -> SumOfPartsResults:
        """Apply holding company discount and other adjustments"""

        # Holding company discount
        num_segments = len(results.operating_segments)
        if num_segments >= 4:
            results.holding_company_discount = 0.15  # 15% discount for complex conglomerates
        elif num_segments >= 2:
            results.holding_company_discount = 0.10  # 10% discount for diversified companies
        else:
            results.holding_company_discount = 0.0   # No discount for focused companies

        # Complexity discount
        results.complexity_discount = min(0.05, num_segments * 0.01)  # 1% per segment, max 5%

        # Synergy premium (for well-integrated companies)
        if self._assess_synergy_potential(results, dataset):
            results.synergy_premium = 0.05  # 5% premium for synergistic businesses
        else:
            results.synergy_premium = 0.0

        return results

    def _assess_synergy_potential(self, results: SumOfPartsResults, dataset: CompanyDataset) -> bool:
        """Assess whether business segments have synergy potential"""

        # Simple heuristic: if segments are in related industries
        segment_names = [seg.segment_name.lower() for seg in results.operating_segments]

        # Look for related keywords
        tech_segments = sum(1 for name in segment_names if any(tech in name for tech in ['software', 'digital', 'tech']))
        consumer_segments = sum(1 for name in segment_names if any(consumer in name for consumer in ['retail', 'consumer', 'brand']))

        # Synergy potential if segments are in related areas
        return tech_segments >= 2 or consumer_segments >= 2

    def _finalize_valuation(self, results: SumOfPartsResults, dataset: CompanyDataset) -> SumOfPartsResults:
        """Finalize sum-of-parts valuation with all adjustments"""

        # Net sum-of-parts
        results.net_sum_of_parts = (
            results.gross_sum_of_parts
            - results.corporate_costs * 10  # Capitalize corporate costs at 10x
            - results.net_debt
            + results.excess_cash
            - results.pension_obligations
            - results.other_liabilities
        )

        # Apply holding company adjustments
        adjustment_factor = (
            1.0
            - results.holding_company_discount
            - results.complexity_discount
            + results.synergy_premium
        )

        results.adjusted_sum_of_parts = results.net_sum_of_parts * adjustment_factor

        # Per share value
        shares_outstanding = dataset.financials.fundamentals.get('sharesOutstanding', 1)
        results.per_share_value = results.adjusted_sum_of_parts / shares_outstanding

        # Confidence assessment
        results.confidence_level = self._assess_confidence_level(results)

        return results

    def _assess_confidence_level(self, results: SumOfPartsResults) -> str:
        """Assess confidence level in sum-of-parts valuation"""

        # High confidence: Few segments, good segment data
        if len(results.operating_segments) <= 2 and all(seg.revenue > 0 for seg in results.operating_segments):
            return 'High'

        # Medium confidence: Moderate complexity
        elif len(results.operating_segments) <= 4:
            return 'Medium'

        # Low confidence: High complexity or poor data
        else:
            return 'Low'

    def _load_industry_multiples(self) -> Dict[str, float]:
        """Load industry-specific multiples"""
        return {
            'software': 15.0,
            'hardware': 12.0,
            'retail': 8.0,
            'manufacturing': 10.0,
            'services': 12.0,
            'real_estate': 14.0,
            'financial': 8.0
        }

    def _load_hc_discounts(self) -> Dict[str, float]:
        """Load holding company discount data"""
        return {
            'conglomerate': 0.15,
            'diversified': 0.10,
            'focused': 0.05
        }

    def _print_sum_of_parts_summary(self, results: SumOfPartsResults):
        """Print comprehensive sum-of-parts summary"""

        print(f"\n🏢 Sum-of-Parts Analysis for {results.target_ticker}:")
        print(f"Analysis Date: {results.analysis_date[:10]}")

        print(f"\n📊 Business Segments:")
        for segment in results.operating_segments:
            print(f"  {segment.segment_name}: ${segment.recommended_value/1e9:.1f}B")
            print(f"    Revenue: ${segment.revenue/1e9:.1f}B | Operating Margin: {segment.operating_margin*100:.1f}%")

        print(f"\n💰 Valuation Summary:")
        print(f"  Gross Sum-of-Parts: ${results.gross_sum_of_parts/1e9:.1f}B")
        print(f"  Corporate Adjustments: ${(results.corporate_costs*10 + results.net_debt)/1e9:.1f}B")
        print(f"  Net Sum-of-Parts: ${results.net_sum_of_parts/1e9:.1f}B")

        print(f"\n⚖️ Holding Company Adjustments:")
        print(f"  HC Discount: {results.holding_company_discount*100:.1f}%")
        print(f"  Complexity Discount: {results.complexity_discount*100:.1f}%")
        print(f"  Synergy Premium: {results.synergy_premium*100:.1f}%")

        print(f"\n🎯 Final Valuation:")
        print(f"  Adjusted Sum-of-Parts: ${results.adjusted_sum_of_parts/1e9:.1f}B")
        print(f"  Per Share Value: ${results.per_share_value:.2f}")
        print(f"  Confidence Level: {results.confidence_level}")


def test_sum_of_parts():
    """Test sum-of-parts analyzer"""

    # Create mock dataset for a diversified company
    class MockSnapshot:
        def __init__(self):
            self.ticker = 'GE'  # Example conglomerate
            self.sector = 'Industrial'
            self.current_price = 100.0
            self.market_cap = 110000000000

    class MockFinancials:
        def __init__(self):
            self.fundamentals = {
                'totalRevenue': 75619000000,
                'operatingIncome': 5415000000,
                'totalAssets': 198887000000,
                'totalDebt': 29000000000,
                'sharesOutstanding': 1100000000
            }

    class MockDataset:
        def __init__(self):
            self.ticker = 'GE'
            self.snapshot = MockSnapshot()
            self.financials = MockFinancials()
            self.supplemental = {}

    dataset = MockDataset()

    # Mock segment data for testing
    segment_data = {
        'business_segments': [
            {
                'segment_name': 'Aviation',
                'financials': {
                    'revenue': 25000000000,
                    'operating_income': 4000000000,
                    'ebitda': 5000000000,
                    'assets': 50000000000
                }
            },
            {
                'segment_name': 'Healthcare',
                'financials': {
                    'revenue': 18000000000,
                    'operating_income': 2700000000,
                    'ebitda': 3200000000,
                    'assets': 35000000000
                }
            },
            {
                'segment_name': 'Power',
                'financials': {
                    'revenue': 15000000000,
                    'operating_income': 1200000000,
                    'ebitda': 1800000000,
                    'assets': 25000000000
                }
            }
        ]
    }

    print("="*60)
    print("TESTING SUM-OF-PARTS ANALYZER")
    print("="*60)

    analyzer = SumOfPartsAnalyzer()

    results = analyzer.analyze_sum_of_parts(
        target_dataset=dataset,
        segment_data=segment_data
    )

    # Compare to current market value
    current_market_value = dataset.snapshot.market_cap
    sop_premium = (results.adjusted_sum_of_parts / current_market_value - 1) * 100

    print(f"\n💼 Investment Implications:")
    print(f"Current Market Value: ${current_market_value/1e9:.1f}B")
    print(f"Sum-of-Parts Value: ${results.adjusted_sum_of_parts/1e9:.1f}B")
    print(f"Premium/(Discount): {sop_premium:+.1f}%")

if __name__ == "__main__":
    test_sum_of_parts()