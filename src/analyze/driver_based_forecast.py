"""
Driver-Based Forecasting Model
Replaces flat growth assumptions with fundamental business drivers:
- Volume × Price decomposition
- Customer count × ARPU models
- Margin bridge analysis
- Working capital drivers (DSO, DPO, inventory turns)
- CapEx as % of sales with depreciation schedules
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from src.data_pipeline.models import CompanyDataset

logger = logging.getLogger(__name__)

@dataclass
class BusinessDrivers:
    """Core business drivers for forecasting"""

    # Revenue Drivers
    unit_volume_growth: float = 0.05  # Annual unit growth
    price_realization: float = 0.02   # Annual price increases
    customer_growth: float = 0.08     # New customer acquisition
    arpu_growth: float = 0.03         # Average revenue per user growth
    market_share_change: float = 0.0  # Market share gain/loss

    # Margin Drivers
    gross_margin_expansion: float = 0.0   # Annual gross margin improvement
    sga_leverage: float = -0.5            # SG&A as % of revenue change
    rd_intensity_change: float = 0.0      # R&D % of revenue change

    # Working Capital Drivers
    dso_days: float = 45.0               # Days sales outstanding
    dpo_days: float = 30.0               # Days payable outstanding
    inventory_turns: float = 6.0         # Annual inventory turnover

    # Capital Allocation
    capex_as_pct_sales: float = 0.04     # CapEx as % of revenue
    depreciation_rate: float = 0.10      # Annual depreciation rate

    # Growth Investment
    customer_acquisition_cost: float = 50.0    # Cost per new customer
    rd_efficiency: float = 1.2                 # Revenue multiplier for R&D

@dataclass
class DriverBasedForecast:
    """Driver-based forecast output"""

    forecast_years: List[int] = field(default_factory=list)
    revenue_components: Dict[str, List[float]] = field(default_factory=dict)
    margin_components: Dict[str, List[float]] = field(default_factory=dict)
    working_capital: Dict[str, List[float]] = field(default_factory=dict)
    capex_schedule: Dict[str, List[float]] = field(default_factory=dict)
    free_cash_flow: List[float] = field(default_factory=list)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

class DriverBasedForecaster:
    """Advanced forecasting using business drivers"""

    def __init__(self):
        self.forecast_horizon = 5  # Years

    def build_driver_forecast(self,
                            dataset: CompanyDataset,
                            drivers: BusinessDrivers,
                            config: Optional[Dict] = None) -> DriverBasedForecast:
        """Build comprehensive driver-based forecast"""

        config = config or {}
        fundamentals = dataset.financials.fundamentals

        print(f"🚀 Building driver-based forecast for {dataset.ticker}...")

        # Initialize base year metrics
        base_metrics = self._extract_base_metrics(dataset)

        # Build forecast components
        forecast = DriverBasedForecast()
        forecast.forecast_years = list(range(
            datetime.now().year + 1,
            datetime.now().year + self.forecast_horizon + 1
        ))

        # Revenue forecast by drivers
        forecast.revenue_components = self._forecast_revenue_drivers(
            base_metrics, drivers, forecast.forecast_years
        )

        # Margin evolution
        forecast.margin_components = self._forecast_margin_drivers(
            base_metrics, drivers, forecast.forecast_years
        )

        # Working capital dynamics
        forecast.working_capital = self._forecast_working_capital(
            base_metrics, drivers, forecast.forecast_years,
            forecast.revenue_components
        )

        # Capital expenditure schedule
        forecast.capex_schedule = self._forecast_capex_schedule(
            base_metrics, drivers, forecast.forecast_years,
            forecast.revenue_components
        )

        # Free cash flow calculation
        forecast.free_cash_flow = self._calculate_free_cash_flow(
            forecast.revenue_components,
            forecast.margin_components,
            forecast.working_capital,
            forecast.capex_schedule
        )

        # Summary metrics and ratios
        forecast.summary_metrics = self._calculate_summary_metrics(
            forecast, base_metrics
        )

        print(f"✅ Driver forecast complete. {len(forecast.forecast_years)} year projection")
        return forecast

    def _extract_base_metrics(self, dataset: CompanyDataset) -> Dict[str, float]:
        """Extract base year metrics from dataset"""
        fundamentals = dataset.financials.fundamentals

        # Revenue metrics
        total_revenue = fundamentals.get('totalRevenue', 0)
        gross_profit = fundamentals.get('grossProfit', total_revenue * 0.3)

        # Operating metrics
        operating_income = fundamentals.get('operatingIncome', gross_profit * 0.5)
        rd_expense = fundamentals.get('researchDevelopment', total_revenue * 0.05)
        sga_expense = fundamentals.get('sellingGeneralAdministrative', total_revenue * 0.15)

        # Balance sheet items
        total_assets = fundamentals.get('totalAssets', total_revenue * 1.5)
        accounts_receivable = fundamentals.get('totalCurrentAssets', total_revenue * 0.1) * 0.3
        inventory = fundamentals.get('inventory', total_revenue * 0.08)
        accounts_payable = fundamentals.get('accountsPayable', total_revenue * 0.05)

        # Cash flow items
        operating_cash_flow = fundamentals.get('operatingCashflow', operating_income * 1.2)
        capex = abs(fundamentals.get('capitalExpenditures', total_revenue * 0.04))

        base_metrics = {
            # P&L Base
            'revenue': total_revenue,
            'gross_profit': gross_profit,
            'gross_margin': gross_profit / total_revenue if total_revenue else 0.3,
            'operating_income': operating_income,
            'operating_margin': operating_income / total_revenue if total_revenue else 0.15,
            'rd_expense': rd_expense,
            'rd_pct_revenue': rd_expense / total_revenue if total_revenue else 0.05,
            'sga_expense': sga_expense,
            'sga_pct_revenue': sga_expense / total_revenue if total_revenue else 0.15,

            # Working Capital Base
            'accounts_receivable': accounts_receivable,
            'inventory': inventory,
            'accounts_payable': accounts_payable,
            'dso': (accounts_receivable / total_revenue) * 365 if total_revenue else 45,
            'dpo': (accounts_payable / total_revenue) * 365 if total_revenue else 30,
            'inventory_turns': total_revenue / inventory if inventory else 6.0,

            # CapEx Base
            'capex': capex,
            'capex_pct_revenue': capex / total_revenue if total_revenue else 0.04,
            'total_assets': total_assets,

            # Cash Flow Base
            'operating_cash_flow': operating_cash_flow
        }

        return base_metrics

    def _forecast_revenue_drivers(self,
                                base_metrics: Dict[str, float],
                                drivers: BusinessDrivers,
                                forecast_years: List[int]) -> Dict[str, List[float]]:
        """Forecast revenue using volume x price and customer x ARPU drivers"""

        base_revenue = base_metrics['revenue']

        revenue_components = {
            'total_revenue': [],
            'volume_driven_revenue': [],
            'price_driven_revenue': [],
            'customer_driven_revenue': [],
            'arpu_component': [],
            'market_share_impact': []
        }

        # Estimate base metrics if not available
        estimated_customers = 1000000  # Could be extracted from filings
        estimated_arpu = base_revenue / estimated_customers if estimated_customers else 100
        estimated_units = estimated_customers * 2  # Assume 2 units per customer
        estimated_unit_price = base_revenue / estimated_units if estimated_units else 50

        current_revenue = base_revenue
        current_customers = estimated_customers
        current_arpu = estimated_arpu
        current_units = estimated_units
        current_unit_price = estimated_unit_price

        for i, year in enumerate(forecast_years):
            year_index = i + 1

            # Volume x Price approach
            current_units *= (1 + drivers.unit_volume_growth)
            current_unit_price *= (1 + drivers.price_realization)
            volume_price_revenue = current_units * current_unit_price

            # Customer x ARPU approach
            current_customers *= (1 + drivers.customer_growth)
            current_arpu *= (1 + drivers.arpu_growth)
            customer_arpu_revenue = current_customers * current_arpu

            # Market share impact
            market_share_revenue = current_revenue * (1 + drivers.market_share_change)

            # Blended approach (weight based on business model)
            # For product companies: 70% volume/price, 30% customer/ARPU
            # For service companies: 30% volume/price, 70% customer/ARPU

            # Assume balanced approach
            blended_revenue = (volume_price_revenue * 0.5) + (customer_arpu_revenue * 0.5)
            blended_revenue *= (1 + drivers.market_share_change)

            current_revenue = blended_revenue

            revenue_components['total_revenue'].append(blended_revenue)
            revenue_components['volume_driven_revenue'].append(volume_price_revenue)
            revenue_components['price_driven_revenue'].append(current_unit_price * estimated_units)
            revenue_components['customer_driven_revenue'].append(customer_arpu_revenue)
            revenue_components['arpu_component'].append(current_arpu)
            revenue_components['market_share_impact'].append(market_share_revenue - (volume_price_revenue * 0.5 + customer_arpu_revenue * 0.5))

        return revenue_components

    def _forecast_margin_drivers(self,
                               base_metrics: Dict[str, float],
                               drivers: BusinessDrivers,
                               forecast_years: List[int]) -> Dict[str, List[float]]:
        """Forecast margins with explicit drivers"""

        margin_components = {
            'gross_margin_pct': [],
            'gross_profit': [],
            'sga_expense': [],
            'sga_pct_revenue': [],
            'rd_expense': [],
            'rd_pct_revenue': [],
            'operating_income': [],
            'operating_margin_pct': []
        }

        base_gross_margin = base_metrics['gross_margin']
        base_sga_pct = base_metrics['sga_pct_revenue']
        base_rd_pct = base_metrics['rd_pct_revenue']

        for i, year in enumerate(forecast_years):
            year_index = i + 1

            # Assume we have revenue forecast
            revenue = base_metrics['revenue'] * ((1.06) ** year_index)  # Simplified

            # Gross margin evolution
            gross_margin_pct = base_gross_margin + (drivers.gross_margin_expansion * year_index)
            gross_profit = revenue * gross_margin_pct

            # SG&A leverage (decreases as % of revenue with scale)
            sga_pct_revenue = base_sga_pct + (drivers.sga_leverage * 0.01 * year_index)
            sga_pct_revenue = max(sga_pct_revenue, 0.05)  # Floor at 5%
            sga_expense = revenue * sga_pct_revenue

            # R&D intensity
            rd_pct_revenue = base_rd_pct + drivers.rd_intensity_change * year_index
            rd_pct_revenue = max(rd_pct_revenue, 0.0)  # Floor at 0%
            rd_expense = revenue * rd_pct_revenue

            # Operating income
            operating_income = gross_profit - sga_expense - rd_expense
            operating_margin_pct = operating_income / revenue if revenue else 0

            margin_components['gross_margin_pct'].append(gross_margin_pct)
            margin_components['gross_profit'].append(gross_profit)
            margin_components['sga_expense'].append(sga_expense)
            margin_components['sga_pct_revenue'].append(sga_pct_revenue)
            margin_components['rd_expense'].append(rd_expense)
            margin_components['rd_pct_revenue'].append(rd_pct_revenue)
            margin_components['operating_income'].append(operating_income)
            margin_components['operating_margin_pct'].append(operating_margin_pct)

        return margin_components

    def _forecast_working_capital(self,
                                base_metrics: Dict[str, float],
                                drivers: BusinessDrivers,
                                forecast_years: List[int],
                                revenue_components: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Forecast working capital using operational drivers"""

        working_capital = {
            'accounts_receivable': [],
            'inventory': [],
            'accounts_payable': [],
            'net_working_capital': [],
            'working_capital_change': [],
            'dso_days': [],
            'dpo_days': [],
            'inventory_turns': []
        }

        prev_nwc = (base_metrics['accounts_receivable'] +
                   base_metrics['inventory'] -
                   base_metrics['accounts_payable'])

        for i, year in enumerate(forecast_years):
            revenue = revenue_components['total_revenue'][i]

            # Days Sales Outstanding
            dso = drivers.dso_days  # Could vary by year
            accounts_receivable = (revenue / 365) * dso

            # Inventory management
            inventory_turns = drivers.inventory_turns
            inventory = revenue / inventory_turns

            # Days Payable Outstanding
            dpo = drivers.dpo_days
            # Assuming accounts payable is against COGS (70% of revenue)
            cogs = revenue * 0.7  # Simplified
            accounts_payable = (cogs / 365) * dpo

            # Net working capital
            net_working_capital = accounts_receivable + inventory - accounts_payable
            working_capital_change = net_working_capital - prev_nwc
            prev_nwc = net_working_capital

            working_capital['accounts_receivable'].append(accounts_receivable)
            working_capital['inventory'].append(inventory)
            working_capital['accounts_payable'].append(accounts_payable)
            working_capital['net_working_capital'].append(net_working_capital)
            working_capital['working_capital_change'].append(working_capital_change)
            working_capital['dso_days'].append(dso)
            working_capital['dpo_days'].append(dpo)
            working_capital['inventory_turns'].append(inventory_turns)

        return working_capital

    def _forecast_capex_schedule(self,
                               base_metrics: Dict[str, float],
                               drivers: BusinessDrivers,
                               forecast_years: List[int],
                               revenue_components: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Forecast capital expenditure with depreciation schedule"""

        capex_schedule = {
            'capex': [],
            'capex_pct_revenue': [],
            'depreciation': [],
            'net_capex': [],
            'cumulative_capex': [],
            'asset_base': []
        }

        base_asset_base = base_metrics['total_assets']
        cumulative_capex = 0

        for i, year in enumerate(forecast_years):
            revenue = revenue_components['total_revenue'][i]

            # CapEx as percentage of revenue
            capex = revenue * drivers.capex_as_pct_sales
            capex_pct_revenue = drivers.capex_as_pct_sales

            # Depreciation (simplified straight-line)
            depreciation = base_asset_base * drivers.depreciation_rate

            # Net CapEx impact
            net_capex = capex - depreciation

            # Update asset base
            base_asset_base += net_capex
            cumulative_capex += capex

            capex_schedule['capex'].append(capex)
            capex_schedule['capex_pct_revenue'].append(capex_pct_revenue)
            capex_schedule['depreciation'].append(depreciation)
            capex_schedule['net_capex'].append(net_capex)
            capex_schedule['cumulative_capex'].append(cumulative_capex)
            capex_schedule['asset_base'].append(base_asset_base)

        return capex_schedule

    def _calculate_free_cash_flow(self,
                                revenue_components: Dict[str, List[float]],
                                margin_components: Dict[str, List[float]],
                                working_capital: Dict[str, List[float]],
                                capex_schedule: Dict[str, List[float]]) -> List[float]:
        """Calculate free cash flow from forecasted components"""

        free_cash_flows = []

        for i in range(len(revenue_components['total_revenue'])):
            # Operating income
            operating_income = margin_components['operating_income'][i]

            # Tax (simplified)
            tax_rate = 0.25
            after_tax_operating_income = operating_income * (1 - tax_rate)

            # Add back depreciation (non-cash)
            depreciation = capex_schedule['depreciation'][i]

            # Subtract working capital change
            wc_change = working_capital['working_capital_change'][i]

            # Subtract CapEx
            capex = capex_schedule['capex'][i]

            # Free Cash Flow
            fcf = after_tax_operating_income + depreciation - wc_change - capex
            free_cash_flows.append(fcf)

        return free_cash_flows

    def _calculate_summary_metrics(self,
                                 forecast: DriverBasedForecast,
                                 base_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate summary metrics and growth rates"""

        if not forecast.revenue_components['total_revenue']:
            return {}

        base_revenue = base_metrics['revenue']
        final_revenue = forecast.revenue_components['total_revenue'][-1]

        revenue_cagr = ((final_revenue / base_revenue) ** (1/len(forecast.forecast_years))) - 1

        avg_operating_margin = np.mean(forecast.margin_components['operating_margin_pct'])
        avg_fcf = np.mean(forecast.free_cash_flow)

        summary = {
            'revenue_cagr': revenue_cagr,
            'final_year_revenue': final_revenue,
            'avg_operating_margin': avg_operating_margin,
            'avg_annual_fcf': avg_fcf,
            'total_fcf': sum(forecast.free_cash_flow),
            'capex_intensity': np.mean(forecast.capex_schedule['capex_pct_revenue']),
            'avg_working_capital': np.mean(forecast.working_capital['net_working_capital']),
            'forecast_quality': 'Driver-Based'
        }

        return summary

def create_default_drivers(dataset: CompanyDataset) -> BusinessDrivers:
    """Create default drivers based on company sector and fundamentals"""

    sector = dataset.snapshot.sector.lower()
    fundamentals = dataset.financials.fundamentals

    # Sector-specific defaults
    if 'technology' in sector:
        return BusinessDrivers(
            unit_volume_growth=0.08,
            price_realization=0.03,
            customer_growth=0.12,
            arpu_growth=0.05,
            gross_margin_expansion=0.005,
            sga_leverage=-0.8,
            rd_intensity_change=0.002,
            capex_as_pct_sales=0.03
        )
    elif 'consumer' in sector:
        return BusinessDrivers(
            unit_volume_growth=0.04,
            price_realization=0.025,
            customer_growth=0.06,
            arpu_growth=0.02,
            gross_margin_expansion=0.002,
            sga_leverage=-0.3,
            capex_as_pct_sales=0.045
        )
    elif 'healthcare' in sector:
        return BusinessDrivers(
            unit_volume_growth=0.06,
            price_realization=0.04,
            customer_growth=0.05,
            arpu_growth=0.04,
            gross_margin_expansion=0.008,
            rd_intensity_change=0.005,
            capex_as_pct_sales=0.06
        )
    else:
        # Default/Industrial
        return BusinessDrivers()

def test_driver_based_forecasting():
    """Test the driver-based forecasting system"""

    # Create mock dataset
    class MockSnapshot:
        def __init__(self):
            self.ticker = 'AAPL'
            self.sector = 'Technology'

    class MockFinancials:
        def __init__(self):
            self.fundamentals = {
                'totalRevenue': 394328000000,
                'grossProfit': 169148000000,
                'operatingIncome': 114301000000,
                'researchDevelopment': 29915000000,
                'sellingGeneralAdministrative': 25094000000,
                'totalAssets': 352755000000,
                'operatingCashflow': 122151000000,
                'capitalExpenditures': 10708000000
            }

    class MockDataset:
        def __init__(self):
            self.ticker = 'AAPL'
            self.snapshot = MockSnapshot()
            self.financials = MockFinancials()

    dataset = MockDataset()

    print("="*60)
    print("TESTING DRIVER-BASED FORECASTING")
    print("="*60)

    # Create forecaster
    forecaster = DriverBasedForecaster()

    # Create custom drivers for Apple
    drivers = BusinessDrivers(
        unit_volume_growth=0.05,      # iPhone unit growth
        price_realization=0.02,       # Price increases
        customer_growth=0.08,         # Services customer growth
        arpu_growth=0.06,            # Services ARPU growth
        gross_margin_expansion=0.003, # Gradual margin improvement
        sga_leverage=-0.5,           # Operating leverage
        rd_intensity_change=0.001,   # Slight R&D increase
        capex_as_pct_sales=0.025     # CapEx efficiency
    )

    # Generate forecast
    forecast = forecaster.build_driver_forecast(dataset, drivers)

    # Display results
    print(f"\n📊 5-Year Driver-Based Forecast for {dataset.ticker}:")
    print(f"Revenue CAGR: {forecast.summary_metrics['revenue_cagr']*100:.1f}%")
    print(f"Final Year Revenue: ${forecast.summary_metrics['final_year_revenue']/1e9:.1f}B")
    print(f"Avg Operating Margin: {forecast.summary_metrics['avg_operating_margin']*100:.1f}%")
    print(f"Total FCF (5Y): ${forecast.summary_metrics['total_fcf']/1e9:.1f}B")

    print(f"\n📈 Revenue Breakdown (Year 5):")
    if len(forecast.revenue_components['total_revenue']) >= 1:
        year5_revenue = forecast.revenue_components['total_revenue'][-1]
        volume_revenue = forecast.revenue_components['volume_driven_revenue'][-1]
        customer_revenue = forecast.revenue_components['customer_driven_revenue'][-1]

        print(f"  Total Revenue: ${year5_revenue/1e9:.1f}B")
        print(f"  Volume x Price: ${volume_revenue/1e9:.1f}B")
        print(f"  Customer x ARPU: ${customer_revenue/1e9:.1f}B")

if __name__ == "__main__":
    test_driver_based_forecasting()