"""
Advanced Monte Carlo Simulation for Equity Valuation
- Parameter distributions for WACC, growth rates, margins
- Correlation matrices between variables
- 10,000+ simulation runs with percentile outputs
- Value-at-Risk calculations (5th, 95th percentiles)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, beta, uniform
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_pipeline.models import CompanyDataset

logger = logging.getLogger(__name__)

@dataclass
class ParameterDistribution:
    """Statistical distribution for a parameter"""
    name: str
    distribution_type: str  # 'normal', 'lognormal', 'beta', 'uniform', 'triangular'
    base_value: float
    std_dev: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    alpha: Optional[float] = None  # For beta distribution
    beta_param: Optional[float] = None  # For beta distribution
    mode: Optional[float] = None  # For triangular distribution

@dataclass
class CorrelationMatrix:
    """Correlation matrix between parameters"""
    parameters: List[str] = field(default_factory=list)
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results"""
    simulation_runs: int = 0
    parameter_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    dcf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    enterprise_values: np.ndarray = field(default_factory=lambda: np.array([]))
    free_cash_flows: Dict[str, np.ndarray] = field(default_factory=dict)

    # Statistical results
    percentiles: Dict[str, float] = field(default_factory=dict)
    var_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Risk metrics
    probability_of_loss: float = 0.0
    expected_shortfall: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Sensitivity analysis
    parameter_sensitivities: Dict[str, float] = field(default_factory=dict)

class AdvancedMonteCarloSimulator:
    """Advanced Monte Carlo simulation for equity valuation"""

    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
        self.random_seed = 42  # For reproducibility
        np.random.seed(self.random_seed)

    def run_simulation(self,
                      dataset: CompanyDataset,
                      parameter_distributions: Dict[str, ParameterDistribution],
                      correlation_matrix: Optional[CorrelationMatrix] = None,
                      config: Optional[Dict] = None) -> MonteCarloResults:
        """Run advanced Monte Carlo simulation"""

        print(f"🎲 Running Monte Carlo simulation ({self.num_simulations:,} iterations)...")

        config = config or {}

        # Initialize results
        results = MonteCarloResults()
        results.simulation_runs = self.num_simulations

        # Generate correlated parameter samples
        results.parameter_samples = self._generate_parameter_samples(
            parameter_distributions, correlation_matrix
        )

        # Run DCF valuations for each simulation
        dcf_values = []
        enterprise_values = []
        annual_fcfs = {f'year_{i+1}': [] for i in range(5)}

        print(f"  📊 Simulating valuations...")

        for i in range(self.num_simulations):
            if i % 1000 == 0:
                print(f"    Progress: {i:,}/{self.num_simulations:,} ({i/self.num_simulations*100:.1f}%)")

            # Extract parameters for this simulation
            sim_params = {param: samples[i] for param, samples in results.parameter_samples.items()}

            # Run DCF with simulated parameters
            dcf_result = self._run_single_dcf_simulation(dataset, sim_params, config)

            dcf_values.append(dcf_result['dcf_value'])
            enterprise_values.append(dcf_result['enterprise_value'])

            # Store annual FCFs
            for year, fcf in dcf_result['annual_fcfs'].items():
                annual_fcfs[year].append(fcf)

        # Convert to numpy arrays
        results.dcf_values = np.array(dcf_values)
        results.enterprise_values = np.array(enterprise_values)
        results.free_cash_flows = {year: np.array(fcfs) for year, fcfs in annual_fcfs.items()}

        # Calculate statistical metrics
        results = self._calculate_statistical_metrics(results)

        # Calculate risk metrics
        results = self._calculate_risk_metrics(results, dataset)

        # Calculate parameter sensitivities
        results = self._calculate_parameter_sensitivities(results)

        print(f"✅ Monte Carlo simulation complete!")
        self._print_simulation_summary(results)

        return results

    def _generate_parameter_samples(self,
                                   parameter_distributions: Dict[str, ParameterDistribution],
                                   correlation_matrix: Optional[CorrelationMatrix] = None) -> Dict[str, np.ndarray]:
        """Generate correlated parameter samples"""

        print(f"  🔢 Generating parameter samples...")

        # Generate independent samples first
        independent_samples = {}

        for param_name, dist in parameter_distributions.items():
            samples = self._sample_from_distribution(dist)
            independent_samples[param_name] = samples

        # Apply correlations if specified
        if correlation_matrix and len(correlation_matrix.correlation_matrix) > 0:
            print(f"  🔗 Applying correlation structure...")
            correlated_samples = self._apply_correlation_structure(
                independent_samples, correlation_matrix
            )
            return correlated_samples

        return independent_samples

    def _sample_from_distribution(self, dist: ParameterDistribution) -> np.ndarray:
        """Sample from a specific distribution"""

        if dist.distribution_type == 'normal':
            samples = np.random.normal(dist.base_value, dist.std_dev, self.num_simulations)

        elif dist.distribution_type == 'lognormal':
            # Log-normal distribution for positive-only variables
            mu = np.log(dist.base_value)
            sigma = dist.std_dev or 0.1
            samples = np.random.lognormal(mu, sigma, self.num_simulations)

        elif dist.distribution_type == 'beta':
            # Beta distribution bounded between min and max
            if dist.alpha and dist.beta_param and dist.min_value is not None and dist.max_value is not None:
                beta_samples = np.random.beta(dist.alpha, dist.beta_param, self.num_simulations)
                samples = dist.min_value + beta_samples * (dist.max_value - dist.min_value)
            else:
                # Default to normal if beta parameters not provided
                samples = np.random.normal(dist.base_value, dist.std_dev or 0.1, self.num_simulations)

        elif dist.distribution_type == 'uniform':
            min_val = dist.min_value or (dist.base_value * 0.8)
            max_val = dist.max_value or (dist.base_value * 1.2)
            samples = np.random.uniform(min_val, max_val, self.num_simulations)

        elif dist.distribution_type == 'triangular':
            min_val = dist.min_value or (dist.base_value * 0.7)
            max_val = dist.max_value or (dist.base_value * 1.3)
            mode = dist.mode or dist.base_value
            samples = np.random.triangular(min_val, mode, max_val, self.num_simulations)

        else:
            # Default to normal distribution
            std = dist.std_dev or (dist.base_value * 0.1)
            samples = np.random.normal(dist.base_value, std, self.num_simulations)

        # Apply bounds if specified
        if dist.min_value is not None:
            samples = np.maximum(samples, dist.min_value)
        if dist.max_value is not None:
            samples = np.minimum(samples, dist.max_value)

        return samples

    def _apply_correlation_structure(self,
                                   independent_samples: Dict[str, np.ndarray],
                                   correlation_matrix: CorrelationMatrix) -> Dict[str, np.ndarray]:
        """Apply correlation structure using Cholesky decomposition"""

        try:
            # Create matrix of independent samples
            param_names = correlation_matrix.parameters
            sample_matrix = np.column_stack([independent_samples[param] for param in param_names])

            # Convert to standard normal
            uniform_samples = np.zeros_like(sample_matrix)
            for i, param in enumerate(param_names):
                # Convert to uniform then to standard normal
                ranks = stats.rankdata(sample_matrix[:, i])
                uniform_samples[:, i] = (ranks - 0.5) / len(ranks)

            # Convert to standard normal
            normal_samples = stats.norm.ppf(uniform_samples)

            # Apply correlation using Cholesky decomposition
            try:
                chol = np.linalg.cholesky(correlation_matrix.correlation_matrix)
                correlated_normal = normal_samples @ chol.T
            except np.linalg.LinAlgError:
                # If matrix is not positive definite, use eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix.correlation_matrix)
                eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
                sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
                correlated_normal = normal_samples @ sqrt_matrix.T

            # Convert back to uniform then to original distributions
            correlated_uniform = stats.norm.cdf(correlated_normal)

            correlated_samples = {}
            for i, param in enumerate(param_names):
                # Convert uniform back to original distribution
                original_samples = independent_samples[param]
                sorted_original = np.sort(original_samples)

                # Use inverse transform sampling
                indices = (correlated_uniform[:, i] * (len(sorted_original) - 1)).astype(int)
                indices = np.clip(indices, 0, len(sorted_original) - 1)

                correlated_samples[param] = sorted_original[indices]

            # Add any parameters not in correlation matrix
            for param, samples in independent_samples.items():
                if param not in correlated_samples:
                    correlated_samples[param] = samples

            return correlated_samples

        except Exception as e:
            logger.warning(f"Correlation application failed: {e}. Using independent samples.")
            return independent_samples

    def _run_single_dcf_simulation(self,
                                  dataset: CompanyDataset,
                                  sim_params: Dict[str, float],
                                  config: Dict) -> Dict[str, Any]:
        """Run a single DCF valuation with simulated parameters"""

        fundamentals = dataset.financials.fundamentals

        # Extract simulated parameters
        revenue_growth = sim_params.get('revenue_growth', 0.05)
        terminal_growth = sim_params.get('terminal_growth', 0.025)
        wacc = sim_params.get('wacc', 0.08)
        operating_margin = sim_params.get('operating_margin', 0.15)
        tax_rate = sim_params.get('tax_rate', 0.25)
        capex_pct_sales = sim_params.get('capex_pct_sales', 0.04)

        # Base financials
        base_revenue = fundamentals.get('totalRevenue', 100e9)
        base_operating_income = fundamentals.get('operatingIncome', base_revenue * operating_margin)
        shares_outstanding = fundamentals.get('sharesOutstanding', 1e9)

        # Project 5-year cash flows
        annual_fcfs = {}
        fcf_projections = []

        current_revenue = base_revenue

        for year in range(1, 6):
            # Revenue projection
            current_revenue *= (1 + revenue_growth)

            # Operating income
            operating_income = current_revenue * operating_margin

            # Taxes
            after_tax_income = operating_income * (1 - tax_rate)

            # CapEx and depreciation (simplified)
            capex = current_revenue * capex_pct_sales
            depreciation = capex * 0.8  # Simplified

            # Free cash flow
            fcf = after_tax_income + depreciation - capex

            fcf_projections.append(fcf)
            annual_fcfs[f'year_{year}'] = fcf

        # Terminal value
        terminal_fcf = fcf_projections[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)

        # Present value calculations
        pv_fcf = sum(fcf / ((1 + wacc) ** year) for year, fcf in enumerate(fcf_projections, 1))
        pv_terminal = terminal_value / ((1 + wacc) ** 5)

        enterprise_value = pv_fcf + pv_terminal
        dcf_value = enterprise_value / shares_outstanding if shares_outstanding > 0 else 0

        return {
            'dcf_value': dcf_value,
            'enterprise_value': enterprise_value,
            'annual_fcfs': annual_fcfs,
            'terminal_value': terminal_value,
            'pv_fcf': pv_fcf,
            'pv_terminal': pv_terminal
        }

    def _calculate_statistical_metrics(self, results: MonteCarloResults) -> MonteCarloResults:
        """Calculate statistical metrics for simulation results"""

        dcf_values = results.dcf_values

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        results.percentiles = {
            f'p{p}': np.percentile(dcf_values, p) for p in percentiles
        }

        # Confidence intervals
        results.confidence_intervals = {
            '90%': (np.percentile(dcf_values, 5), np.percentile(dcf_values, 95)),
            '95%': (np.percentile(dcf_values, 2.5), np.percentile(dcf_values, 97.5)),
            '99%': (np.percentile(dcf_values, 0.5), np.percentile(dcf_values, 99.5))
        }

        # Basic statistics
        results.percentiles.update({
            'mean': np.mean(dcf_values),
            'std': np.std(dcf_values),
            'min': np.min(dcf_values),
            'max': np.max(dcf_values)
        })

        return results

    def _calculate_risk_metrics(self, results: MonteCarloResults, dataset: CompanyDataset) -> MonteCarloResults:
        """Calculate risk metrics and Value-at-Risk"""

        dcf_values = results.dcf_values
        current_price = dataset.snapshot.current_price

        # Value-at-Risk calculations
        results.var_metrics = {
            'var_5%': np.percentile(dcf_values, 5),
            'var_1%': np.percentile(dcf_values, 1),
            'cvar_5%': np.mean(dcf_values[dcf_values <= np.percentile(dcf_values, 5)]),  # Expected Shortfall
            'cvar_1%': np.mean(dcf_values[dcf_values <= np.percentile(dcf_values, 1)])
        }

        # Probability of loss (vs current price)
        if current_price:
            results.probability_of_loss = np.mean(dcf_values < current_price)
        else:
            results.probability_of_loss = 0.0

        # Expected Shortfall (average loss in worst 5% scenarios)
        var_5_threshold = results.var_metrics['var_5%']
        worst_5_percent = dcf_values[dcf_values <= var_5_threshold]
        results.expected_shortfall = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else 0

        # Distribution shape metrics
        results.skewness = stats.skew(dcf_values)
        results.kurtosis = stats.kurtosis(dcf_values)

        return results

    def _calculate_parameter_sensitivities(self, results: MonteCarloResults) -> MonteCarloResults:
        """Calculate parameter sensitivities using correlation analysis"""

        dcf_values = results.dcf_values

        for param_name, param_samples in results.parameter_samples.items():
            try:
                # Calculate correlation between parameter and DCF value
                correlation = np.corrcoef(param_samples, dcf_values)[0, 1]
                results.parameter_sensitivities[param_name] = correlation
            except:
                results.parameter_sensitivities[param_name] = 0.0

        return results

    def _print_simulation_summary(self, results: MonteCarloResults):
        """Print summary of simulation results"""

        print(f"\n📊 Monte Carlo Simulation Results ({results.simulation_runs:,} runs):")
        print(f"Mean DCF Value: ${results.percentiles['mean']:.2f}")
        print(f"Std Deviation: ${results.percentiles['std']:.2f}")

        print(f"\n📈 Key Percentiles:")
        print(f"P5 (VaR 95%): ${results.percentiles['p5']:.2f}")
        print(f"P25: ${results.percentiles['p25']:.2f}")
        print(f"P50 (Median): ${results.percentiles['p50']:.2f}")
        print(f"P75: ${results.percentiles['p75']:.2f}")
        print(f"P95: ${results.percentiles['p95']:.2f}")

        print(f"\n⚠️ Risk Metrics:")
        print(f"VaR (5%): ${results.var_metrics['var_5%']:.2f}")
        print(f"Expected Shortfall: ${results.expected_shortfall:.2f}")
        print(f"Probability of Loss: {results.probability_of_loss*100:.1f}%")

        print(f"\n📊 Distribution Shape:")
        print(f"Skewness: {results.skewness:.3f}")
        print(f"Kurtosis: {results.kurtosis:.3f}")

        print(f"\n🎯 Top Parameter Sensitivities:")
        sorted_sensitivities = sorted(results.parameter_sensitivities.items(),
                                    key=lambda x: abs(x[1]), reverse=True)
        for param, sensitivity in sorted_sensitivities[:5]:
            print(f"  {param}: {sensitivity:.3f}")

def create_default_parameter_distributions(dataset: CompanyDataset) -> Dict[str, ParameterDistribution]:
    """Create default parameter distributions based on company fundamentals"""

    fundamentals = dataset.financials.fundamentals
    sector = dataset.snapshot.sector.lower()

    # Base values from fundamentals
    revenue_growth_base = fundamentals.get('revenueGrowth', 0.05)
    operating_margin_base = fundamentals.get('operatingMargin', 0.15)

    # Sector-specific volatilities
    if 'technology' in sector:
        revenue_vol = 0.15  # Higher volatility for tech
        margin_vol = 0.05
        wacc_vol = 0.015
    elif 'utility' in sector:
        revenue_vol = 0.05  # Lower volatility for utilities
        margin_vol = 0.02
        wacc_vol = 0.008
    else:
        revenue_vol = 0.10  # Default volatility
        margin_vol = 0.03
        wacc_vol = 0.012

    distributions = {
        'revenue_growth': ParameterDistribution(
            name='Revenue Growth',
            distribution_type='normal',
            base_value=revenue_growth_base,
            std_dev=revenue_vol,
            min_value=-0.1,  # Max 10% decline
            max_value=0.3    # Max 30% growth
        ),

        'terminal_growth': ParameterDistribution(
            name='Terminal Growth',
            distribution_type='beta',
            base_value=0.025,
            alpha=2,
            beta_param=4,
            min_value=0.01,
            max_value=0.04
        ),

        'wacc': ParameterDistribution(
            name='WACC',
            distribution_type='normal',
            base_value=0.08,
            std_dev=wacc_vol,
            min_value=0.05,
            max_value=0.15
        ),

        'operating_margin': ParameterDistribution(
            name='Operating Margin',
            distribution_type='normal',
            base_value=operating_margin_base,
            std_dev=margin_vol,
            min_value=0.0,
            max_value=0.5
        ),

        'tax_rate': ParameterDistribution(
            name='Tax Rate',
            distribution_type='uniform',
            base_value=0.25,
            min_value=0.21,
            max_value=0.28
        ),

        'capex_pct_sales': ParameterDistribution(
            name='CapEx % Sales',
            distribution_type='lognormal',
            base_value=0.04,
            std_dev=0.02,
            min_value=0.01,
            max_value=0.12
        )
    }

    return distributions

def create_default_correlation_matrix() -> CorrelationMatrix:
    """Create default correlation matrix for financial parameters"""

    parameters = ['revenue_growth', 'terminal_growth', 'wacc', 'operating_margin', 'tax_rate', 'capex_pct_sales']

    # Default correlation matrix (economic intuition)
    correlation_matrix = np.array([
        #  rev_gr  term_gr  wacc   op_mar  tax    capex
        [  1.00,   0.30,  -0.20,  0.15,   0.00,  0.25 ],  # revenue_growth
        [  0.30,   1.00,  -0.40,  0.10,   0.00,  0.15 ],  # terminal_growth
        [ -0.20,  -0.40,   1.00, -0.25,   0.00, -0.10 ],  # wacc
        [  0.15,   0.10,  -0.25,  1.00,   0.00,  0.00 ],  # operating_margin
        [  0.00,   0.00,   0.00,  0.00,   1.00,  0.00 ],  # tax_rate
        [  0.25,   0.15,  -0.10,  0.00,   0.00,  1.00 ]   # capex_pct_sales
    ])

    return CorrelationMatrix(
        parameters=parameters,
        correlation_matrix=correlation_matrix
    )

def test_monte_carlo_simulation():
    """Test the advanced Monte Carlo simulation"""

    # Create mock dataset
    class MockSnapshot:
        def __init__(self):
            self.ticker = 'AAPL'
            self.sector = 'Technology'
            self.current_price = 150.0

    class MockFinancials:
        def __init__(self):
            self.fundamentals = {
                'totalRevenue': 394328000000,
                'operatingIncome': 114301000000,
                'operatingMargin': 0.29,
                'revenueGrowth': 0.08,
                'sharesOutstanding': 15728700000
            }

    class MockDataset:
        def __init__(self):
            self.ticker = 'AAPL'
            self.snapshot = MockSnapshot()
            self.financials = MockFinancials()

    dataset = MockDataset()

    print("="*60)
    print("TESTING ADVANCED MONTE CARLO SIMULATION")
    print("="*60)

    # Create simulator
    simulator = AdvancedMonteCarloSimulator(num_simulations=5000)  # Reduced for testing

    # Create parameter distributions
    distributions = create_default_parameter_distributions(dataset)

    # Create correlation matrix
    correlation_matrix = create_default_correlation_matrix()

    # Run simulation
    results = simulator.run_simulation(
        dataset=dataset,
        parameter_distributions=distributions,
        correlation_matrix=correlation_matrix
    )

    # Additional analysis
    print(f"\n💼 Investment Decision Framework:")
    current_price = dataset.snapshot.current_price
    prob_upside = np.mean(results.dcf_values > current_price) * 100
    median_upside = (results.percentiles['p50'] / current_price - 1) * 100

    print(f"Current Price: ${current_price:.2f}")
    print(f"Probability of Upside: {prob_upside:.1f}%")
    print(f"Median Upside: {median_upside:+.1f}%")

    # Risk-adjusted recommendation
    if prob_upside > 70 and median_upside > 15:
        recommendation = "STRONG BUY"
    elif prob_upside > 60 and median_upside > 5:
        recommendation = "BUY"
    elif prob_upside > 40:
        recommendation = "HOLD"
    else:
        recommendation = "SELL"

    print(f"Monte Carlo Recommendation: {recommendation}")

if __name__ == "__main__":
    test_monte_carlo_simulation()