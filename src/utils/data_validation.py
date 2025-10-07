"""
Data Validation and Sanity Check System
Ensures valuation results are realistic and within acceptable bounds
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    adjusted_value: Optional[float] = None
    confidence_penalty: float = 0.0

class DataValidator:
    """Comprehensive data validation and sanity checking system"""

    def __init__(self):
        # Define realistic bounds for different metrics
        self.validation_bounds = {
            # Valuation bounds (as multiples of market cap)
            'dcf_value': {'min_multiple': 0.1, 'max_multiple': 50.0},
            'relative_value': {'min_multiple': 0.1, 'max_multiple': 20.0},
            'sum_of_parts_value': {'min_multiple': 0.1, 'max_multiple': 10.0},
            'asset_based_value': {'min_multiple': 0.05, 'max_multiple': 5.0},
            'real_options_value': {'min_multiple': 0.0, 'max_multiple': 5.0},

            # Financial ratios
            'pe_ratio': {'min': 1.0, 'max': 200.0},
            'pb_ratio': {'min': 0.1, 'max': 50.0},
            'ev_ebitda': {'min': 1.0, 'max': 100.0},
            'revenue_growth': {'min': -0.8, 'max': 5.0},  # -80% to 500%
            'margin': {'min': -1.0, 'max': 1.0},  # -100% to 100%

            # Market data
            'risk_free_rate': {'min': 0.0, 'max': 0.20},  # 0% to 20%
            'volatility': {'min': 0.05, 'max': 2.0},  # 5% to 200%
            'beta': {'min': -3.0, 'max': 5.0},
            'wacc': {'min': 0.01, 'max': 0.50},  # 1% to 50%
        }

    def validate_valuation_results(self, dataset, results: Dict[str, float]) -> ValidationResult:
        """Validate multi-method valuation results"""
        warnings = []
        errors = []
        adjusted_results = {}
        total_confidence_penalty = 0.0

        # Get reference values for validation
        market_cap = dataset.snapshot.market_cap or 0
        current_price = dataset.snapshot.current_price or 0

        # If we don't have market cap, estimate it from current price and shares
        if market_cap == 0 and current_price > 0:
            shares = dataset.snapshot.shares_outstanding or 0
            if shares > 0:
                market_cap = current_price * shares

        # If still no reference, use DCF as baseline
        reference_value = market_cap or results.get('dcf_value', 0)

        if reference_value == 0:
            errors.append("No reference value available for validation")
            return ValidationResult(is_valid=False, warnings=warnings, errors=errors)

        # Validate each valuation method
        for method, value in results.items():
            if method.endswith('_value') and value > 0:
                validation = self._validate_single_valuation(method, value, reference_value)


                if not validation.is_valid:
                    if validation.adjusted_value is not None:
                        adjusted_results[method] = validation.adjusted_value
                        warnings.append(f"{method}: Adjusted from ${value:,.0f} to ${validation.adjusted_value:,.0f}")
                        total_confidence_penalty += validation.confidence_penalty
                    else:
                        errors.append(f"{method}: {', '.join(validation.errors)}")

                warnings.extend(validation.warnings)

        # Update results with adjusted values
        for method, adjusted_value in adjusted_results.items():
            results[method] = adjusted_value

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            confidence_penalty=min(total_confidence_penalty, 0.5)  # Cap at 50% penalty
        )

    def _validate_single_valuation(self, method: str, value: float, reference_value: float) -> ValidationResult:
        """Validate a single valuation method"""
        warnings = []
        errors = []
        adjusted_value = None
        confidence_penalty = 0.0

        # Check if method has validation bounds
        if method not in self.validation_bounds:
            return ValidationResult(is_valid=True, warnings=warnings, errors=errors)

        bounds = self.validation_bounds[method]

        # Calculate multiple of reference value
        multiple = value / reference_value if reference_value > 0 else 0

        # Special handling for Real Options based on company size
        if method == 'real_options_value':
            # For large companies (>$100B market cap), Real Options should be much smaller
            if reference_value > 100_000_000_000:  # >$100B
                # Large mature companies: Real Options should be max 10% of market cap
                max_reasonable_value = reference_value * 0.1
                if value > max_reasonable_value:
                    adjusted_value = max_reasonable_value
                    confidence_penalty = 0.4
                    warnings.append(f"Real Options value ${value:,.0f} unrealistic for large mature company")
                    logger.warning(f"Real Options: Adjusted for large company from ${value:,.0f} to ${adjusted_value:,.0f}")
                    return ValidationResult(
                        is_valid=False,
                        warnings=warnings,
                        errors=errors,
                        adjusted_value=adjusted_value,
                        confidence_penalty=confidence_penalty
                    )

        # Check bounds
        min_multiple = bounds['min_multiple']
        max_multiple = bounds['max_multiple']

        if multiple < min_multiple:
            if multiple < min_multiple * 0.1:  # Extremely low
                errors.append(f"Value ${value:,.0f} is unrealistically low ({multiple:.2f}x reference)")
            else:
                warnings.append(f"Value ${value:,.0f} is unusually low ({multiple:.2f}x reference)")
                confidence_penalty = 0.1

        elif multiple > max_multiple:
            if multiple > max_multiple * 10:  # Extremely high - adjust it
                adjusted_value = reference_value * max_multiple
                confidence_penalty = 0.3
                logger.warning(f"{method}: Adjusted extreme value from ${value:,.0f} to ${adjusted_value:,.0f}")
            elif multiple > max_multiple * 2:  # Very high
                warnings.append(f"Value ${value:,.0f} is unrealistically high ({multiple:.2f}x reference)")
                confidence_penalty = 0.2
            else:
                warnings.append(f"Value ${value:,.0f} is unusually high ({multiple:.2f}x reference)")
                confidence_penalty = 0.1

        is_valid = len(errors) == 0 and adjusted_value is None

        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            adjusted_value=adjusted_value,
            confidence_penalty=confidence_penalty
        )

    def validate_financial_ratios(self, ratios: Dict[str, float]) -> ValidationResult:
        """Validate financial ratios"""
        warnings = []
        errors = []

        for ratio_name, value in ratios.items():
            if ratio_name in self.validation_bounds and value is not None:
                bounds = self.validation_bounds[ratio_name]

                if value < bounds['min'] or value > bounds['max']:
                    if abs(value) > bounds['max'] * 10:
                        errors.append(f"{ratio_name}: {value:.2f} is extremely unrealistic")
                    else:
                        warnings.append(f"{ratio_name}: {value:.2f} is outside normal range ({bounds['min']:.2f} - {bounds['max']:.2f})")

        return ValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors
        )

    def validate_market_data(self, market_data: Dict[str, float]) -> ValidationResult:
        """Validate market data inputs"""
        warnings = []
        errors = []

        for data_name, value in market_data.items():
            if data_name in self.validation_bounds and value is not None:
                bounds = self.validation_bounds[data_name]

                if value < bounds['min']:
                    if value < 0 and bounds['min'] >= 0:
                        errors.append(f"{data_name}: {value:.4f} cannot be negative")
                    else:
                        warnings.append(f"{data_name}: {value:.4f} is unusually low")

                elif value > bounds['max']:
                    if value > bounds['max'] * 5:
                        errors.append(f"{data_name}: {value:.4f} is extremely unrealistic")
                    else:
                        warnings.append(f"{data_name}: {value:.4f} is unusually high")

        return ValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors
        )

    def apply_sanity_adjustments(self, results: Dict[str, Any], validation_result: ValidationResult) -> Dict[str, Any]:
        """Apply sanity adjustments to results"""
        if not validation_result.warnings and not validation_result.errors:
            return results

        # Log validation issues
        for warning in validation_result.warnings:
            logger.warning(f"Data validation warning: {warning}")

        for error in validation_result.errors:
            logger.error(f"Data validation error: {error}")

        # Apply confidence penalty to confidence scores
        if hasattr(results, 'confidence_scores') and validation_result.confidence_penalty > 0:
            for method in results.confidence_scores:
                results.confidence_scores[method] *= (1 - validation_result.confidence_penalty)

        return results

    def get_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of all validation results"""
        total_warnings = sum(len(vr.warnings) for vr in validation_results)
        total_errors = sum(len(vr.errors) for vr in validation_results)
        overall_valid = all(vr.is_valid for vr in validation_results)

        return {
            'overall_valid': overall_valid,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'confidence_impact': sum(vr.confidence_penalty for vr in validation_results),
            'validation_score': max(0.0, 1.0 - (total_warnings * 0.1 + total_errors * 0.3))
        }

# Global validator instance
data_validator = DataValidator()