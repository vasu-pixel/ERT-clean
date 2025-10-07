# src/utils/data_validator.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation with details"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 0-100 validation score

class DataIntegrityError(Exception):
    """Raised when critical data integrity issues are found"""
    pass

class StaleDataError(Exception):
    """Raised when data is too old to be reliable"""
    pass

class DataValidator:
    """Comprehensive data validation for equity research reports"""

    def __init__(self):
        self.validation_rules = {
            'critical_fields': [
                'current_price', 'market_cap', 'ticker', 'company_name'
            ],
            'financial_metrics': [
                'pe_ratio', 'revenue_growth', 'profit_margin'
            ],
            'min_price': 0.01,  # Minimum valid stock price
            'max_price': 10000,  # Maximum reasonable stock price
            'min_market_cap': 1_000_000,  # $1M minimum market cap
            'max_pe_ratio': 1000,  # Maximum reasonable P/E
            'min_pe_ratio': -100,  # Minimum reasonable P/E (can be negative)
            'data_freshness_days': 7  # Data older than 7 days triggers warning
        }

    def validate_company_profile(self, company_profile) -> ValidationResult:
        """Validate company profile data comprehensively"""
        errors = []
        warnings = []

        try:
            # Check critical fields
            critical_errors = self._validate_critical_fields(company_profile)
            errors.extend(critical_errors)

            # Check financial metrics
            financial_warnings = self._validate_financial_metrics(company_profile)
            warnings.extend(financial_warnings)

            # Check data consistency
            consistency_errors = self._validate_data_consistency(company_profile)
            errors.extend(consistency_errors)

            # Check for placeholder/null values
            placeholder_errors = self._validate_no_placeholders(company_profile)
            errors.extend(placeholder_errors)

            # Calculate validation score
            score = self._calculate_validation_score(errors, warnings)

            is_valid = len(errors) == 0 and score >= 70

            if errors:
                logger.error(f"Data validation failed for {company_profile.ticker}: {errors}")
            if warnings:
                logger.warning(f"Data validation warnings for {company_profile.ticker}: {warnings}")

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                score=score
            )

        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation process error: {str(e)}"],
                warnings=[],
                score=0
            )

    def _validate_critical_fields(self, profile) -> List[str]:
        """Validate that critical fields are present and valid"""
        errors = []

        # Check ticker
        if not hasattr(profile, 'ticker') or not profile.ticker:
            errors.append("Missing or empty ticker symbol")
        elif len(profile.ticker) > 10:
            errors.append(f"Invalid ticker length: {profile.ticker}")

        # Check company name
        if not hasattr(profile, 'company_name') or not profile.company_name:
            errors.append("Missing or empty company name")

        # Check current price
        if not hasattr(profile, 'current_price') or not profile.current_price:
            errors.append("Missing current price")
        elif profile.current_price <= 0:
            errors.append(f"Invalid current price: ${profile.current_price}")
        elif profile.current_price < self.validation_rules['min_price']:
            errors.append(f"Current price too low: ${profile.current_price}")
        elif profile.current_price > self.validation_rules['max_price']:
            errors.append(f"Current price suspiciously high: ${profile.current_price}")

        # Check market cap
        if not hasattr(profile, 'market_cap') or not profile.market_cap:
            errors.append("Missing market capitalization")
        elif profile.market_cap <= 0:
            errors.append(f"Invalid market cap: ${profile.market_cap}")
        elif profile.market_cap < self.validation_rules['min_market_cap']:
            errors.append(f"Market cap too low: ${profile.market_cap}")

        return errors

    def _validate_financial_metrics(self, profile) -> List[str]:
        """Validate financial metrics for reasonableness"""
        warnings = []

        if not hasattr(profile, 'financial_metrics') or not profile.financial_metrics:
            warnings.append("No financial metrics available")
            return warnings

        metrics = profile.financial_metrics

        # Check P/E ratio
        pe_ratio = metrics.get('pe_ratio')
        if pe_ratio and pe_ratio != 'N/A':
            try:
                pe_float = float(pe_ratio)
                if pe_float < self.validation_rules['min_pe_ratio']:
                    warnings.append(f"P/E ratio unusually low: {pe_float}")
                elif pe_float > self.validation_rules['max_pe_ratio']:
                    warnings.append(f"P/E ratio unusually high: {pe_float}")
            except (ValueError, TypeError):
                warnings.append(f"Invalid P/E ratio format: {pe_ratio}")

        # Check revenue growth
        revenue_growth = metrics.get('revenue_growth')
        if revenue_growth and revenue_growth != 'N/A':
            try:
                growth_float = float(revenue_growth)
                if growth_float < -50:
                    warnings.append(f"Revenue decline severe: {growth_float}%")
                elif growth_float > 200:
                    warnings.append(f"Revenue growth unusually high: {growth_float}%")
            except (ValueError, TypeError):
                warnings.append(f"Invalid revenue growth format: {revenue_growth}")

        # Check profit margins
        profit_margin = metrics.get('profit_margin')
        if profit_margin and profit_margin != 'N/A':
            try:
                margin_float = float(profit_margin)
                if margin_float < -100:
                    warnings.append(f"Profit margin extremely negative: {margin_float}%")
                elif margin_float > 80:
                    warnings.append(f"Profit margin unusually high: {margin_float}%")
            except (ValueError, TypeError):
                warnings.append(f"Invalid profit margin format: {profit_margin}")

        return warnings

    def _validate_data_consistency(self, profile) -> List[str]:
        """Check for internal data consistency"""
        errors = []

        # Check recommendation vs target price consistency
        if hasattr(profile, 'recommendation') and hasattr(profile, 'target_price') and hasattr(profile, 'current_price'):
            if profile.target_price and profile.current_price:
                upside = (profile.target_price - profile.current_price) / profile.current_price

                if profile.recommendation == 'BUY' and upside < -0.05:
                    errors.append(f"BUY rating inconsistent with negative upside: {upside:.1%}")
                elif profile.recommendation == 'SELL' and upside > 0.05:
                    errors.append(f"SELL rating inconsistent with positive upside: {upside:.1%}")

        # Check market cap vs price consistency (if shares outstanding available)
        if hasattr(profile, 'market_cap') and hasattr(profile, 'current_price'):
            if profile.market_cap and profile.current_price:
                implied_shares = profile.market_cap / profile.current_price
                if implied_shares < 1_000_000:  # Less than 1M shares seems low
                    errors.append(f"Implied shares outstanding very low: {implied_shares:,.0f}")
                elif implied_shares > 100_000_000_000:  # More than 100B shares seems high
                    errors.append(f"Implied shares outstanding very high: {implied_shares:,.0f}")

        return errors

    def _validate_no_placeholders(self, profile) -> List[str]:
        """Check for placeholder or null values in critical fields"""
        errors = []

        # Common placeholder indicators
        placeholder_values = [0, 0.0, '0.00', 'N/A', None, '', 'TBD', 'null']

        # Check deterministic forecasts if available
        if hasattr(profile, 'deterministic') and profile.deterministic:
            deterministic = profile.deterministic

            # Check revenue forecasts
            forecast = deterministic.get('forecast', {})
            revenue = forecast.get('revenue', {})

            for period, value in revenue.items():
                if value in placeholder_values:
                    errors.append(f"Revenue forecast for {period} is placeholder: {value}")

            # Check EPS forecasts
            eps = forecast.get('eps', {})
            for period, value in eps.items():
                if value in placeholder_values:
                    errors.append(f"EPS forecast for {period} is placeholder: {value}")

            # Check DCF value
            valuation = deterministic.get('valuation', {})
            dcf_value = valuation.get('dcf_value')
            if dcf_value in placeholder_values:
                errors.append(f"DCF valuation is placeholder: {dcf_value}")

        return errors

    def _calculate_validation_score(self, errors: List[str], warnings: List[str]) -> float:
        """Calculate overall validation score (0-100)"""
        base_score = 100

        # Deduct points for errors (critical)
        error_penalty = len(errors) * 25  # 25 points per error

        # Deduct points for warnings (less critical)
        warning_penalty = len(warnings) * 5  # 5 points per warning

        final_score = max(0, base_score - error_penalty - warning_penalty)
        return final_score

    def validate_data_freshness(self, ticker: str, data_date: Optional[datetime] = None) -> ValidationResult:
        """Validate that data is fresh enough for analysis"""
        errors = []
        warnings = []

        if not data_date:
            warnings.append("No data timestamp available for freshness check")
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                score=80
            )

        current_time = datetime.now()
        age_days = (current_time - data_date).days

        if age_days > 30:
            errors.append(f"Data is stale: {age_days} days old (>30 days)")
        elif age_days > self.validation_rules['data_freshness_days']:
            warnings.append(f"Data is somewhat stale: {age_days} days old")

        is_valid = len(errors) == 0
        score = max(0, 100 - (age_days * 2))  # 2 points per day of staleness

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=score
        )

    def validate_complete_report(self, report_content: str, metadata: Dict) -> ValidationResult:
        """Validate complete report for quality and completeness"""
        errors = []
        warnings = []

        # Check for missing sections
        required_sections = [
            "Executive Summary",
            "Financial Analysis",
            "Valuation",
            "Investment Thesis",
            "Risk Analysis"
        ]

        for section in required_sections:
            if section not in report_content:
                errors.append(f"Missing required section: {section}")

        # Check for error content
        error_phrases = [
            "temporarily unavailable",
            "$0.00",
            "Section generation",
            "Please ensure Ollama",
            "technical difficulties"
        ]

        for phrase in error_phrases:
            if phrase in report_content:
                errors.append(f"Error content found: {phrase}")

        # Check recommendation consistency
        import re
        recommendations = re.findall(r'\b(BUY|HOLD|SELL)\b', report_content)
        unique_recommendations = set(recommendations)

        if len(unique_recommendations) > 1:
            errors.append(f"Inconsistent recommendations found: {unique_recommendations}")
        elif len(unique_recommendations) == 0:
            errors.append("No investment recommendation found")

        # Check target price consistency
        target_prices = re.findall(r'\$(\d+\.?\d*)', report_content)
        if len(set(target_prices)) > 3:  # More than 3 different price targets seems excessive
            warnings.append("Multiple different target prices found - check consistency")

        # Check minimum word count
        word_count = len(report_content.split())
        if word_count < 5000:
            warnings.append(f"Report may be too short: {word_count} words (expected 5000+)")

        # Calculate score
        score = self._calculate_validation_score(errors, warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            score=score
        )

def validate_financial_data_pipeline(data_dict: Dict) -> ValidationResult:
    """Standalone function to validate financial data from pipeline"""
    validator = DataValidator()
    errors = []
    warnings = []

    # Check for empty or null data
    if not data_dict:
        errors.append("Empty data dictionary received")
        return ValidationResult(False, errors, warnings, 0)

    # Check required data fields
    required_fields = ['price_data', 'financial_statements', 'market_data']
    for field in required_fields:
        if field not in data_dict:
            errors.append(f"Missing required data field: {field}")

    # Validate price data
    price_data = data_dict.get('price_data', {})
    if price_data:
        current_price = price_data.get('current_price')
        if not current_price or current_price <= 0:
            errors.append("Invalid or missing current price in price data")

    # Calculate score
    score = validator._calculate_validation_score(errors, warnings)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        score=score
    )