#!/usr/bin/env python3
"""
Comprehensive Evaluation System for Finetuned Equity Research Models

This script implements quantitative and qualitative evaluation metrics
to assess the performance of finetuned Ollama models for equity research tasks.
"""

import os
import json
import logging
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import statistics
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    """Evaluation metric types"""
    FINANCIAL_ACCURACY = "financial_accuracy"
    INVESTMENT_REASONING = "investment_reasoning"
    TECHNICAL_DEPTH = "technical_depth"
    STRUCTURE_QUALITY = "structure_quality"
    FACTUAL_CONSISTENCY = "factual_consistency"
    RESPONSE_LENGTH = "response_length"
    PERPLEXITY = "perplexity"

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    metric: EvaluationMetric
    score: float
    max_score: float
    details: Dict
    timestamp: datetime

class EquityResearchEvaluator:
    """Comprehensive evaluation system for equity research models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

        # Financial terms and concepts for validation
        self.financial_keywords = {
            'valuation_metrics': ['P/E', 'EV/EBITDA', 'P/B', 'PEG', 'EV/Revenue', 'FCF yield'],
            'profitability_metrics': ['ROE', 'ROA', 'ROIC', 'gross margin', 'operating margin', 'net margin'],
            'liquidity_metrics': ['current ratio', 'quick ratio', 'cash ratio', 'working capital'],
            'leverage_metrics': ['debt-to-equity', 'debt-to-assets', 'interest coverage', 'EBITDA coverage'],
            'growth_metrics': ['revenue growth', 'earnings growth', 'FCF growth', 'CAGR'],
            'investment_terms': ['DCF', 'NPV', 'WACC', 'terminal value', 'discount rate', 'beta']
        }

        # Quality indicators for different report sections
        self.section_indicators = {
            'executive_summary': ['recommendation', 'price target', 'upside', 'catalyst', 'risk'],
            'financial_analysis': ['revenue', 'margin', 'cash flow', 'balance sheet', 'ratio'],
            'valuation': ['DCF', 'comparable', 'multiple', 'assumption', 'sensitivity'],
            'investment_thesis': ['bull case', 'bear case', 'catalyst', 'timeline', 'probability'],
            'risk_analysis': ['business risk', 'financial risk', 'regulatory', 'competition', 'mitigation']
        }

    def evaluate_financial_accuracy(self, response: str, expected_metrics: Dict) -> EvaluationResult:
        """Evaluate accuracy of financial metrics and calculations"""

        score = 0.0
        max_score = 100.0
        details = {
            'metrics_mentioned': 0,
            'calculations_present': False,
            'metric_accuracy': {},
            'terminology_usage': 0
        }

        # Check for financial metric mentions
        total_metrics = sum(len(metrics) for metrics in self.financial_keywords.values())
        mentioned_metrics = 0

        for category, metrics in self.financial_keywords.items():
            for metric in metrics:
                if metric.lower() in response.lower():
                    mentioned_metrics += 1
                    details['metric_accuracy'][metric] = True

        details['metrics_mentioned'] = mentioned_metrics
        details['terminology_usage'] = (mentioned_metrics / total_metrics) * 100

        # Check for calculation presence
        calculation_patterns = [
            r'\\d+\\.\\d+%',  # Percentages
            r'\\$\\d+[,\\d]*\\.?\\d*[MB]?',  # Dollar amounts
            r'\\d+\\.\\d+x',  # Multiples
            r'=\\s*\\d+',  # Equals calculations
        ]

        for pattern in calculation_patterns:
            if re.search(pattern, response):
                details['calculations_present'] = True
                break

        # Scoring algorithm
        terminology_score = min(details['terminology_usage'], 40.0)  # Max 40 points
        calculation_score = 30.0 if details['calculations_present'] else 0.0  # Max 30 points
        structure_score = 30.0 if len(response.split()) > 100 else 15.0  # Max 30 points

        score = terminology_score + calculation_score + structure_score

        return EvaluationResult(
            metric=EvaluationMetric.FINANCIAL_ACCURACY,
            score=score,
            max_score=max_score,
            details=details,
            timestamp=datetime.now()
        )

    def evaluate_investment_reasoning(self, response: str) -> EvaluationResult:
        """Evaluate quality of investment reasoning and recommendations"""

        score = 0.0
        max_score = 100.0
        details = {
            'has_recommendation': False,
            'has_price_target': False,
            'has_catalysts': False,
            'has_risks': False,
            'reasoning_depth': 0,
            'logical_structure': False
        }

        response_lower = response.lower()

        # Check for investment recommendation
        rec_keywords = ['buy', 'sell', 'hold', 'strong buy', 'strong sell', 'overweight', 'underweight']
        details['has_recommendation'] = any(keyword in response_lower for keyword in rec_keywords)

        # Check for price target
        price_patterns = [
            r'price target.*\\$\\d+',
            r'target price.*\\$\\d+',
            r'\\$\\d+.*target',
            r'fair value.*\\$\\d+'
        ]
        details['has_price_target'] = any(re.search(pattern, response_lower) for pattern in price_patterns)

        # Check for catalysts
        catalyst_keywords = ['catalyst', 'driver', 'opportunity', 'growth factor', 'upside']
        details['has_catalysts'] = any(keyword in response_lower for keyword in catalyst_keywords)

        # Check for risks
        risk_keywords = ['risk', 'threat', 'challenge', 'headwind', 'downside', 'concern']
        details['has_risks'] = any(keyword in response_lower for keyword in risk_keywords)

        # Evaluate reasoning depth
        reasoning_indicators = [
            'because', 'due to', 'as a result', 'given that', 'considering',
            'based on', 'evidence suggests', 'analysis shows', 'data indicates'
        ]
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        details['reasoning_depth'] = min(reasoning_count, 5)  # Cap at 5

        # Check logical structure
        sections = ['summary', 'analysis', 'conclusion', 'recommendation']
        structure_score = sum(1 for section in sections if section in response_lower)
        details['logical_structure'] = structure_score >= 2

        # Scoring algorithm
        component_scores = {
            'recommendation': 25.0 if details['has_recommendation'] else 0.0,
            'price_target': 20.0 if details['has_price_target'] else 0.0,
            'catalysts': 20.0 if details['has_catalysts'] else 0.0,
            'risks': 15.0 if details['has_risks'] else 0.0,
            'reasoning': details['reasoning_depth'] * 3.0,  # Max 15 points
            'structure': 15.0 if details['logical_structure'] else 0.0
        }

        score = sum(component_scores.values())

        return EvaluationResult(
            metric=EvaluationMetric.INVESTMENT_REASONING,
            score=score,
            max_score=max_score,
            details={**details, 'component_scores': component_scores},
            timestamp=datetime.now()
        )

    def evaluate_technical_depth(self, response: str) -> EvaluationResult:
        """Evaluate technical depth and sophistication of analysis"""

        score = 0.0
        max_score = 100.0
        details = {
            'advanced_concepts': 0,
            'quantitative_analysis': False,
            'industry_knowledge': 0,
            'methodology_mentioned': False,
            'comparative_analysis': False
        }

        response_lower = response.lower()

        # Advanced financial concepts
        advanced_concepts = [
            'dcf', 'wacc', 'capm', 'beta', 'risk premium', 'terminal value',
            'monte carlo', 'sensitivity analysis', 'scenario analysis',
            'sum of parts', 'nav', 'replacement cost', 'liquidation value'
        ]
        concept_count = sum(1 for concept in advanced_concepts if concept in response_lower)
        details['advanced_concepts'] = min(concept_count, 8)

        # Quantitative analysis indicators
        quant_patterns = [
            r'\\d+%.*growth',
            r'margin.*\\d+%',
            r'\\d+x.*multiple',
            r'\\d+\\.\\d+.*ratio',
            r'\\$\\d+.*billion'
        ]
        details['quantitative_analysis'] = any(re.search(pattern, response_lower) for pattern in quant_patterns)

        # Industry knowledge indicators
        industry_terms = [
            'market share', 'competitive advantage', 'barriers to entry',
            'switching costs', 'network effects', 'economies of scale',
            'regulatory environment', 'industry dynamics', 'secular trends'
        ]
        industry_count = sum(1 for term in industry_terms if term in response_lower)
        details['industry_knowledge'] = min(industry_count, 5)

        # Methodology mentions
        methodology_terms = ['methodology', 'approach', 'framework', 'model', 'assumptions']
        details['methodology_mentioned'] = any(term in response_lower for term in methodology_terms)

        # Comparative analysis
        comparison_terms = ['compared to', 'versus', 'relative to', 'peer group', 'benchmark']
        details['comparative_analysis'] = any(term in response_lower for term in comparison_terms)

        # Scoring algorithm
        component_scores = {
            'advanced_concepts': details['advanced_concepts'] * 4.0,  # Max 32 points
            'quantitative': 20.0 if details['quantitative_analysis'] else 0.0,
            'industry_knowledge': details['industry_knowledge'] * 4.0,  # Max 20 points
            'methodology': 15.0 if details['methodology_mentioned'] else 0.0,
            'comparative': 13.0 if details['comparative_analysis'] else 0.0
        }

        score = sum(component_scores.values())

        return EvaluationResult(
            metric=EvaluationMetric.TECHNICAL_DEPTH,
            score=score,
            max_score=max_score,
            details={**details, 'component_scores': component_scores},
            timestamp=datetime.now()
        )

    def evaluate_structure_quality(self, response: str) -> EvaluationResult:
        """Evaluate structure and organization of the response"""

        score = 0.0
        max_score = 100.0
        details = {
            'has_headers': False,
            'has_bullet_points': False,
            'paragraph_count': 0,
            'appropriate_length': False,
            'clear_sections': 0,
            'professional_format': False
        }

        # Check for headers
        header_patterns = [r'^#+\\s+', r'^[A-Z][^\\n]*:$', r'^\\*\\*[^\\*]+\\*\\*']
        details['has_headers'] = any(re.search(pattern, response, re.MULTILINE) for pattern in header_patterns)

        # Check for bullet points
        bullet_patterns = [r'^\\s*[•\\-\\*]\\s+', r'^\\s*\\d+\\.\\s+']
        details['has_bullet_points'] = any(re.search(pattern, response, re.MULTILINE) for pattern in bullet_patterns)

        # Count paragraphs
        paragraphs = [p.strip() for p in response.split('\\n\\n') if p.strip()]
        details['paragraph_count'] = len(paragraphs)

        # Check length appropriateness
        word_count = len(response.split())
        details['appropriate_length'] = 150 <= word_count <= 2000

        # Check for clear sections
        section_indicators = ['summary', 'analysis', 'valuation', 'recommendation', 'risks', 'conclusion']
        section_count = sum(1 for indicator in section_indicators if indicator in response.lower())
        details['clear_sections'] = min(section_count, 6)

        # Professional formatting
        professional_indicators = [
            details['has_headers'],
            details['has_bullet_points'],
            details['paragraph_count'] >= 3,
            '**' in response or '*' in response,  # Bold/italic formatting
            '$' in response  # Financial figures
        ]
        details['professional_format'] = sum(professional_indicators) >= 3

        # Scoring algorithm
        component_scores = {
            'headers': 20.0 if details['has_headers'] else 0.0,
            'bullets': 15.0 if details['has_bullet_points'] else 0.0,
            'paragraphs': min(details['paragraph_count'] * 3.0, 15.0),  # Max 15 points
            'length': 20.0 if details['appropriate_length'] else 0.0,
            'sections': details['clear_sections'] * 3.0,  # Max 18 points
            'professional': 12.0 if details['professional_format'] else 0.0
        }

        score = sum(component_scores.values())

        return EvaluationResult(
            metric=EvaluationMetric.STRUCTURE_QUALITY,
            score=score,
            max_score=max_score,
            details={**details, 'component_scores': component_scores},
            timestamp=datetime.now()
        )

    def run_model_evaluation(self, prompt: str) -> str:
        """Run the model with a given prompt and return response"""

        try:
            cmd = f"ollama run {self.model_name} '{prompt}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"Model execution failed: {result.stderr}")
                return ""

        except subprocess.TimeoutExpired:
            logger.error("Model execution timed out")
            return ""
        except Exception as e:
            logger.error(f"Error running model: {e}")
            return ""

    def comprehensive_evaluation(self, test_prompts: List[str]) -> Dict:
        """Run comprehensive evaluation across multiple prompts"""

        all_results = []
        prompt_responses = []

        logger.info(f"Starting comprehensive evaluation of {self.model_name}")

        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"Evaluating prompt {i}/{len(test_prompts)}")

            # Get model response
            response = self.run_model_evaluation(prompt)

            if not response:
                logger.warning(f"Empty response for prompt {i}")
                continue

            prompt_responses.append({
                'prompt': prompt,
                'response': response,
                'word_count': len(response.split())
            })

            # Run all evaluation metrics
            evaluations = [
                self.evaluate_financial_accuracy(response, {}),
                self.evaluate_investment_reasoning(response),
                self.evaluate_technical_depth(response),
                self.evaluate_structure_quality(response)
            ]

            all_results.extend(evaluations)

        # Aggregate results by metric
        metric_scores = {}
        for result in all_results:
            metric_name = result.metric.value
            if metric_name not in metric_scores:
                metric_scores[metric_name] = []
            metric_scores[metric_name].append(result.score / result.max_score * 100)

        # Calculate summary statistics
        summary_stats = {}
        for metric, scores in metric_scores.items():
            summary_stats[metric] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }

        # Overall score (weighted average)
        weights = {
            'financial_accuracy': 0.25,
            'investment_reasoning': 0.30,
            'technical_depth': 0.25,
            'structure_quality': 0.20
        }

        overall_score = sum(
            weights.get(metric, 0.25) * stats['mean']
            for metric, stats in summary_stats.items()
        )

        evaluation_report = {
            'model_name': self.model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_prompts_count': len(test_prompts),
            'successful_responses': len(prompt_responses),
            'overall_score': overall_score,
            'metric_scores': summary_stats,
            'detailed_results': [
                {
                    'metric': result.metric.value,
                    'score': result.score,
                    'max_score': result.max_score,
                    'percentage': result.score / result.max_score * 100,
                    'details': result.details
                }
                for result in all_results
            ],
            'prompt_responses': prompt_responses
        }

        return evaluation_report

    def save_evaluation_report(self, report: Dict, filename: Optional[str] = None) -> str:
        """Save evaluation report to file"""

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.model_name}_evaluation_{timestamp}.json"

        report_path = self.results_dir / filename

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to {report_path}")
        return str(report_path)

    def create_evaluation_summary(self, report: Dict) -> str:
        """Create human-readable evaluation summary"""

        summary = f"""
# EVALUATION REPORT - {report['model_name'].upper()}

## Overall Performance
**Score: {report['overall_score']:.1f}/100**

## Metric Breakdown
"""

        for metric, stats in report['metric_scores'].items():
            metric_name = metric.replace('_', ' ').title()
            summary += f"### {metric_name}\n- **Average Score**: {stats['mean']:.1f}/100\n- **Range**: {stats['min']:.1f} - {stats['max']:.1f}\n- **Standard Deviation**: {stats['std']:.1f}\n\n"

        summary += f"""
## Response Quality Analysis
- **Total Prompts Tested**: {report['test_prompts_count']}
- **Successful Responses**: {report['successful_responses']}
- **Success Rate**: {(report['successful_responses']/report['test_prompts_count']*100):.1f}%

## Recommendations
"""

        # Generate recommendations based on scores
        overall_score = report['overall_score']

        if overall_score >= 80:
            summary += "✅ **Excellent Performance** - Model demonstrates strong equity research capabilities\n"
        elif overall_score >= 65:
            summary += "⚠️ **Good Performance** - Model shows solid capabilities with room for improvement\n"
        elif overall_score >= 50:
            summary += "🔧 **Moderate Performance** - Additional training recommended\n"
        else:
            summary += "❌ **Poor Performance** - Significant retraining required\n"

        # Specific recommendations
        financial_score = report['metric_scores'].get('financial_accuracy', {}).get('mean', 0)
        reasoning_score = report['metric_scores'].get('investment_reasoning', {}).get('mean', 0)
        technical_score = report['metric_scores'].get('technical_depth', {}).get('mean', 0)

        if financial_score < 60:
            summary += "- Focus training on financial metrics and calculations\n"
        if reasoning_score < 60:
            summary += "- Improve investment reasoning and recommendation structure\n"
        if technical_score < 60:
            summary += "- Enhance technical depth and industry knowledge\n"

        summary += f"\n**Evaluation Date**: {report['evaluation_timestamp']}\n"

        return summary

def create_standard_test_prompts() -> List[str]:
    """Create standard test prompts for equity research evaluation"""

    return [
        "Generate an executive summary for Apple Inc. (AAPL) with a BUY recommendation and $200 price target.",
        "Analyze Microsoft's financial performance focusing on cloud revenue growth and profitability metrics.",
        "Conduct a DCF valuation for Tesla with assumptions about EV market growth and margin expansion.",
        "Assess the investment risks for NVIDIA in the AI semiconductor market, including competition and regulation.",
        "Compare Amazon's e-commerce and AWS segments, highlighting growth drivers and margin profiles.",
        "Evaluate Google's competitive position in digital advertising and cloud computing markets.",
        "Analyze Johnson & Johnson's pharmaceutical pipeline and assess patent cliff risks.",
        "Examine Meta's transition to the metaverse and its impact on long-term growth prospects."
    ]

def main():
    """Main evaluation workflow"""

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Finetuned Equity Research Model")
    parser.add_argument('model_name', help='Name of the Ollama model to evaluate')
    parser.add_argument('--custom-prompts', type=str, help='Path to custom test prompts JSON file')
    parser.add_argument('--output', type=str, help='Output filename for evaluation report')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = EquityResearchEvaluator(args.model_name)

    # Load test prompts
    if args.custom_prompts and Path(args.custom_prompts).exists():
        with open(args.custom_prompts, 'r') as f:
            test_data = json.load(f)
            test_prompts = test_data.get('prompts', [])
    else:
        test_prompts = create_standard_test_prompts()

    print(f"🧪 Evaluating model: {args.model_name}")
    print(f"📝 Test prompts: {len(test_prompts)}")
    print()

    # Run comprehensive evaluation
    report = evaluator.comprehensive_evaluation(test_prompts)

    # Save detailed report
    report_path = evaluator.save_evaluation_report(report, args.output)

    # Create and save summary
    summary = evaluator.create_evaluation_summary(report)
    summary_path = evaluator.results_dir / f"{args.model_name}_summary.md"

    with open(summary_path, 'w') as f:
        f.write(summary)

    # Display results
    print("🎯 EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print()
    print("📊 Metric Scores:")
    for metric, stats in report['metric_scores'].items():
        print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.1f}/100")
    print()
    print(f"📄 Detailed Report: {report_path}")
    print(f"📋 Summary: {summary_path}")

if __name__ == "__main__":
    main()