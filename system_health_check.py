#!/usr/bin/env python3
"""
System Health Check for Enhanced Equity Research Tool (ERT)
Tests all critical components and validates fixes
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemHealthChecker:
    """Comprehensive system health checker"""

    def __init__(self):
        self.test_results = []
        self.overall_status = "HEALTHY"

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all system health checks"""
        logger.info("🏥 Starting ERT System Health Check...")

        checks = [
            ("Import Dependencies", self._check_imports),
            ("Ollama Engine Health", self._check_ollama_engine),
            ("OpenAI Engine Health", self._check_openai_engine),
            ("Data Validator", self._check_data_validator),
            ("Data Pipeline", self._check_data_pipeline),
            ("Report Generator", self._check_report_generator),
            ("File System Permissions", self._check_file_system),
            ("Configuration Files", self._check_configuration)
        ]

        for check_name, check_function in checks:
            try:
                logger.info(f"🔍 Running: {check_name}")
                result = check_function()
                self.test_results.append({
                    "name": check_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, str) else "Check completed"
                })
                if not result:
                    self.overall_status = "UNHEALTHY"
            except Exception as e:
                logger.error(f"❌ {check_name} failed: {e}")
                self.test_results.append({
                    "name": check_name,
                    "status": "ERROR",
                    "details": str(e)
                })
                self.overall_status = "UNHEALTHY"

        return self._generate_health_report()

    def _check_imports(self) -> bool:
        """Check if all required modules can be imported"""
        try:
            from src.utils.ollama_engine import OllamaEngine
            from src.utils.openai_engine import OpenAIEngine
            from src.utils.data_validator import DataValidator, DataIntegrityError
            from src.stock_report_generator import StockReportGenerator
            from src.data_pipeline import DataOrchestrator
            logger.info("✅ All imports successful")
            return True
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}")
            return False

    def _check_ollama_engine(self) -> bool:
        """Check Ollama engine functionality"""
        try:
            from src.utils.ollama_engine import OllamaEngine

            engine = OllamaEngine()
            connection_status = engine.test_connection()

            if connection_status:
                logger.info("✅ Ollama engine: Connected and ready")
                # Test basic generation
                test_response = engine.call_ollama("Test: What is 2+2?", max_tokens=20)
                if test_response and len(test_response) > 0:
                    logger.info("✅ Ollama generation test: Passed")
                    return True
                else:
                    logger.warning("⚠️ Ollama generation test: Failed")
                    return False
            else:
                logger.warning("⚠️ Ollama engine: Not available (this is OK if using OpenAI)")
                return True  # Not critical if OpenAI is available

        except Exception as e:
            logger.warning(f"⚠️ Ollama check failed: {e}")
            return True  # Not critical

    def _check_openai_engine(self) -> bool:
        """Check OpenAI engine functionality"""
        try:
            from src.utils.openai_engine import OpenAIEngine

            engine = OpenAIEngine()
            connection_status = engine.test_connection()

            if connection_status:
                logger.info("✅ OpenAI engine: Connected and ready")
                return True
            else:
                logger.warning("⚠️ OpenAI engine: Not available (check API key)")
                return False

        except Exception as e:
            logger.warning(f"⚠️ OpenAI check failed: {e}")
            return False

    def _check_data_validator(self) -> bool:
        """Check data validation functionality"""
        try:
            from src.utils.data_validator import DataValidator
            from src.stock_report_generator import CompanyProfile

            validator = DataValidator()

            # Create test company profile
            test_profile = CompanyProfile(
                ticker="AAPL",
                company_name="Apple Inc.",
                sector="Technology",
                industry="Consumer Electronics",
                market_cap=3000000000000,  # $3T
                current_price=150.0,
                target_price=160.0,
                recommendation="BUY",
                financial_metrics={
                    'pe_ratio': 25.0,
                    'revenue_growth': 8.5,
                    'profit_margin': 25.0,
                    'roe': 30.0
                },
                competitors=["MSFT", "GOOGL"],
                esg_data={},
                risk_factors=["Market competition", "Regulatory risks"],
                deterministic={}
            )

            validation_result = validator.validate_company_profile(test_profile)

            if validation_result.is_valid:
                logger.info(f"✅ Data validator: Passed (score: {validation_result.score:.1f}/100)")
                return True
            else:
                logger.error(f"❌ Data validator: Failed - {validation_result.errors}")
                return False

        except Exception as e:
            logger.error(f"❌ Data validator check failed: {e}")
            return False

    def _check_data_pipeline(self) -> bool:
        """Check data pipeline functionality"""
        try:
            from src.data_pipeline import DataOrchestrator

            orchestrator = DataOrchestrator()
            logger.info("✅ Data pipeline: Initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Data pipeline check failed: {e}")
            return False

    def _check_report_generator(self) -> bool:
        """Check report generator functionality"""
        try:
            from src.stock_report_generator import StockReportGenerator
            from src.utils.openai_engine import OpenAIEngine

            # Try with OpenAI first
            try:
                ai_engine = OpenAIEngine()
                if ai_engine.test_connection():
                    generator = StockReportGenerator(ai_engine)
                    logger.info("✅ Report generator: Initialized with OpenAI")
                    return True
            except:
                pass

            # Fallback to Ollama
            try:
                from src.utils.ollama_engine import OllamaEngine
                ai_engine = OllamaEngine()
                if ai_engine.test_connection():
                    generator = StockReportGenerator(ai_engine)
                    logger.info("✅ Report generator: Initialized with Ollama")
                    return True
            except:
                pass

            logger.error("❌ Report generator: No AI engine available")
            return False

        except Exception as e:
            logger.error(f"❌ Report generator check failed: {e}")
            return False

    def _check_file_system(self) -> bool:
        """Check file system permissions"""
        try:
            # Check reports directory
            reports_dir = os.path.join(project_root, "reports")
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
                logger.info("✅ Created reports directory")

            # Test write permissions
            test_file = os.path.join(reports_dir, "health_check_test.txt")
            with open(test_file, 'w') as f:
                f.write("Health check test")

            os.remove(test_file)
            logger.info("✅ File system: Write permissions OK")
            return True

        except Exception as e:
            logger.error(f"❌ File system check failed: {e}")
            return False

    def _check_configuration(self) -> bool:
        """Check configuration files"""
        try:
            config_checks = []

            # Check .env file
            env_file = os.path.join(project_root, ".env")
            if os.path.exists(env_file):
                config_checks.append("✅ .env file exists")
            else:
                config_checks.append("⚠️ .env file missing (optional)")

            # Check if API keys are set
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                config_checks.append("✅ OpenAI API key configured")
            else:
                config_checks.append("⚠️ OpenAI API key missing")

            logger.info("Configuration status: " + "; ".join(config_checks))
            return True

        except Exception as e:
            logger.error(f"❌ Configuration check failed: {e}")
            return False

    def _generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        errors = sum(1 for result in self.test_results if result["status"] == "ERROR")
        total = len(self.test_results)

        health_score = (passed / total) * 100 if total > 0 else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.overall_status,
            "health_score": health_score,
            "summary": {
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "errors": errors
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        failed_checks = [r for r in self.test_results if r["status"] in ["FAIL", "ERROR"]]

        for check in failed_checks:
            if "Ollama" in check["name"]:
                recommendations.append("Consider starting Ollama service: `ollama serve`")
            elif "OpenAI" in check["name"]:
                recommendations.append("Check OpenAI API key configuration in .env file")
            elif "Data Pipeline" in check["name"]:
                recommendations.append("Verify data source connections and permissions")
            elif "File System" in check["name"]:
                recommendations.append("Check write permissions for reports directory")

        if not recommendations:
            recommendations.append("System is healthy! No immediate actions required.")

        return recommendations

def main():
    """Main health check execution"""
    print("🚀 ERT System Health Check")
    print("=" * 50)

    checker = SystemHealthChecker()
    health_report = checker.run_all_checks()

    print(f"\n📊 HEALTH REPORT")
    print(f"Overall Status: {health_report['overall_status']}")
    print(f"Health Score: {health_report['health_score']:.1f}%")
    print(f"Checks Passed: {health_report['summary']['passed']}/{health_report['summary']['total_checks']}")

    if health_report['summary']['failed'] > 0 or health_report['summary']['errors'] > 0:
        print(f"\n❌ ISSUES DETECTED:")
        for result in health_report['detailed_results']:
            if result['status'] in ['FAIL', 'ERROR']:
                print(f"  - {result['name']}: {result['status']} - {result['details']}")

    print(f"\n💡 RECOMMENDATIONS:")
    for rec in health_report['recommendations']:
        print(f"  • {rec}")

    # Save detailed report
    report_file = os.path.join(project_root, "system_health_report.json")
    import json
    with open(report_file, 'w') as f:
        json.dump(health_report, f, indent=2)

    print(f"\n📝 Detailed report saved to: {report_file}")

    if health_report['overall_status'] == 'HEALTHY':
        print("\n🎉 System is ready for equity research report generation!")
        return 0
    else:
        print(f"\n⚠️ System needs attention before production use.")
        return 1

if __name__ == "__main__":
    exit(main())