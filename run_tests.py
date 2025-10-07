#!/usr/bin/env python3
"""
Automated test runner for ERT system
Provides comprehensive testing with reporting and metrics
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging_config import ert_logger


class TestRunner:
    """Enhanced test runner with logging and reporting"""

    def __init__(self, log_results=True):
        self.log_results = log_results
        self.start_time = None
        self.results = {}

    def run_test_suite(self, test_type="all", verbose=False, markers=None):
        """Run test suite with specified parameters"""
        self.start_time = datetime.now(timezone.utc)

        if self.log_results:
            ert_logger.log_user_action(
                action="test_suite_started",
                test_type=test_type,
                verbose=verbose,
                markers=markers
            )

        try:
            # Build pytest command
            cmd = ["python", "-m", "pytest"]

            if verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")

            # Add test paths based on type
            if test_type == "unit":
                cmd.extend(["-m", "unit", "tests/"])
            elif test_type == "integration":
                cmd.extend(["-m", "integration", "tests/"])
            elif test_type == "performance":
                cmd.extend(["-m", "performance", "tests/"])
            elif test_type == "fast":
                cmd.extend(["-m", "not slow", "tests/"])
            elif test_type == "all":
                cmd.append("tests/")
            else:
                cmd.append(f"tests/test_{test_type}.py")

            # Add custom markers if specified
            if markers:
                cmd.extend(["-m", markers])

            # Add output options
            cmd.extend([
                "--tb=short",
                "--strict-markers",
                "--strict-config"
            ])

            print(f"🧪 Running {test_type} tests...")
            print(f"Command: {' '.join(cmd)}")
            print("=" * 80)

            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Process results
            self.results = {
                "test_type": test_type,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                "timestamp": self.start_time.isoformat()
            }

            # Log results
            if self.log_results:
                success = result.returncode == 0
                ert_logger.log_performance_metric(
                    operation=f"test_suite_{test_type}",
                    duration=self.results["duration"],
                    success=success,
                    module="test_runner",
                    test_count=self._count_tests(result.stdout),
                    failed_count=self._count_failures(result.stdout)
                )

                ert_logger.log_user_action(
                    action="test_suite_completed",
                    test_type=test_type,
                    success=success,
                    duration=self.results["duration"]
                )

            return self.results

        except subprocess.TimeoutExpired:
            error_msg = f"Test suite {test_type} timed out after 5 minutes"
            print(f"❌ {error_msg}")

            if self.log_results:
                ert_logger.log_error_event(
                    error=TimeoutError(error_msg),
                    operation="test_suite_timeout",
                    module="test_runner",
                    test_type=test_type
                )

            return {
                "test_type": test_type,
                "return_code": -1,
                "error": "timeout",
                "duration": 300
            }

        except Exception as e:
            print(f"❌ Test execution failed: {e}")

            if self.log_results:
                ert_logger.log_error_event(
                    error=e,
                    operation="test_suite_execution",
                    module="test_runner",
                    test_type=test_type
                )

            return {
                "test_type": test_type,
                "return_code": -1,
                "error": str(e),
                "duration": 0
            }

    def _count_tests(self, output):
        """Count total tests from pytest output"""
        lines = output.split('\n')
        for line in lines:
            if ' passed' in line or ' failed' in line:
                # Look for pattern like "5 passed, 2 failed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part in ['passed', 'failed', 'error', 'skipped']:
                        try:
                            return int(parts[i-1])
                        except (ValueError, IndexError):
                            continue
        return 0

    def _count_failures(self, output):
        """Count failed tests from pytest output"""
        lines = output.split('\n')
        for line in lines:
            if ' failed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'failed':
                        try:
                            return int(parts[i-1])
                        except (ValueError, IndexError):
                            continue
        return 0

    def print_results(self):
        """Print formatted test results"""
        if not self.results:
            print("❌ No test results available")
            return

        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        test_type = self.results.get("test_type", "unknown")
        return_code = self.results.get("return_code", -1)
        duration = self.results.get("duration", 0)

        if return_code == 0:
            print(f"✅ {test_type.upper()} TESTS PASSED")
        else:
            print(f"❌ {test_type.upper()} TESTS FAILED")

        print(f"Duration: {duration:.2f} seconds")
        print(f"Return code: {return_code}")

        # Print stdout if available
        stdout = self.results.get("stdout", "")
        if stdout:
            print("\n📝 Test Output:")
            print("-" * 40)
            # Print last few lines of output
            lines = stdout.strip().split('\n')
            for line in lines[-10:]:  # Last 10 lines
                print(line)

        # Print stderr if there are errors
        stderr = self.results.get("stderr", "")
        if stderr and return_code != 0:
            print("\n🚨 Error Output:")
            print("-" * 40)
            print(stderr)

    def save_results(self, output_file="test_results.json"):
        """Save test results to file"""
        if not self.results:
            return

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"💾 Results saved to {output_path}")


def run_comprehensive_tests():
    """Run comprehensive test suite with all test types"""
    print("🚀 Starting Comprehensive ERT Test Suite")
    print("=" * 80)

    runner = TestRunner(log_results=True)
    all_results = {}

    test_suites = [
        ("data_pipeline", "Data Pipeline Tests"),
        ("valuation_models", "Valuation Model Tests"),
        ("logging_monitoring", "Logging & Monitoring Tests"),
        ("integration", "Integration Tests")
    ]

    total_passed = 0
    total_failed = 0

    for test_name, description in test_suites:
        print(f"\n🔍 Running {description}...")

        try:
            result = runner.run_test_suite(test_type=test_name, verbose=True)
            all_results[test_name] = result

            if result.get("return_code") == 0:
                print(f"✅ {description} - PASSED")
                total_passed += 1
            else:
                print(f"❌ {description} - FAILED")
                total_failed += 1

        except Exception as e:
            print(f"💥 {description} - CRASHED: {e}")
            all_results[test_name] = {"error": str(e), "return_code": -1}
            total_failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📊 Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"logs/comprehensive_test_results_{timestamp}.json"
    Path("logs").mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"💾 Comprehensive results saved to {results_file}")

    # Return overall success
    return total_failed == 0


def main():
    """Main test runner CLI"""
    parser = argparse.ArgumentParser(description="ERT Test Runner")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "performance", "fast", "comprehensive",
                "data_pipeline", "valuation_models", "logging_monitoring"],
        help="Type of tests to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-m", "--markers", help="Custom pytest markers")
    parser.add_argument("--save", help="Save results to file")
    parser.add_argument("--no-log", action="store_true", help="Disable logging")

    args = parser.parse_args()

    if args.test_type == "comprehensive":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)

    # Run single test suite
    runner = TestRunner(log_results=not args.no_log)

    try:
        results = runner.run_test_suite(
            test_type=args.test_type,
            verbose=args.verbose,
            markers=args.markers
        )

        runner.print_results()

        if args.save:
            runner.save_results(args.save)

        # Exit with test result code
        sys.exit(results.get("return_code", 1))

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Test runner crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()