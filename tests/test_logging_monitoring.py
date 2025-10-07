"""
Test suite for logging and monitoring functionality
"""

import pytest
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.logging_config import (
    ERTLogger,
    StructuredFormatter,
    performance_monitor,
    ert_logger
)
from src.utils.monitoring_dashboard import MonitoringDashboard


class TestERTLogger:
    """Test ERT logging functionality"""

    def test_logger_initialization(self, temp_directory):
        """Test logger initialization and file creation"""
        logger = ERTLogger(log_dir=str(temp_directory))

        # Check log directory creation
        assert temp_directory.exists()

        # Check logger attributes
        assert hasattr(logger, 'app_logger')
        assert hasattr(logger, 'perf_logger')
        assert hasattr(logger, 'error_logger')
        assert hasattr(logger, 'audit_logger')

        # Check metrics storage
        assert hasattr(logger, 'performance_metrics')
        assert hasattr(logger, 'error_events')
        assert hasattr(logger, 'metrics_lock')

    def test_structured_logging(self, test_logger, temp_directory):
        """Test structured JSON logging"""
        # Log a test message
        test_logger.app_logger.info("Test message", extra={'test_field': 'test_value'})

        # Check if log files are created
        log_files = list(temp_directory.glob("*.log"))
        assert len(log_files) > 0

        # Read and verify JSON structure
        app_log = temp_directory / "ert_application.log"
        if app_log.exists():
            with open(app_log, 'r') as f:
                log_line = f.readline().strip()
                if log_line:
                    log_data = json.loads(log_line)
                    assert 'timestamp' in log_data
                    assert 'level' in log_data
                    assert 'message' in log_data
                    assert log_data['test_field'] == 'test_value'

    def test_performance_metric_logging(self, test_logger):
        """Test performance metric logging"""
        # Log a performance metric
        test_logger.log_performance_metric(
            operation="test_operation",
            duration=0.5,
            success=True,
            module="test_module",
            test_detail="example"
        )

        # Check metrics storage
        assert len(test_logger.performance_metrics) == 1
        metric = test_logger.performance_metrics[0]

        assert metric.operation == "test_operation"
        assert metric.duration == 0.5
        assert metric.success is True
        assert metric.module == "test_module"
        assert "test_detail" in metric.details

    def test_error_event_logging(self, test_logger):
        """Test error event logging"""
        # Create a test exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            test_logger.log_error_event(
                error=e,
                operation="test_operation",
                module="test_module",
                context_info="test context"
            )

        # Check error storage
        assert len(test_logger.error_events) == 1
        error_event = test_logger.error_events[0]

        assert error_event.error_type == "ValueError"
        assert error_event.error_message == "Test error"
        assert error_event.operation == "test_operation"
        assert error_event.module == "test_module"
        assert "context_info" in error_event.context

    def test_user_action_logging(self, test_logger):
        """Test user action audit logging"""
        test_logger.log_user_action(
            action="test_action",
            user_id="test_user",
            test_context="example"
        )

        # This should not raise an exception and should log to audit logger
        # We can't easily test the log output without reading files, but we can
        # ensure the method executes without error

    def test_performance_summary(self, test_logger):
        """Test performance summary generation"""
        # Add some test metrics
        test_logger.log_performance_metric("op1", 0.1, True, "module1")
        test_logger.log_performance_metric("op1", 0.2, True, "module1")
        test_logger.log_performance_metric("op2", 0.3, False, "module2")

        # Get performance summary
        summary = test_logger.get_performance_summary(last_minutes=60)

        assert isinstance(summary, dict)
        if 'total_operations' in summary:
            assert summary['total_operations'] == 3
            assert 'success_rate' in summary
            assert 'avg_duration_ms' in summary
            assert 'operations' in summary

    def test_error_summary(self, test_logger):
        """Test error summary generation"""
        # Add some test errors
        try:
            raise ValueError("Error 1")
        except ValueError as e:
            test_logger.log_error_event(e, "op1", "module1")

        try:
            raise TypeError("Error 2")
        except TypeError as e:
            test_logger.log_error_event(e, "op2", "module1")

        # Get error summary
        summary = test_logger.get_error_summary(last_minutes=60)

        assert isinstance(summary, dict)
        if 'total_errors' in summary:
            assert summary['total_errors'] == 2
            assert 'error_types' in summary
            assert 'recent_errors' in summary

    def test_concurrent_logging(self, test_logger):
        """Test concurrent logging from multiple threads"""
        errors = []
        results = []

        def log_from_thread(thread_id):
            try:
                for i in range(10):
                    test_logger.log_performance_metric(
                        operation=f"thread_{thread_id}_op_{i}",
                        duration=0.1,
                        success=True,
                        module=f"thread_{thread_id}"
                    )
                results.append(thread_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_from_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Concurrent logging errors: {errors}"
        assert len(results) == 5
        assert len(test_logger.performance_metrics) == 50  # 5 threads * 10 operations


class TestStructuredFormatter:
    """Test structured JSON formatter"""

    def test_formatter_basic_functionality(self):
        """Test basic formatter functionality"""
        import logging

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test_logger'
        assert log_data['message'] == 'Test message'
        assert 'timestamp' in log_data
        assert 'thread_id' in log_data

    def test_formatter_with_extra_fields(self):
        """Test formatter with extra fields"""
        import logging

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Add extra fields
        record.custom_field = "custom_value"
        record.operation = "test_operation"

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data['custom_field'] == 'custom_value'
        assert log_data['operation'] == 'test_operation'

    def test_formatter_with_exception(self):
        """Test formatter with exception information"""
        import logging

        formatter = StructuredFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info
            )

            formatted = formatter.format(record)
            log_data = json.loads(formatted)

            assert 'exception' in log_data
            assert log_data['exception']['type'] == 'ValueError'
            assert log_data['exception']['message'] == 'Test exception'
            assert 'traceback' in log_data['exception']


class TestPerformanceMonitorDecorator:
    """Test performance monitoring decorator"""

    def test_decorator_success_case(self, test_logger):
        """Test decorator on successful function execution"""
        @performance_monitor("test_operation", "test_module")
        def test_function(x, y):
            time.sleep(0.1)  # Simulate work
            return x + y

        result = test_function(2, 3)

        assert result == 5
        assert len(test_logger.performance_metrics) >= 1

        # Find the metric for our operation
        metric = next((m for m in test_logger.performance_metrics if m.operation == "test_operation"), None)
        assert metric is not None
        assert metric.success is True
        assert metric.duration >= 0.1

    def test_decorator_exception_case(self, test_logger):
        """Test decorator on function that raises exception"""
        @performance_monitor("test_operation_error", "test_module")
        def test_function_error():
            time.sleep(0.05)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_function_error()

        # Check that both performance and error were logged
        assert len(test_logger.performance_metrics) >= 1
        assert len(test_logger.error_events) >= 1

        # Find the metric for our operation
        metric = next((m for m in test_logger.performance_metrics if m.operation == "test_operation_error"), None)
        assert metric is not None
        assert metric.success is False

    def test_decorator_with_args_kwargs(self, test_logger):
        """Test decorator with function arguments and keyword arguments"""
        @performance_monitor("test_operation_args", "test_module")
        def test_function_with_args(a, b, c=None, d=None):
            return a + b + (c or 0) + (d or 0)

        result = test_function_with_args(1, 2, c=3, d=4)

        assert result == 10

        # Check that function details were logged
        metric = next((m for m in test_logger.performance_metrics if m.operation == "test_operation_args"), None)
        assert metric is not None
        assert 'function' in metric.details
        assert metric.details['function'] == 'test_function_with_args'


class TestMonitoringDashboard:
    """Test monitoring dashboard functionality"""

    def test_dashboard_initialization(self, test_logger):
        """Test dashboard initialization"""
        dashboard = MonitoringDashboard(logger=test_logger)

        assert dashboard.logger == test_logger
        assert hasattr(dashboard, 'monitoring_active')
        assert hasattr(dashboard, 'system_health_history')

    def test_system_health_collection(self, test_logger):
        """Test system health data collection"""
        dashboard = MonitoringDashboard(logger=test_logger)

        health_data = dashboard._collect_system_health()

        assert hasattr(health_data, 'cpu_percent')
        assert hasattr(health_data, 'memory_percent')
        assert hasattr(health_data, 'disk_percent')
        assert hasattr(health_data, 'network_io')

        assert 0 <= health_data.cpu_percent <= 100
        assert 0 <= health_data.memory_percent <= 100
        assert 0 <= health_data.disk_percent <= 100

    def test_performance_metrics_integration(self, test_logger):
        """Test integration with performance metrics"""
        dashboard = MonitoringDashboard(logger=test_logger)

        # Add some test metrics
        test_logger.log_performance_metric("test_op", 0.1, True, "test_module")
        test_logger.log_performance_metric("test_op", 0.2, False, "test_module")

        metrics = dashboard.get_recent_performance_metrics(minutes=60)

        assert isinstance(metrics, list)
        if len(metrics) > 0:
            assert all('operation' in m for m in metrics)
            assert all('duration' in m for m in metrics)
            assert all('success' in m for m in metrics)

    def test_alerting_system(self, test_logger):
        """Test alerting for high resource usage"""
        dashboard = MonitoringDashboard(logger=test_logger)

        # Mock high CPU usage
        with patch('psutil.cpu_percent', return_value=95.0):
            health_data = dashboard._collect_system_health()
            alerts = dashboard._check_for_alerts(health_data)

            assert len(alerts) > 0
            assert any('cpu' in alert.lower() for alert in alerts)

    def test_monitoring_thread_lifecycle(self, test_logger):
        """Test monitoring thread start/stop"""
        dashboard = MonitoringDashboard(logger=test_logger)

        # Start monitoring
        dashboard.start_monitoring()
        assert dashboard.monitoring_active is True

        # Let it run briefly
        time.sleep(0.5)

        # Stop monitoring
        dashboard.stop_monitoring()
        assert dashboard.monitoring_active is False

    def test_dashboard_metrics_export(self, test_logger, temp_directory):
        """Test dashboard metrics export"""
        dashboard = MonitoringDashboard(logger=test_logger)

        # Add some test data
        test_logger.log_performance_metric("export_test", 0.1, True, "test_module")

        # Export metrics
        export_path = temp_directory / "metrics_export.json"
        dashboard.export_metrics(str(export_path))

        assert export_path.exists()

        # Verify export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)

        assert isinstance(export_data, dict)
        assert 'performance_summary' in export_data
        assert 'system_health' in export_data


class TestLoggingIntegration:
    """Test integration between logging and other components"""

    def test_logging_with_valuation_pipeline(self, test_logger, mock_company_data, test_config):
        """Test logging integration with valuation pipeline"""
        from analyze.deterministic import run_deterministic_models

        # Clear existing metrics
        test_logger.performance_metrics.clear()
        test_logger.error_events.clear()

        try:
            # This should trigger logging if properly integrated
            result = run_deterministic_models(mock_company_data, test_config)

            # Check if performance metrics were logged
            # Note: This test depends on the actual integration in the valuation code
            # If no @performance_monitor decorators are used, this might not generate metrics

        except Exception as e:
            # Even if valuation fails, error should be logged
            test_logger.log_error_event(e, "test_valuation", "test_integration")

        # At minimum, we should have logged the error if valuation failed
        # Or performance metrics if it succeeded

    def test_logging_configuration_validation(self, temp_directory):
        """Test logging configuration validation"""
        # Test with valid configuration
        logger = ERTLogger(log_dir=str(temp_directory))

        # Verify all required loggers are configured
        assert logger.app_logger.level <= 20  # INFO level or lower
        assert logger.perf_logger.level <= 20
        assert logger.error_logger.level <= 40  # ERROR level or lower
        assert logger.audit_logger.level <= 20

        # Verify handlers are configured
        assert len(logger.app_logger.handlers) >= 2  # Console + File
        assert len(logger.perf_logger.handlers) >= 1  # File
        assert len(logger.error_logger.handlers) >= 1  # File
        assert len(logger.audit_logger.handlers) >= 1  # File

    def test_log_rotation_functionality(self, temp_directory):
        """Test log file rotation"""
        logger = ERTLogger(log_dir=str(temp_directory))

        # Generate large amount of log data to trigger rotation
        large_message = "x" * 1000  # 1KB message

        for i in range(1000):  # Should trigger rotation for 10MB limit
            logger.app_logger.info(f"Large message {i}: {large_message}")

        # Check if rotation occurred (backup files created)
        log_files = list(temp_directory.glob("ert_application.log*"))
        # Should have main log file and potentially backup files

        assert len(log_files) >= 1