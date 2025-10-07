"""
Comprehensive Logging and Monitoring System for ERT
Provides structured logging, performance metrics, and monitoring capabilities
"""

import logging
import logging.handlers
import json
import time
import functools
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
import traceback
import sys

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    operation: str
    duration: float
    timestamp: datetime
    success: bool
    details: Dict[str, Any]
    module: str
    thread_id: str

@dataclass
class ErrorEvent:
    """Error event data structure"""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    module: str
    operation: str
    context: Dict[str, Any]

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def format(self, record):
        # Create base log structure
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': getattr(record, 'module', record.name),
            'thread_id': threading.current_thread().name,
            'process_id': record.process
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        # Add extra fields if present
        for key, value in record.__dict__.items():
            if (key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']
                and key not in log_data):  # Avoid overwriting existing keys
                log_data[key] = value

        return json.dumps(log_data, default=str)

class ERTLogger:
    """Enhanced logging system for ERT with monitoring capabilities"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Performance metrics storage
        self.performance_metrics: List[PerformanceMetric] = []
        self.error_events: List[ErrorEvent] = []
        self.metrics_lock = threading.Lock()

        # Configure loggers
        self._setup_loggers()

        # Start metrics collection
        self.start_time = datetime.now(timezone.utc)

    def _setup_loggers(self):
        """Setup structured logging configuration"""

        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        json_formatter = StructuredFormatter()

        # Main application logger
        self.app_logger = logging.getLogger('ert')
        self.app_logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.app_logger.addHandler(console_handler)

        # File handler - rotating logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'ert_application.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(json_formatter)
        self.app_logger.addHandler(file_handler)

        # Performance logger
        self.perf_logger = logging.getLogger('ert.performance')
        self.perf_logger.setLevel(logging.INFO)

        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'ert_performance.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(json_formatter)
        self.perf_logger.addHandler(perf_handler)

        # Error logger
        self.error_logger = logging.getLogger('ert.errors')
        self.error_logger.setLevel(logging.ERROR)

        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'ert_errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        self.error_logger.addHandler(error_handler)

        # Audit logger for user actions
        self.audit_logger = logging.getLogger('ert.audit')
        self.audit_logger.setLevel(logging.INFO)

        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'ert_audit.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(json_formatter)
        self.audit_logger.addHandler(audit_handler)

    def log_performance_metric(self, operation: str, duration: float,
                             success: bool, module: str, **details):
        """Log performance metric"""
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=datetime.now(timezone.utc),
            success=success,
            details=details,
            module=module,
            thread_id=threading.current_thread().name
        )

        with self.metrics_lock:
            self.performance_metrics.append(metric)

        # Log to performance logger
        self.perf_logger.info(
            f"Performance metric: {operation}",
            extra={
                'operation': operation,
                'duration_ms': duration * 1000,
                'success': success,
                'component': module,  # Renamed from 'module' to avoid conflict
                'details': details
            }
        )

    def log_error_event(self, error: Exception, operation: str,
                       module: str, **context):
        """Log error event with context"""
        error_event = ErrorEvent(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now(timezone.utc),
            module=module,
            operation=operation,
            context=context
        )

        with self.metrics_lock:
            self.error_events.append(error_event)

        # Log to error logger
        self.error_logger.error(
            f"Error in {operation}: {str(error)}",
            extra={
                'operation': operation,
                'component': module,  # Renamed from 'module' to avoid conflict
                'error_type': type(error).__name__,
                'context': context
            },
            exc_info=True
        )

    def log_user_action(self, action: str, user_id: str = "system",
                       **context):
        """Log user action for audit trail"""
        self.audit_logger.info(
            f"User action: {action}",
            extra={
                'action': action,
                'user_id': user_id,
                'context': context,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )

    def get_performance_summary(self, last_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for last N minutes"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=last_minutes)

        with self.metrics_lock:
            recent_metrics = [
                m for m in self.performance_metrics
                if m.timestamp > cutoff_time
            ]

        if not recent_metrics:
            return {'message': 'No recent metrics available'}

        # Calculate statistics
        durations = [m.duration for m in recent_metrics]
        success_count = sum(1 for m in recent_metrics if m.success)

        # Group by operation
        operations = {}
        for metric in recent_metrics:
            op = metric.operation
            if op not in operations:
                operations[op] = {
                    'count': 0,
                    'total_duration': 0,
                    'success_count': 0,
                    'avg_duration': 0
                }

            operations[op]['count'] += 1
            operations[op]['total_duration'] += metric.duration
            if metric.success:
                operations[op]['success_count'] += 1

        # Calculate averages
        for op_data in operations.values():
            if op_data['count'] > 0:
                op_data['avg_duration'] = op_data['total_duration'] / op_data['count']
                op_data['success_rate'] = op_data['success_count'] / op_data['count']

        return {
            'time_window_minutes': last_minutes,
            'total_operations': len(recent_metrics),
            'success_rate': success_count / len(recent_metrics),
            'avg_duration_ms': (sum(durations) / len(durations)) * 1000,
            'operations': operations,
            'uptime_minutes': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
        }

    def get_error_summary(self, last_minutes: int = 60) -> Dict[str, Any]:
        """Get error summary for last N minutes"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=last_minutes)

        with self.metrics_lock:
            recent_errors = [
                e for e in self.error_events
                if e.timestamp > cutoff_time
            ]

        if not recent_errors:
            return {'message': 'No recent errors'}

        # Group by error type
        error_types = {}
        for error in recent_errors:
            error_type = error.error_type
            if error_type not in error_types:
                error_types[error_type] = {
                    'count': 0,
                    'modules': set(),
                    'operations': set()
                }

            error_types[error_type]['count'] += 1
            error_types[error_type]['modules'].add(error.module)
            error_types[error_type]['operations'].add(error.operation)

        # Convert sets to lists for JSON serialization
        for error_data in error_types.values():
            error_data['modules'] = list(error_data['modules'])
            error_data['operations'] = list(error_data['operations'])

        return {
            'time_window_minutes': last_minutes,
            'total_errors': len(recent_errors),
            'error_types': error_types,
            'recent_errors': [
                {
                    'type': e.error_type,
                    'message': e.error_message,
                    'module': e.module,
                    'operation': e.operation,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in recent_errors[-5:]  # Last 5 errors
            ]
        }

def performance_monitor(operation: str, module: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get module name if not provided
            func_module = module or func.__module__

            start_time = time.time()
            success = False
            error = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = e
                # Log the error
                if hasattr(ert_logger, 'log_error_event'):
                    ert_logger.log_error_event(
                        error=e,
                        operation=operation,
                        module=func_module,
                        function=func.__name__,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys())
                    )
                raise
            finally:
                duration = time.time() - start_time

                # Log performance metric
                if hasattr(ert_logger, 'log_performance_metric'):
                    ert_logger.log_performance_metric(
                        operation=operation,
                        duration=duration,
                        success=success,
                        module=func_module,
                        function=func.__name__,
                        error_type=type(error).__name__ if error else None
                    )

        return wrapper
    return decorator

# Global logger instance
ert_logger = ERTLogger()

# Export commonly used loggers
app_logger = ert_logger.app_logger
perf_logger = ert_logger.perf_logger
error_logger = ert_logger.error_logger
audit_logger = ert_logger.audit_logger