"""
Real-time Monitoring Dashboard for ERT
Provides system health monitoring and performance visualization
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import threading
from dataclasses import dataclass
import psutil
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_config import ert_logger

@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_mb: Dict[str, float]
    active_threads: int
    timestamp: datetime

class MonitoringDashboard:
    """Real-time monitoring dashboard for ERT system"""

    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.system_health_history: List[SystemHealth] = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.last_network_io = None

    def start_monitoring(self):
        """Start real-time system monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        ert_logger.app_logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        ert_logger.app_logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                health = self._collect_system_health()
                self.system_health_history.append(health)

                # Keep only last 100 entries (about 50 minutes at 30s intervals)
                if len(self.system_health_history) > 100:
                    self.system_health_history = self.system_health_history[-100:]

                # Log system health if concerning
                self._check_system_alerts(health)

                time.sleep(self.update_interval)

            except Exception as e:
                ert_logger.error_logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.update_interval)

    def _collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Network I/O
            network_io = psutil.net_io_counters()
            if self.last_network_io:
                bytes_sent_mb = (network_io.bytes_sent - self.last_network_io.bytes_sent) / (1024 * 1024)
                bytes_recv_mb = (network_io.bytes_recv - self.last_network_io.bytes_recv) / (1024 * 1024)
            else:
                bytes_sent_mb = 0
                bytes_recv_mb = 0

            self.last_network_io = network_io

            network_io_mb = {
                'sent_mb': bytes_sent_mb,
                'received_mb': bytes_recv_mb
            }

            # Active threads
            active_threads = threading.active_count()

            return SystemHealth(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_percent,
                network_io_mb=network_io_mb,
                active_threads=active_threads,
                timestamp=datetime.now()
            )

        except Exception as e:
            ert_logger.error_logger.error(f"Error collecting system health: {e}")
            # Return default values on error
            return SystemHealth(
                cpu_percent=0,
                memory_percent=0,
                disk_usage_percent=0,
                network_io_mb={'sent_mb': 0, 'received_mb': 0},
                active_threads=0,
                timestamp=datetime.now()
            )

    def _check_system_alerts(self, health: SystemHealth):
        """Check for system alerts and log warnings"""
        alerts = []

        if health.cpu_percent > 80:
            alerts.append(f"High CPU usage: {health.cpu_percent:.1f}%")

        if health.memory_percent > 85:
            alerts.append(f"High memory usage: {health.memory_percent:.1f}%")

        if health.disk_usage_percent > 90:
            alerts.append(f"High disk usage: {health.disk_usage_percent:.1f}%")

        if health.active_threads > 50:
            alerts.append(f"High thread count: {health.active_threads}")

        for alert in alerts:
            ert_logger.app_logger.warning(f"System alert: {alert}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        # System health
        current_health = self._collect_system_health() if not self.system_health_history else self.system_health_history[-1]

        # Performance metrics from logger
        perf_summary = ert_logger.get_performance_summary(last_minutes=60)
        error_summary = ert_logger.get_error_summary(last_minutes=60)

        # Cache status from market data provider
        cache_status = self._get_cache_status()

        # Calculate system trends
        trends = self._calculate_trends()

        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'current': {
                    'cpu_percent': current_health.cpu_percent,
                    'memory_percent': current_health.memory_percent,
                    'disk_usage_percent': current_health.disk_usage_percent,
                    'active_threads': current_health.active_threads,
                    'network_io_mb': current_health.network_io_mb
                },
                'status': self._get_health_status(current_health),
                'trends': trends
            },
            'performance': perf_summary,
            'errors': error_summary,
            'cache': cache_status,
            'uptime': {
                'minutes': (datetime.now() - ert_logger.start_time).total_seconds() / 60,
                'start_time': ert_logger.start_time.isoformat()
            }
        }

        return dashboard_data

    def _get_health_status(self, health: SystemHealth) -> str:
        """Determine overall health status"""
        if (health.cpu_percent > 80 or
            health.memory_percent > 85 or
            health.disk_usage_percent > 90):
            return "WARNING"
        elif (health.cpu_percent > 60 or
              health.memory_percent > 70 or
              health.disk_usage_percent > 75):
            return "CAUTION"
        else:
            return "HEALTHY"

    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate trends from health history"""
        if len(self.system_health_history) < 5:
            return {'cpu': 'STABLE', 'memory': 'STABLE', 'disk': 'STABLE'}

        recent = self.system_health_history[-5:]
        older = self.system_health_history[-10:-5] if len(self.system_health_history) >= 10 else recent

        def get_trend(recent_vals, older_vals):
            recent_avg = sum(recent_vals) / len(recent_vals)
            older_avg = sum(older_vals) / len(older_vals)

            if recent_avg > older_avg * 1.1:
                return "INCREASING"
            elif recent_avg < older_avg * 0.9:
                return "DECREASING"
            else:
                return "STABLE"

        return {
            'cpu': get_trend([h.cpu_percent for h in recent], [h.cpu_percent for h in older]),
            'memory': get_trend([h.memory_percent for h in recent], [h.memory_percent for h in older]),
            'disk': get_trend([h.disk_usage_percent for h in recent], [h.disk_usage_percent for h in older])
        }

    def _get_cache_status(self) -> Dict[str, Any]:
        """Get cache status from market data provider"""
        try:
            from data_pipeline.market_data import MarketDataProvider
            provider = MarketDataProvider()
            return provider.get_cache_status()
        except Exception as e:
            ert_logger.error_logger.error(f"Error getting cache status: {e}")
            return {'error': 'Cache status unavailable'}

    def export_metrics_json(self, filepath: str = None) -> str:
        """Export current metrics to JSON file"""
        if filepath is None:
            filepath = f"logs/ert_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        dashboard_data = self.get_dashboard_data()

        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        ert_logger.audit_logger.info(f"Metrics exported to {filepath}")
        return filepath

    def print_dashboard(self):
        """Print formatted dashboard to console"""
        data = self.get_dashboard_data()

        print("\n" + "="*60)
        print("📊 ERT SYSTEM DASHBOARD")
        print("="*60)

        # System Health
        health = data['system_health']
        status_emoji = {"HEALTHY": "✅", "CAUTION": "⚠️", "WARNING": "❌"}
        print(f"\n🖥️  SYSTEM HEALTH: {status_emoji.get(health['status'], '❓')} {health['status']}")
        print(f"   CPU: {health['current']['cpu_percent']:.1f}% ({health['trends']['cpu']})")
        print(f"   Memory: {health['current']['memory_percent']:.1f}% ({health['trends']['memory']})")
        print(f"   Disk: {health['current']['disk_usage_percent']:.1f}% ({health['trends']['disk']})")
        print(f"   Threads: {health['current']['active_threads']}")

        # Performance
        perf = data['performance']
        if 'total_operations' in perf:
            print(f"\n⚡ PERFORMANCE (Last 60 min)")
            print(f"   Operations: {perf['total_operations']}")
            print(f"   Success Rate: {perf['success_rate']:.1%}")
            print(f"   Avg Duration: {perf['avg_duration_ms']:.1f}ms")

        # Errors
        errors = data['errors']
        if 'total_errors' in errors:
            print(f"\n❌ ERRORS (Last 60 min)")
            print(f"   Total Errors: {errors['total_errors']}")
            if errors['total_errors'] > 0:
                print(f"   Error Types: {len(errors['error_types'])}")

        # Cache
        cache = data['cache']
        if 'total_entries' in cache:
            print(f"\n💾 CACHE STATUS")
            print(f"   Active Entries: {cache['total_entries']}")
            print(f"   Expired Entries: {cache.get('expired_entries', 0)}")

        # Uptime
        uptime = data['uptime']
        print(f"\n⏰ UPTIME: {uptime['minutes']:.1f} minutes")

        print("="*60)

# Global monitoring dashboard instance
monitoring_dashboard = MonitoringDashboard()

def start_monitoring():
    """Start system monitoring"""
    monitoring_dashboard.start_monitoring()

def stop_monitoring():
    """Stop system monitoring"""
    monitoring_dashboard.stop_monitoring()

def get_dashboard():
    """Get dashboard data"""
    return monitoring_dashboard.get_dashboard_data()

def print_dashboard():
    """Print dashboard to console"""
    monitoring_dashboard.print_dashboard()