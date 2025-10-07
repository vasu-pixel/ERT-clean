"""Ollama-specific wrapper for the stock report generator.

Provides a drop-in replacement for the OpenAI-based generator so the
status dashboard and other tooling can work with the local Ollama engine.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.stock_report_generator import StockReportGenerator
from src.utils.ollama_engine import OllamaEngine, get_global_engine

logger = logging.getLogger(__name__)


class EnhancedEquityResearchGenerator:
    """Convenience wrapper that binds ``StockReportGenerator`` to Ollama."""

    def __init__(self) -> None:
        self._engine = get_global_engine()
        self._generator = StockReportGenerator(self._engine)
        logger.info("Configuration loaded successfully")

    def fetch_comprehensive_data(self, ticker: str):
        """Proxy to the underlying generator for dashboard usage."""
        return self._generator.fetch_comprehensive_data(ticker)

    def generate_comprehensive_report(self, ticker: str) -> str:
        """Generate the full markdown report using the Ollama engine."""
        return self._generator.generate_comprehensive_report(ticker)

    def get_model_info(self) -> Dict:
        return self._engine.get_model_info()


def _collect_report_metadata(limit: int = 10) -> Dict[str, Optional[str]]:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return {"count": 0, "latest_file": None, "latest_modified_at": None}

    report_files = sorted(
        reports_dir.glob("*.md"),
        key=lambda path: path.stat().st_mtime
    )

    if not report_files:
        return {"count": 0, "latest_file": None, "latest_modified_at": None}

    latest = report_files[-1]
    return {
        "count": len(report_files[:limit]),
        "latest_file": latest.name,
        "latest_modified_at": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
    }


def get_system_status() -> Dict:
    """Return health information for the dashboard."""
    engine = get_global_engine()
    reachable = engine.test_connection()
    model_info = engine.get_model_info()

    report_meta = _collect_report_metadata()

    return {
        "ollama": {
            "reachable": reachable,
            "model": model_info.get("model"),
            "base_url": model_info.get("base_url"),
            "max_tokens": model_info.get("max_tokens"),
            "temperature": model_info.get("temperature"),
        },
        "reports": report_meta,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
