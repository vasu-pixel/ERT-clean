"""Utilities for loading report templates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_template(path: str | Path | None = None) -> Dict[str, Any]:
    search_paths = []
    if path:
        search_paths.append(Path(path))
    search_paths.extend([
        Path("report_template.json"),
        Path("config/report_template.json"),
        Path("configs/report_template.json"),
        Path("templates/report_template.json"),
    ])

    for candidate in search_paths:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError("Report template configuration not found. Checked: " + ", ".join(str(p) for p in search_paths))
