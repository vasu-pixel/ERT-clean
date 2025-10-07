"""File-backed document store for retrieval chunks."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DocumentChunk:
    ticker: str
    chunk_id: str
    text: str
    metadata: Dict[str, str]


class FileBackedDocumentStore:
    """Persist chunks to disk so they can be reused by the retriever."""

    def __init__(self, base_path: Path | str = Path("data/retrieval_store")) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upsert(self, ticker: str, chunks: List[DocumentChunk]) -> None:
        ticker = ticker.upper()
        path = self.base_path / f"{ticker}.json"
        serialized = [asdict(chunk) for chunk in chunks]
        path.write_text(json.dumps(serialized, indent=2))

    def load(self, ticker: str) -> List[DocumentChunk]:
        ticker = ticker.upper()
        path = self.base_path / f"{ticker}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [DocumentChunk(**item) for item in data]

    def list_tickers(self) -> List[str]:
        return [p.stem for p in self.base_path.glob("*.json")]
