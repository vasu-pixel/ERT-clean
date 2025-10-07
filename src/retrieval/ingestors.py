"""Document ingestion helpers for retrieval."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.fetch.sec_filings import download_latest_10k

from .document_store import DocumentChunk, FileBackedDocumentStore

logger = logging.getLogger(__name__)


class SECIngestor:
    """Fetch and chunk 10-K filings for retrieval."""

    def __init__(self, store: FileBackedDocumentStore | None = None) -> None:
        self.store = store or FileBackedDocumentStore()

    def ingest_latest_10k(self, ticker: str) -> List[DocumentChunk]:
        ticker = ticker.upper()
        filing_path = download_latest_10k(ticker)
        if not filing_path:
            logger.warning("No 10-K available for %s", ticker)
            return []

        try:
            text = Path(filing_path).read_text(errors="ignore")
        except Exception as exc:
            logger.error("Failed to read filing for %s: %s", ticker, exc)
            return []

        filing_path_obj = Path(filing_path)
        file_metadata = {
            "source": "10-K",
            "filing_file": filing_path_obj.name,
            "filing_accession": filing_path_obj.parent.name,
        }

        chunks = self._chunk_text(ticker, text, file_metadata=file_metadata)
        if chunks:
            self.store.upsert(ticker, chunks)
        return chunks

    def _chunk_text(
        self,
        ticker: str,
        text: str,
        max_chars: int = 1200,
        file_metadata: Optional[Dict[str, str]] = None,
    ) -> List[DocumentChunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[DocumentChunk] = []
        buffer: List[str] = []
        current_len = 0
        chunk_index = 0

        for paragraph in paragraphs:
            if current_len + len(paragraph) > max_chars and buffer:
                chunk_text = "\n\n".join(buffer)
                chunks.append(
                    DocumentChunk(
                        ticker=ticker,
                        chunk_id=f"{ticker}_chunk_{chunk_index}",
                        text=chunk_text,
                        metadata={
                            "source": (file_metadata or {}).get("source", "10-K"),
                            "order": str(chunk_index),
                            "filing": (file_metadata or {}).get("filing_file"),
                            "accession": (file_metadata or {}).get("filing_accession"),
                        },
                    )
                )
                buffer = []
                current_len = 0
                chunk_index += 1

            buffer.append(paragraph)
            current_len += len(paragraph)

        if buffer:
            chunk_text = "\n\n".join(buffer)
            chunks.append(
                DocumentChunk(
                    ticker=ticker,
                    chunk_id=f"{ticker}_chunk_{chunk_index}",
                    text=chunk_text,
                    metadata={
                        "source": (file_metadata or {}).get("source", "10-K"),
                        "order": str(chunk_index),
                        "filing": (file_metadata or {}).get("filing_file"),
                        "accession": (file_metadata or {}).get("filing_accession"),
                    },
                )
            )

        return chunks
