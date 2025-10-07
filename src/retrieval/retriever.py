"""Semantic retrieval over document chunks using lightweight TF-IDF embeddings."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from .document_store import DocumentChunk, FileBackedDocumentStore


class KeywordRetriever:
    """Lightweight TF-IDF retriever for on-disk chunk stores."""

    def __init__(self, store: FileBackedDocumentStore | None = None) -> None:
        self.store = store or FileBackedDocumentStore()

    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in self.TOKEN_PATTERN.findall(text)]

    def _build_tfidf_vectors(self, chunks: List[DocumentChunk]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        doc_term_counts: List[Counter[str]] = []
        doc_freq: Counter[str] = Counter()

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            counts = Counter(tokens)
            doc_term_counts.append(counts)
            for token in counts:
                doc_freq[token] += 1

        total_docs = len(chunks)
        vectors: List[Dict[str, float]] = []

        for counts in doc_term_counts:
            length = sum(counts.values()) or 1
            vector: Dict[str, float] = {}
            for token, freq in counts.items():
                tf = freq / length
                idf = math.log((1 + total_docs) / (1 + doc_freq[token])) + 1
                vector[token] = tf * idf
            vectors.append(vector)

        idf_lookup = {
            token: math.log((1 + total_docs) / (1 + freq)) + 1
            for token, freq in doc_freq.items()
        }

        return vectors, idf_lookup

    def _cosine_similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(token, 0.0) for token, value in a.items())
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, ticker: str, query: str, top_k: int = 3) -> List[DocumentChunk]:
        chunks = self.store.load(ticker)
        if not chunks or not query.strip():
            return []

        vectors, idf_lookup = self._build_tfidf_vectors(chunks)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return chunks[:top_k]

        query_counts = Counter(query_tokens)
        length = sum(query_counts.values()) or 1
        query_vector: Dict[str, float] = {}
        for token, freq in query_counts.items():
            tf = freq / length
            idf = idf_lookup.get(token, math.log((1 + len(chunks)) / 1) + 1)
            query_vector[token] = tf * idf

        scored: List[Tuple[float, DocumentChunk]] = []
        for vector, chunk in zip(vectors, chunks):
            score = self._cosine_similarity(query_vector, vector)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored[:top_k] if score > 0]

        if len(top_chunks) < top_k:
            # Backfill with remaining highest-score chunks regardless of threshold
            additional = [chunk for _, chunk in scored if chunk not in top_chunks]
            top_chunks.extend(additional[: max(0, top_k - len(top_chunks))])

        return top_chunks
