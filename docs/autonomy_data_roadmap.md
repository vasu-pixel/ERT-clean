# Autonomy & Data Orchestration Roadmap

This document captures the rollout plan for evolving ERT into an autonomous, AI-driven equity research platform.

## Phase 0 – Current State Summary
- `StockReportGenerator.fetch_comprehensive_data()` pulls fundamentals directly from `yfinance`, calculates metrics inline, and returns a `CompanyProfile` dataclass.
- SEC filing downloader (`src/fetch/sec_filings.py`) and other alternative data sources are not integrated into report generation.
- Narrative sections rely solely on prompt templates with scalar metrics; no retrieval grounding or deterministic validation.
- The dashboard (now repaired) delegates work to the generator but lacks orchestration features such as retries, queue prioritization, or telemetry.

## Phase 1 – Foundational Data Layer
1. **Unified Data Service**
   - Introduce `DataOrchestrator` responsible for fetching, normalizing, and caching fundamentals, estimates, ownership, and alternative datasets.
   - Persist normalized payloads (initially on-disk Parquet/JSON; evolvable to Postgres or DuckDB).
   - Provide typed accessors consumed by downstream analyzers.
2. **Initial Schema**
   - Company snapshot (entity metadata, prices, classification)
   - Financial statements (income, balance sheet, cash flow, ratios)
   - Market metadata (ownership, insiders, institutional flows)
   - Event streams (earnings, news headlines)

## Phase 2 – Retrieval-Augmented Intelligence
1. **Document Ingestion Pipeline**
   - Ingest 10-K filings via the SEC downloader, earnings transcripts, and relevant news.
   - Chunk and embed documents (Ollama-compatible embeddings or external service if available).
   - Store embeddings + metadata for retrieval (FAISS / chroma / sqlite prototype).
2. **Prompt Integration**
   - Extend `generate_*_with_ai` helpers to pull contextual snippets via retriever.
   - Implement prompt guards ensuring cited evidence is surfaced in outputs.

## Phase 3 – Deterministic Analytics Layer
1. **Valuation & Forecast Engines**
   - Deterministic DCF engine with scenario configuration.
   - Factor-based relative valuation calculators (multiples vs peers).
   - Forecast models for revenue/earnings using historical time series + configurable assumptions.
2. **Risk & Scenario Modeling**
   - Liquidity, leverage, macro sensitivity metrics computed algorithmically.
   - Output standardized scorecards consumed by narrative generation.

## Phase 4 – Guardrails & Quality Assurance
1. **Cross-Validation**
   - Check LLM recommendations and price targets against deterministic outputs; flag discrepancies.
   - Enforce data freshness thresholds (e.g., fail if fundamentals >72h old).
2. **Automated Scoring**
   - NLP quality checks (length, section completeness, sentiment balance).
   - Regression tests comparing new reports to historical baselines.

## Phase 5 – Orchestrated Operations
1. **Dashboard Enhancements**
   - Promote status dashboard to an orchestrator: scheduling, retries, concurrency controls, telemetry charts.
   - Surface guardrail violations and data freshness diagnostics in real time.
2. **Job & Data Pipelines**
   - Add CLI/cron entry points for nightly ingestion and report refresh cycles.

## Deliverables Timeline
- **Sprint 1**: Ship foundational `DataOrchestrator` module + integrate fundamentals path.
- **Sprint 2**: Prototype retrieval pipeline on one 10-K; update prompts to use retrieved context.
- **Sprint 3**: Deliver valuation/risk calculators and first guardrail checks.
- **Sprint 4**: Upgrade dashboard to orchestrator with telemetry + retries.

## Open Questions
- Preferred storage technology for normalized datasets (local Parquet vs managed DB)?
- Available embedding model for local retrieval (Mistral instruct vs sentence transformers)?
- Compliance requirements for storing filings/news locally?

