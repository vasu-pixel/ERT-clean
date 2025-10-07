# Data Pipeline Skeleton

The `data_pipeline` package bootstraps the autonomous data layer rollout. It
currently wraps the existing `yfinance` fundamentals fetcher behind a
`DataOrchestrator` that normalizes payloads into typed dataclasses with
lightweight caching.

## Modules
- `models.py` – dataclasses for company snapshots, financial datasets, and the
  aggregate `CompanyDataset` that downstream components consume.
- `orchestrator.py` – `DataOrchestrator` with a `refresh_company_data(ticker)`
  entrypoint that centralizes data fetching and prepares metadata for future
  persistence, enrichment, and guardrail checks.

Future iterations will extend the orchestrator to:
- Merge additional data sources (estimates, ownership, alternative data).
- Persist normalized data to local storage or databases for reuse.
- Trigger ingestion jobs and expose freshness/validation metadata to the
  dashboard orchestrator.
