# Architecture Decision Records

This directory records the major architectural and engineering trade-offs made in TabuLens.

Each record follows the lifecycle:

`Proposed -> Accepted -> (Superseded or Deprecated)`

Existing records are kept for historical context; when a design choice changes, a new ADR is added that supersedes the previous one.

## Index

| ID | Title | Status | Date |
|---|---|---|---|
| [ADR-001](0001-local-first-tabular-workbench.md) | Local-First Tabular Workbench Over Cloud GenAI Assistant | Accepted | 2026-11-16 |
| [ADR-002](0002-leak-free-pipelines-and-threshold-tuning.md) | Pipeline-Encapsulated Preprocessing and Out-of-Fold Threshold Tuning | Accepted | 2026-11-16 |
| [ADR-003](0003-statistical-drift-ks-fdr-and-tvd.md) | Statistical Drift Auditing with Two-Sample KS, Benjamini-Hochberg FDR, and TVD | Accepted | 2026-11-16 |
| [ADR-004](0004-hmac-signed-model-artifacts.md) | HMAC-SHA256 Signing Boundary for Joblib Model Artifacts | Accepted | 2026-11-11 |
| [ADR-005](0005-bounded-ingestion-and-memory-ceilings.md) | Bounded Local Ingestion and Workstation Memory Ceilings | Accepted | 2026-11-11 |
