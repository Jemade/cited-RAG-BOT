# Engineering notes: cited-RAG-BOT

## Purpose and scope

PDF question answering with inspectable page citations. This repository is an independently inspectable project; customer adoption, production scale and commercial readiness are not claimed without evidence.

## Request and data flow

PDF upload → text/page extraction → dense/BM25 retrieval → reranking → generation and citation checks → source inspector.

## Implementation map

Primary implementation and review locations: `backend/app`, `frontend/src`, `evaluation`, `docs/architecture.md`. Dependency manifests and `.github/workflows/` specify installation and automated checks. Read the source for exact contracts and data models.

## Local verification

From `backend` in a configured virtual environment:

```sh
pip install -r requirements.txt
pytest -q tests
```

From the repository root, run `python scripts/repository_check.py` for documentation and tracked-file checks. CI evidence is available in [GitHub Actions](https://github.com/Jemade/cited-RAG-BOT/actions). Green hygiene checks alone do not mean application tests passed.

## Decisions and boundaries

Offline mock output is not a live model evaluation. Model downloads and database setup are required. Scanned PDFs need suitable OCR support; document access controls need deployment review.

Use the README's current run instructions and configuration examples. Keep provider credentials outside Git. Test changes against controlled fixtures before enabling external services. Health checks indicate process/service state, not end-to-end correctness.

## Review and operational evidence

[Review checklist](REVIEW_CHECKLIST.md) distinguishes repository evidence from outstanding human and deployment validation. Report measured workload, environment and method with any performance claim. Document incident fixes through reproducible issues and regression tests; do not invent user counts or peer reviews.

## Reuse and licensing

No repository-wide reuse license has been selected. Public visibility alone does not grant an open-source reuse license. Ownership and third-party asset rights must be confirmed before licensing this project.
