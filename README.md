# CiteRAG

[![CI](https://github.com/Jemade/cited-RAG-BOT/actions/workflows/ci.yml/badge.svg)](https://github.com/Jemade/cited-RAG-BOT/actions/workflows/ci.yml)

PDF question answering with page-aware retrieval and inspectable citations. Upload a document, ask a question, and open the source page behind an answer.

## Features

- PDF text extraction and rendered page previews.
- Paragraph-aware chunks with document and page metadata.
- Dense and BM25 retrieval combined through reciprocal rank fusion.
- Cross-encoder reranking and low-confidence refusal.
- Citation validation and a React source inspector.
- Optional model providers and an offline mock generation path.

## Run with Docker

```bash
git clone https://github.com/Jemade/cited-RAG-BOT.git
cd cited-RAG-BOT
docker compose up --build
```

Open http://localhost:3000 for the interface or http://localhost:8000/docs for the API. Compose includes PostgreSQL with pgvector and persistent volumes. Configure provider keys through `.env` if you want live generation. Without provider credentials, the automatic generation mode can use the offline mock.

## Local development

Install the backend requirements in a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --port 8080
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000. Vite proxies API requests to port **8080**. API documentation is at http://localhost:8080/docs.

Database, retrieval models, provider selection, and confidence settings are defined in `backend/app/core/config.py` and `.env.example`. The example selects PostgreSQL with pgvector; configure that database or an appropriate local SQLite connection. Retrieval models may download weights on first use.

## Verification

Run the backend tests from `backend/`:

```bash
pytest -q tests
```

The `evaluation/` directory contains benchmark tooling. See [architecture](docs/architecture.md) for the retrieval pipeline.

## Current scope

The mock generation path is an offline demonstration, not a live model evaluation. Citation checks and confidence thresholds reduce unsupported answers but do not guarantee factual correctness. Scanned PDFs require text extraction support beyond ordinary embedded PDF text.
