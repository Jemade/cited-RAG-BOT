# CiteRAG

CiteRAG is a production-oriented PDF question-answering system designed to eliminate ungrounded answers and hallucinated citations in Retrieval-Augmented Generation (RAG).

Rather than treating RAG as an opaque vector lookup that feeds text to an LLM, CiteRAG tracks every passage back to its source document and 1-indexed page number, verifies that cited claims actually match the source text, refuses to answer when retrieval confidence is too low, and provides an interactive source inspector with visual PDF page renders.

---

## What It Does

1. **Ingests PDFs with strict metadata preservation**: Extracts text, detects and filters repeated headers/footers, and renders 150 DPI page preview images for visual verification.
2. **Paragraph-aware chunking**: Splits documents along natural paragraph boundaries and sentences (~1200 characters with 200 character overlap) rather than arbitrary character offsets.
3. **Hybrid Retrieval**: Combines dense vector retrieval (`all-MiniLM-L6-v2` via PostgreSQL + `pgvector`) with BM25 keyword retrieval using Reciprocal Rank Fusion (RRF, $k=60$).
4. **Cross-Encoder Reranking**: Re-scores candidate chunks with `ms-marco-MiniLM-L-6-v2` to evaluate joint query-chunk attention.
5. **Low-Confidence Refusal**: If the top cross-encoder relevance score falls below a calibrated threshold ($\tau = -2.5$), the system explicitly refuses to answer rather than fabricating responses.
6. **Strict Citation Validation**: Inspects generated citations against retrieved context, strips hallucinated page markers, and maps valid claims directly to chunk IDs and supporting text snippets.
7. **Source Inspection UI**: A React interface allowing users to inspect the exact PDF page image and highlighted chunk behind every answer.

---

## Why RAG Needs Citations

Standard RAG architectures suffer from three distinct failure modes:

1. **Hallucinated Citations**: LLMs frequently output convincing citation markers (e.g., `[Source: Page 14]`) for pages that do not exist or passages that were never retrieved.
2. **Nearest-Neighbor Hallucinations**: Vector databases always return the nearest top-$K$ vectors even when the query is completely unrelated to the corpus. The generator then tries to fabricate an answer from irrelevant context.
3. **Loss of Verifiability**: When answers do not link to specific pages and snippets, human reviewers cannot verify factual claims without manually reading the entire source document.

CiteRAG addresses these issues by decoupling retrieval from generation, enforcing cross-encoder confidence thresholds before calling the LLM, and programmatically validating all citation references post-generation.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        A[PDF Document] --> B[PyMuPDF Parser]
        B -->|Page Text & Pixmaps| C[Header/Footer Filter]
        C --> D[Paragraph-Aware Chunker]
        D -->|Page-Aware Chunks + Metadata| E1[Dense Embedding Generator]
        D -->|Page-Aware Chunks + Metadata| E2[BM25 Index Builder]
        E1 -->|384-d Vectors| F1[(PostgreSQL + pgvector)]
        E2 -->|Inverted Index| F2[(BM25 In-Memory Cache)]
    end

    subgraph Query & Retrieval Pipeline
        Q[User Query] --> G1[Dense Retriever]
        Q --> G2[BM25 Retriever]
        F1 -->|Top K Candidates| G1
        F2 -->|Top K Candidates| G2
        G1 --> H[Reciprocal Rank Fusion]
        G2 --> H
        H -->|Fused Top K| I[Cross-Encoder Reranker]
        I -->|Reranked Scores| J{Confidence Check: score >= threshold?}
    end

    subgraph Refusal & Generation
        J -->|No: score < threshold| K[Low-Confidence Refusal Response]
        J -->|Yes: score >= threshold| L[Context Builder]
        L -->|Bounded Prompt + Strict Rules| M[LLM Provider]
        M -->|Raw Answer with Markers| N[Citation Validator]
        N -->|Verify Pages & Lexical Match| O[Verified Answer + Citations]
    end

    subgraph Source Inspection
        O --> P[Frontend UI & Source Viewer]
        B -.->|Page Previews PNG| P
    end
```

See [docs/architecture.md](docs/architecture.md) for the complete component-by-component specification.

---

## Retrieval Pipeline

1. **Dense Retrieval**:
   - Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional normalized vectors.
   - Indexed in PostgreSQL with `pgvector` using an HNSW index (`vector_cosine_ops`).
   - Retrieves top $K=20$ candidates based on cosine similarity.

2. **BM25 Retrieval**:
   - Tokenizes text while preserving technical identifiers, decimals, and hyphens (`WMT-2014`, `10-K`, `0.0001`, `P100`).
   - Computes BM25Okapi scores across the corpus, retrieving top $K=20$ candidates.

3. **Reciprocal Rank Fusion (RRF)**:
   - Merges candidate rankings into a unified score:
     $$RRF(d) = \sum_{m \in \{dense, bm25\}} \frac{1}{k + rank_m(d)}$$
     with $k=60$.

4. **Cross-Encoder Reranking**:
   - Evaluates the top candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Prunes the list to the top $N=5$ most relevant chunks.

---

## Hybrid Search

### Why RRF over Score Normalization?
- **Scale Incompatibility**: Cosine similarity produces bounded values in $[0, 1]$, whereas BM25 produces unbounded non-negative values dependent on document length and term frequency.
- **Outlier Sensitivity**: Min-max normalization depends on the minimum and maximum scores observed in a single query's result set. An outlier score distorts the weights of all other candidates.
- **Empirical Robustness**: Reciprocal Rank Fusion relies purely on rank order, making it scale-invariant and resilient across heterogeneous queries.

---

## Reranking

Dense retrieval evaluates inner products between independent dense vectors: $\langle \mathbf{q}, \mathbf{d} \rangle$. This representation compresses complex textual nuances into a single vector.

The Cross-Encoder evaluates full cross-attention between token representations of the query and candidate chunk: $\text{CrossEncoder}([\mathbf{q}; \mathbf{d}])$. This captures intricate lexical nuances, negation, and specific parameter constraints that bi-encoders miss.

---

## Citation System

CiteRAG does not trust the LLM to emit correct citations. The citation pipeline works as follows:

1. **Prompt Grounding**: The LLM receives numbered context blocks formatted as `[Source ID: X | Page: Y | Chunk ID: Z]`.
2. **Marker Extraction**: A regex scans the output for citations like `[Page 4]` or `[Source 1]`.
3. **Context Verification**: The validator checks whether the cited page was actually provided in the retrieved chunks. If the LLM cites a page not present in the retrieved context, the citation is stripped and logged as a hallucination.
4. **Snippet Grounding**: The validator calculates n-gram overlap between the sentence making the claim and the candidate chunks on that page, identifying the exact supporting sentence and snippet.
5. **Structured Return**: The API returns verified `Citation` objects containing `document_id`, `page_number`, `chunk_id`, and `snippet`.

---

## Confidence / Refusal

When a user asks an out-of-domain question (e.g., "What was the quarterly revenue of Alphabet in Q4 2023?" against a 2017 Transformer paper), standard vector search still retrieves the closest vectors.

CiteRAG evaluates the top cross-encoder relevance score against `CONFIDENCE_THRESHOLD` (default: `-2.5`). If the top score is below this threshold:
- Generation is aborted immediately.
- The system returns: `"I don't know based on the documents available. No sufficiently relevant source was found to answer this question."`
- `refused: true` is recorded in the response and in system metrics.

---

## Evaluation

We evaluate CiteRAG against a 18-question benchmark on the *Attention Is All You Need* (Vaswani et al., 2017) paper:
- **14 Answerable Questions**: Exact ground-truth pages and evidence for parameters (e.g., Adam hyperparameters, 4000 warmup steps, 28.4 BLEU score, $d_{\text{model}}=512$, $h=8$ heads).
- **4 Unanswerable Questions**: Questions intentionally not answered in the paper (e.g., GPT-4 training cost, ibuprofen dosage) to test refusal behavior.

### Benchmark Results

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Reranking |
|---|---|---|---|---|
| **Hit Rate @ 5** | 50.0% | 100.0% | 92.9% | **100.0%** |
| **MRR @ 5** | 0.3381 | 0.9643 | 0.7798 | **1.0000** |
| **Citation Precision** | 42.9% | 73.3% | 50.0% | **60.0%** |
| **Unanswerable Refusal Rate** | 75.0% | 75.0% | 75.0% | **75.0%** |
| **False Refusal Rate** | 57.1% | 7.1% | 14.3% | **0.0%** |
| **Avg Latency** | 6.5 ms | 5.5 ms | 8.6 ms | 9.3 ms |

*See `evaluation/reports/benchmark_report.md` for complete data and analysis.*

---

## API

### Endpoints

- `POST /v1/documents`: Upload and ingest a PDF document.
- `GET /v1/documents`: List all ingested documents with page counts and chunk counts.
- `GET /v1/documents/{id}`: Get document metadata.
- `DELETE /v1/documents/{id}`: Delete document, chunks, and cached previews.
- `POST /v1/documents/{id}/query`: Query a specific document.
- `POST /v1/query`: Query across all indexed documents.
- `GET /v1/documents/{id}/sources/{page}`: Get page text, chunk snippets, and preview availability.
- `GET /v1/documents/{id}/preview/{page}`: Serve the rendered PNG preview image for a page.
- `GET /health`: Service and database health check.
- `GET /metrics`: Retrieval statistics, query counts, and refusal rates.

### Example Query Request

```bash
curl -X POST http://localhost:8000/v1/documents/{id}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What optimizer was used for training and what were its hyperparameters?",
    "retrieval_mode": "hybrid_rerank",
    "top_k": 20,
    "top_n": 5,
    "confidence_threshold": -2.5
  }'
```

### Example Query Response

```json
{
  "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "query": "What optimizer was used for training and what were its hyperparameters?",
  "answer": "We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9. [Page 7]",
  "citations": [
    {
      "document_id": "eval-doc-001",
      "page_number": 7,
      "chunk_id": "eval-doc-001_p7_c12",
      "citation_marker": "[Page 7]",
      "snippet": "We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9.",
      "validated": true,
      "match_score": 0.89
    }
  ],
  "retrieval_mode": "hybrid_rerank",
  "confidence_score": 1.4821,
  "refused": false,
  "refusal_reason": null,
  "latency_ms": 78.4
}
```

---

## UI

The frontend is built with React 18, Vite, and Tailwind CSS.

- **Document Management**: Drag-and-drop PDF upload with real-time status indicators.
- **Interactive Citations**: Clickable `[Page X]` badges within answers.
- **Side-by-Side Source Inspector**: Inspect the high-resolution raster page preview alongside the extracted text and highlighted chunk snippet.
- **Multi-Stage Score Inspection**: Inspect dense similarity, BM25 score, RRF rank, and cross-encoder logit for every retrieved candidate.

---

## Local Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 16 with pgvector (optional for local SQLite fallback)

### 1. Clone & Setup Backend
```bash
git clone https://github.com/your-username/citerag.git
cd citerag/backend

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` in `backend/`:
```env
DATABASE_URL=postgresql+psycopg2://citerag:citerag_password@localhost:5432/citerag_db
LLM_PROVIDER=auto
OPENAI_API_KEY=your_key_here  # Optional: falls back to deterministic mock if omitted
CONFIDENCE_THRESHOLD=-2.5
```

### 3. Run Backend
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Run Frontend
```bash
cd ../frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## Docker

Run the entire stack (PostgreSQL + pgvector, FastAPI, React + Nginx) with a single command:

```bash
docker-compose up --build
```

- **Frontend**: `http://localhost:3000`
- **Backend API & Swagger Docs**: `http://localhost:8000/docs`
- **PostgreSQL**: `localhost:5432`

---

## Testing

Run the full pytest suite:

```bash
pytest backend/tests/ -v
```

Run the benchmark evaluation:

```bash
python3 evaluation/run_benchmark.py
```

---

## Deployment

- **Backend**: Can be containerized and deployed to AWS ECS, GCP Cloud Run, or Render. Minimum recommended RAM: 2GB (for cross-encoder and embedding models).
- **Database**: Managed PostgreSQL with `pgvector` enabled (e.g., Supabase, Neon, AWS RDS for PostgreSQL).
- **Frontend**: Static build deployed to Cloudflare Pages, Vercel, or served via Nginx in Docker.

---

## Limitations

- **Scanned Image PDFs**: Does not perform OCR on pure image PDFs. PyMuPDF extracts embedded text streams. For scanned documents, an upstream OCR pipeline (e.g. Tesseract or Google Cloud Vision) is required.
- **Table Structure**: Complex multi-page nested tables are flattened into text blocks. Tabular cell layout relationships are not fully reconstructed.
- **Memory Footprint**: Loading both the bi-encoder (`all-MiniLM-L6-v2`) and cross-encoder (`ms-marco-MiniLM-L-6-v2`) requires approximately 800MB of RAM.

---

## Future Improvements

- [ ] **Table Parser**: Extract structured Markdown/HTML representations of PDF tables using `pdfplumber` or `unstructured`.
- [ ] **Hierarchical Chunking**: Implement parent-child chunk relationships where small chunks are retrieved and parent sections are passed to the LLM.
- [ ] **Bounding Box Highlighting**: Draw bounding box overlays directly on the rendered PDF preview image in the Source Viewer.
- [ ] **Asynchronous Task Queue**: Offload heavy PDF embedding generation to Celery or Redis Queue for large documents (>100 pages).
