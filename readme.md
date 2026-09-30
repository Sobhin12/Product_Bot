# Policy Bot

A retrieval-augmented Q&A bot for health-insurance policy documents. Upload a policy PDF, and the bot parses it along its clause structure, indexes it, and answers questions from the policy text only, citing clause numbers and pages. Answers stream as they are generated.

- **Clause-aware ingestion.** It uses Docling to parse the PDF. Chunk boundaries follow the policy's numbering (`4.`, `4.1`, `A.`, Annexures), not a fixed size. It also handles lookup tables, the Customer Information Sheet, reference lists and the "Simplified for you" sidebars.
- **Parent/child retrieval.** Small child chunks are what gets embedded and searched. The full clause or table (the parent) is what the LLM sees.
- **Document lifecycle.** Uploads are deduplicated by content hash, and ingestion runs asynchronously with status tracking. You can retry, delete, or replace a document safely: the new version is indexed before the old one is removed.
- **Streaming answers.** Every answer reports its token counts, latency and the retrieved clauses.

## Architecture

```mermaid
flowchart LR
    UI[Streamlit UI] -->|HTTP| API[FastAPI]

    subgraph Upload
        API -->|POST /upload| LC[lifecycle.upload<br/>validate · hash · dedup]
        LC --> RUN[IngestionRunner<br/>asyncio + semaphore]
        RUN --> BLOB[(S3 / data/blobs<br/>raw PDF + report.md)]
        RUN --> ING[ingestion.run<br/>Docling → clean → chunk]
        ING --> LOAD[retrieval.load]
        LOAD --> EMB[Databricks<br/>gte-large-en embeddings]
        LOAD --> PG[(Postgres<br/>documents · parents · chunks)]
        LOAD --> VEC[(Amazon S3 Vectors)]
    end

    subgraph Query
        API -->|POST /query| GEN[Generator]
        GEN --> RET[retrieve<br/>policy filter → vector search → parents]
        RET --> PG
        RET --> VEC
        GEN --> LLM[LLM<br/>Databricks Llama 3.3 / Bedrock Claude]
    end
```

| Package | Responsibility |
|---|---|
| `src/ingestion` | Turns a PDF into parent and child chunks: parsing, boilerplate removal, sidebar pairing, clause tree, and table chunking. |
| `src/retrieval` | Embedding client, S3 Vectors adapter, Postgres schema, document loading and query-time retrieval. |
| `src/upload` | Upload validation and dedup, raw-file storage, and the background ingestion runner with retries and a stuck-document sweep. |
| `src/generation` | Prompt assembly (retrieved text is wrapped in delimiters and treated as data), streaming LLM clients with retry and backoff, and answer orchestration. |
| `src/api` | FastAPI application. |
| `src/frontend` | Streamlit UI with a Chat tab and a Documents tab. |

The design docs are in [`docs/`](docs/): the [HLD](docs/HLD.md), the [Ingestion LLD](docs/Ingestion_LLD.md), the [File Upload LLD](docs/file_upload_LLD.md) and the [Embedding, Storage & Retrieval LLD](docs/Embeddings,Storage,Retrieval.md).

## Getting started

### Prerequisites

- Python 3.11+
- Docker, to run Postgres
- A Databricks workspace with an embedding endpoint (`gte-large-en`) and, by default, a chat endpoint
- AWS credentials with access to an S3 Vectors bucket and index. The index must use **dimension 1024, cosine distance and float32**. Bedrock access is optional.

### Setup

```bash
python -m venv mvenv
mvenv\Scripts\activate            # Windows  (source mvenv/bin/activate elsewhere)
pip install -r requirements.txt
pip install -e .                  # makes the src/ packages importable

docker compose up -d              # Postgres 16 on localhost:5433
```

Create a `.env` file at the repo root:

```dotenv
# Embeddings + default LLM (Databricks)
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=...
EMBEDDING_MODEL=databricks-gte-large-en

# Vector store
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_VECTORS_BUCKET=...
S3_VECTORS_INDEX=...

# Optional
# S3_DOCS_BUCKET=...              # raw uploads; unset = local data/blobs/ (dev only)
# DATABASE_URL=postgresql://bot:bot@localhost:5433/bot
# LLM_PROVIDER=databricks         # or "bedrock"
# DATABRICKS_LLM_MODEL=databricks-meta-llama-3-3-70b-instruct
# BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
# POLICY_BOT_API_URL=http://localhost:8765   # used by the frontend
```

### Run

```bash
uvicorn api.main:app --port 8765          # API; creates the DB schema on startup
streamlit run src/frontend/app.py         # UI at http://localhost:8501
```

In the UI, open **Documents** and upload a PDF. Give it a name that starts with its policy name, for example `ReAssure 3.0 Policy Wordings`, and wait until its status is `indexed`. Then go to **Chat**, pick the policy and ask a question.

The chat's policy picker lists the prefixes stored in the `policies` table. `ReAssure 3.0` is added automatically. To register another policy, call the API:

```bash
curl -X POST localhost:8765/policies -H "Content-Type: application/json" -d '{"name": "Aspire"}'
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `POST` | `/upload` | Multipart form with `file` (PDF), `display_name`, `is_update` and `replaces_doc_id`. Returns `202 {"doc_id", "status": "accepted"}` when ingestion starts, or `200` with `duplicate` or `no_changes` when the content is already indexed. |
| `GET` | `/documents?limit=&offset=` | Lists documents that haven't been deleted, newest first. |
| `GET` | `/documents/{doc_id}` | One document, with its `status`, `retry_count` and `error_message`. |
| `POST` | `/documents/{doc_id}/retry` | Re-runs a `failed` document from the start (up to 3 retries). |
| `DELETE` | `/documents/{doc_id}` | Removes the document's vectors, rows and raw file. Not allowed while the document is processing. |
| `GET` / `POST` | `/policies` | Lists or adds the policy prefixes shown in the chat picker. |
| `POST` | `/query` | Takes `{"query", "policy"}` and returns an NDJSON stream: one or more `{"type":"token","text":...}` events, then one `{"type":"done","input_tokens","output_tokens","latency_ms","timings","parents"}` event. `timings` gives milliseconds per stage: embed, vector search, db, retrieval total, LLM queue, first token and generation. |

A document's status moves through `pending → scanning → parsing → chunking → embedding → indexed`. If a stage fails, the status becomes `failed`. Deleting a document is a soft delete that sets the status to `deleted`.

## How it works

1. **Upload.** The API streams the file to disk. While doing so it checks the PDF signature and the 50 MB limit and computes a SHA-256 hash. If the same content is already indexed, nothing is stored. Otherwise the API inserts a `documents` row and schedules an ingestion task. At most two ingestion tasks run at once.
2. **Ingest.** The task stores the raw file and checks that the PDF can be read (it must not be encrypted and must have at most 500 pages). It parses the PDF with Docling (OCR is off), strips repeated headers and footers, and separates out the sidebars. It then builds a clause tree and emits parents and children, splitting children at 500 tokens. Tables are chunked according to their type.
3. **Load.** In a single transaction, it inserts `parents` and `chunks` into Postgres, embeds the children, and writes the vectors to S3 Vectors in batches of 100. If anything fails, the vectors already written are deleted.
4. **Replace.** A replacement is ingested as a new document. The old document is deleted only after the new one is indexed. If the new ingestion fails, the old document is left untouched.
5. **Query.** The bot finds the indexed documents whose name starts with the selected policy and embeds the question. It searches S3 Vectors (top 10, filtered to those documents), deduplicates the hits by parent and fetches the parent texts. The prompt wraps each parent in a `<context>` tag. At most 4 LLM calls run at once across all users. Failed LLM calls are retried with backoff, but only before the first token has been streamed.

## Offline tools

To run ingestion without the API, which is useful for tuning the chunkers:

```bash
python -m ingestion.run                     # every PDF in data/input/ → data/output/<stem>/
python -m ingestion.run path/to/file.pdf
python -m retrieval.load <stem> --display-name "ReAssure 3.0 Policy Wordings"
```

Each output folder contains `parents.jsonl`, `children.jsonl` and `report.md`, a human-readable view of every chunk for review. It also contains debug files: `blocks.jsonl`, `sidebars.jsonl`, `removed_boilerplate.jsonl` and `tables/`.

## Tests

```bash
docker compose up -d
pytest -q
```

The tests use in-memory fakes for the vector store, the embeddings, the LLM and the blob storage, so they make no cloud calls. The tests that need Postgres run against the local database and are skipped if it isn't running. They only create and delete rows whose names start with `TEST `.

## Status and roadmap

**Done for v1:** ingestion, upload lifecycle, semantic retrieval, streaming generation, the API and the Streamlit UI.

**Planned in the [HLD](docs/HLD.md) but not built yet:**
- Authentication and per-user rate limiting
- Observability: metrics, structured logs and tracing
- RAGAS offline evaluation
- Multi-turn chat
- Reranking or hybrid search
- Caching
- DOCX input
