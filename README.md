# Hybrid RAG

A retrieval-augmented question-answering system over a corpus of research PDFs, built in stages and evaluated at each stage: hybrid retrieval (BM25 + dense, RRF), cross-encoder reranking, and answer-quality checks (citation validity, groundedness, refusal correctness) — served through a FastAPI backend, a minimal browser frontend, and Docker Compose.

## Architecture

```
PDFs -> chunk (1000 chars, 200 overlap) -> embed (BAAI/bge-small-en-v1.5) + BM25 index
                        ↓   
query -> hybrid retrieval (reciprocal rank fusion) -> [cross-encoder rerank] -> top-k chunks
                        ↓
LLM answers only from context, cites [n], refuses if unsupported
```

Elasticsearch stores both the text (BM25) and the embeddings (cosine kNN) in one index.

## Highlights

- **Reranking measurably improves retrieval** on queries with typos (95% CI excludes 0 on recall@10, MRR@10, nDCG@10); the gain is directionally consistent but not statistically significant on clean queries. [Full results →](EVALUATION.md)
- **Answer-quality evaluation** (citation validity, groundedness against retrieved text, correct refusal on out-of-corpus questions) scored 1.00 across a hand-written test set. [Full results →](EVALUATION.md)
- Fully containerized (Docker Compose: API + Elasticsearch), with 37 passing tests, most independent of a live Elasticsearch instance or LLM key.

## Quickstart

### Docker (recommended)

```bash
git clone <repo-url> && cd hybrid-rag
cp .env.example .env   # set LLM_API_KEY and LLM_MODEL at minimum
docker compose up -d --build --wait
docker compose run --rm api python -m scripts.ingest
```

Open `http://localhost:8000/`.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set LLM_API_KEY and LLM_MODEL at minimum
python -m scripts.ingest
python -m uvicorn app.api:app --port 8000
```

Requires a running Elasticsearch (see `docker-compose.yml` for a working config).

## API

| endpoint | purpose |
|---|---|
| `GET /health` | liveness check |
| `POST /search` | retrieve chunks — `{"query": "...", "mode": "rerank", "k": 5}` |
| `POST /ask` | retrieve + answer with citations — same request body |

```bash
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "your question", "k": 5}'
```

`mode`: `dense` \| `bm25` \| `hybrid` \| `rerank` (default). `k`: 1–20.

| status | meaning |
|---|---|
| 422 | invalid request |
| 404 | nothing retrieved |
| 503 | retrieval failed (e.g. Elasticsearch down) |
| 502 | LLM call failed |

Interactive docs: `http://localhost:8000/docs`.

## Evaluation

Two evaluation layers — full methodology, tables, and confidence intervals in **[EVALUATION.md](EVALUATION.md)**:

1. **Retrieval quality** — dense vs. BM25 vs. hybrid vs. reranked, 52 questions, paired bootstrap significance testing.
2. **Answer quality** — does the LLM's answer stay grounded in what was retrieved, cite correctly, and refuse when it should.

## Tests

```bash
python -m pytest -q
```

37 tests. Most run without a live Elasticsearch instance or LLM key (retrieval and generation are stubbed).

## Project layout

```
app/            FastAPI service, retrieval modes, LLM generation, static frontend
scripts/        CLI entry points -- ingest, ask, evaluate, evaluate_answers, ...
tests/          pytest suite
evaluation/     question sets and saved evaluation results
data/raw/       source PDFs (not committed)
```

## Stack

Python · FastAPI · Elasticsearch (BM25 + dense kNN) · sentence-transformers (bi-encoder + cross-encoder) · Gemini API · Docker Compose
EOF