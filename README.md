# hybrid-rag

Question answering over a small collection of research papers, built in stages with each stage measured:
dense retrieval, then hybrid retrieval (BM25 + dense, merged with reciprocal rank fusion), then reranking,
with cited answers from an LLM.


## How it works

- Ingestion: PDFs are parsed page by page and split into 1000-character chunks with 200 characters of overlap. Every chunk keeps its file name and page number.
- Storage and search: Elasticsearch holds the text (BM25) and a 384-dimension embedding (BAAI/bge-small-en-v1.5, cosine kNN).
- Hybrid retrieval: the top 50 results from BM25 and from dense search are merged with reciprocal rank fusion (k = 60), implemented in Python.
- Answers: the top 5 chunks go to an LLM with instructions to answer only from the context, cite passages as [1], [2], and refuse when the context is insufficient.

## Retrieval evaluation

Corpus: 15 neuroplasticity papers (868 chunks). Questions: 52 (34 drafted by an LLM from random passages and reviewed by hand, 18 written by hand). A retrieved chunk counts as correct if it comes from a labelled paper and page. Latency was measured on a laptop CPU with one Elasticsearch node.

Clean questions (n = 52):

| Retriever | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Latency mean / p95 (ms) |
|---|---|---|---|---|---|
| Dense | 0.933 | 0.942 | 0.913 | 0.909 | 46 / 60 |
| BM25 | 0.942 | 0.952 | 0.909 | 0.905 | 6 / 8 |
| Hybrid (RRF) | 0.933 | 0.962 | 0.932 | 0.919 | 46 / 50 |

Same questions with typos added to the queries (n = 52):

| Retriever | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Latency mean / p95 (ms) |
|---|---|---|---|---|---|
| Dense | 0.923 | 0.942 | 0.853 | 0.860 | 48 / 65 |
| BM25 | 0.885 | 0.913 | 0.819 | 0.832 | 5 / 6 |
| Hybrid (RRF) | 0.904 | 0.923 | 0.884 | 0.870 | 49 / 55 |

Drafted questions (n = 34) vs hand-written questions (n = 18), clean queries:

| Retriever | Drafted Recall@5 | Drafted MRR@10 | Hand-written Recall@5 | Hand-written MRR@10 |
|---|---|---|---|---|
| Dense | 0.985 | 0.941 | 0.833 | 0.861 |
| BM25 | 0.971 | 0.941 | 0.889 | 0.847 |
| Hybrid (RRF) | 0.956 | 0.960 | 0.889 | 0.880 |

Paired bootstrap comparisons over questions (n = 52; difference and 95% confidence interval):

| Comparison | Queries | Metric | Difference | 95% CI |
|---|---|---|---|---|
| Hybrid - Dense | clean | MRR@10 | +0.019 | -0.043 to +0.077 |
| Hybrid - BM25 | clean | MRR@10 | +0.024 | -0.045 to +0.090 |
| Dense - BM25 | clean | MRR@10 | +0.005 | -0.096 to +0.106 |
| Hybrid - Dense | with typos | MRR@10 | +0.031 | -0.044 to +0.108 |
| Hybrid - BM25 | with typos | MRR@10 | +0.065 | -0.010 to +0.142 |
| Dense - BM25 | with typos | Recall@5 | +0.038 | -0.058 to +0.144 |

What the results show:

- On this question set the three retrievers cannot be told apart. Every confidence interval on the differences includes zero, for clean queries and queries with typos, on all four metrics.
- Point estimates only: hybrid has the highest MRR@10 on clean queries (0.932 vs 0.913 dense and 0.909 BM25) and with typos (0.884 vs 0.853 and 0.819). BM25 has the lowest score on all four metrics with typos. MRR@10 dropped by 0.048 for hybrid, 0.060 for dense and 0.090 for BM25 when typos were added, but these drops were not tested formally.
- All three retrievers score lower on the 18 hand-written questions (recall@5 0.83 to 0.89) than on the 34 drafted questions (0.96 to 0.99). This suggests drafted questions, which reuse the passage wording, flatter the clean scores. With 18 questions one question is worth 0.056, so the retrievers cannot be ranked on that subset.
- Two questions are missed completely by hybrid, and both are paraphrase gaps. "What is the duration of the heightened brain activity following a period of being blindfolded?" shares almost no words with the passage ("persisted for at least 30 min"), and "which cortical layer ... thickens" does not match "the fourth layer of cortex hypertrophies". In the second case, two of the top three results are reference-list chunks. The rest of hybrid's lost recall is half credit on questions with two labelled pages.

Limits: 52 questions on one small corpus and one labeller. 34 questions were drafted by an LLM from the passages they are tested against. Several questions share pages, so the intervals are somewhat optimistic, and many comparisons were computed.

## Reproduce

```bash
python -m scripts.ingest --recreate
python -m scripts.evaluate --label baseline
python -m scripts.evaluate --typos --label typos
python -m scripts.compare_runs evaluation/results/<results file>.json
```

## Cross-encoder reranking

Hybrid retrieval scores the query and each chunk separately. A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the query and one chunk together, which is slower but more precise. Version 3 takes hybrid's top 30 candidates, reranks them and returns the top k. It is a fourth retrieval mode, `rerank`, so the same evaluation compares all four modes. Nothing was tuned: default model, 30 candidates.

### Results on 52 questions (34 drafted with a local LLM, 18 hand-written)

Clean queries:

| mode | recall@5 | recall@10 | MRR@10 | nDCG@10 | latency ms (mean / p95) |
|---|---|---|---|---|---|
| dense | 0.933 | 0.942 | 0.913 | 0.909 | 49 / 63 |
| bm25 | 0.942 | 0.952 | 0.909 | 0.905 | 5 / 6 |
| hybrid | 0.933 | 0.962 | 0.932 | 0.919 | 44 / 50 |
| rerank | 0.981 | 0.990 | 0.986 | 0.973 | 2379 / 2801 |

Queries with typos:

| mode | recall@5 | recall@10 | MRR@10 | nDCG@10 | latency ms (mean / p95) |
|---|---|---|---|---|---|
| dense | 0.923 | 0.942 | 0.853 | 0.860 | 35 / 43 |
| bm25 | 0.885 | 0.913 | 0.819 | 0.832 | 4 / 5 |
| hybrid | 0.904 | 0.923 | 0.884 | 0.870 | 45 / 48 |
| rerank | 0.971 | 1.000 | 0.955 | 0.955 | 2364 / 2705 |

### Rerank vs hybrid, paired bootstrap, 95% interval

| queries | metric | rerank - hybrid | 95% CI | excludes 0 |
|---|---|---|---|---|
| clean | recall@5 | +0.048 | [+0.000, +0.106] | no |
| clean | recall@10 | +0.029 | [+0.000, +0.077] | no |
| clean | MRR@10 | +0.053 | [-0.010, +0.123] | no |
| clean | nDCG@10 | +0.054 | [+0.001, +0.114] | barely |
| typos | recall@5 | +0.067 | [+0.000, +0.144] | no |
| typos | recall@10 | +0.077 | [+0.019, +0.144] | yes |
| typos | MRR@10 | +0.071 | [+0.009, +0.144] | yes |
| typos | nDCG@10 | +0.085 | [+0.030, +0.149] | yes |

Subsets (descriptive only, 18 and 34 questions, no intervals computed):

| subset | mode | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| hand-written (18) | hybrid | 0.889 | 0.889 | 0.880 | 0.848 |
| hand-written (18) | rerank | 0.944 | 0.972 | 1.000 | 0.965 |
| drafted (34) | hybrid | 0.956 | 1.000 | 0.960 | 0.957 |
| drafted (34) | rerank | 1.000 | 1.000 | 0.978 | 0.977 |

### What the evidence supports

- On queries with typos, reranking improves recall@10, MRR@10 and nDCG@10 over hybrid; all three intervals exclude 0.
- On clean queries the differences point the same way, but only nDCG@10 excludes 0, and only barely. The other clean-query differences are within noise.
- Dense, BM25 and hybrid are not distinguishable from each other on this question set.
- Both failure cases moved to rank 1 with reranking: q011 (`neuroplasticity_11.pdf`, page 4) and h12 (`neuroplasticity_13.pdf`, page 2). Neither was in hybrid's top 5.

### Limitations

- Reranking costs about 50x the latency (44 ms to about 2.4 s mean, 2.8 s p95) on a CPU-only laptop.
- 52 questions give wide intervals, and 34 of them were drafted by a local LLM from the chunks themselves.
- Reference-list chunks still appear at lower ranks (for h12, ranks 3 and 5), because ingestion does not separate references from body text.

## Version 4: FastAPI service

Retrieval and question answering are now served over HTTP. The embedding model and the reranker load once at startup, so requests do not pay the model-loading cost. `rerank` is the default retrieval mode (set `DEFAULT_MODE` in `.env` to change it).

Run from the project root:

```bash
python -m uvicorn app.api:app --port 8000
```

Interactive docs: `http://localhost:8000/docs`

| endpoint | purpose |
|---|---|
| `GET /health` | liveness check |
| `POST /search` | retrieve chunks. Body: `{"query": "...", "mode": "rerank", "k": 5}` |
| `POST /ask` | retrieve, then answer with the LLM. Returns the answer and numbered citations |

`mode` is one of `dense`, `bm25`, `hybrid`, `rerank`. `k` is 1 to 20.

Example:

```bash
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "What is the duration of the heightened brain activity following a period of being blindfolded?", "k": 5}'
```

Response (4 of the 5 citations omitted here):

```json
{
  "answer": "The heightened brain activity (increased fMRI signal) persisted for at least 30 minutes after re-exposure to light following 60 minutes of blindfolding [1].",
  "mode": "rerank",
  "citations": [
    {"number": 1, "location": "neuroplasticity_11.pdf, page 4", "source": "neuroplasticity_11.pdf", "page": 4, "score": -2.5079569816589355}
  ]
}
```

Notes:

- `citations` lists the retrieved chunks in rank order. The answer marks the ones it used with `[n]`.
- In `rerank` mode `score` is the raw cross-encoder output, which can be negative. Compare it only within one response, not across modes.
- A `/search` request in `rerank` mode took about 2.6 s on a CPU-only laptop with the models already loaded (one measurement, not a benchmark).

| status | meaning |
|---|---|
| 422 | invalid request (blank query, unknown mode, `k` out of range) |
| 404 | nothing retrieved (is the index empty?) |
| 503 | retrieval failed (for example, Elasticsearch is down) |
| 502 | the LLM call failed |

The suite has 29 tests. The API tests replace retrieval and the LLM with stubs, so they run without Elasticsearch or an API key.