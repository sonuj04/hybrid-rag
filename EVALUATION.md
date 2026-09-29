# Evaluation

## Retrieval quality

Corpus: 15 neuroplasticity papers (868 chunks). Questions: 52 (34 drafted by an LLM from random passages and reviewed by hand, 18 written by hand). A retrieved chunk counts as correct if it comes from a labelled paper and page. Latency was measured on a laptop CPU with one Elasticsearch node.

Reranking (Version 3) adds a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as a fourth mode: hybrid's top 30 candidates are reranked and the top k returned. Nothing was tuned — default model, 30 candidates.

### Clean queries (n = 52)

| mode | recall@5 | recall@10 | MRR@10 | nDCG@10 | latency ms (mean / p95) |
|---|---|---|---|---|---|
| dense | 0.933 | 0.942 | 0.913 | 0.909 | 49 / 63 |
| bm25 | 0.942 | 0.952 | 0.909 | 0.905 | 5 / 6 |
| hybrid | 0.933 | 0.962 | 0.932 | 0.919 | 44 / 50 |
| rerank | 0.981 | 0.990 | 0.986 | 0.973 | 2379 / 2801 |

### Queries with typos (n = 52)

| mode | recall@5 | recall@10 | MRR@10 | nDCG@10 | latency ms (mean / p95) |
|---|---|---|---|---|---|
| dense | 0.923 | 0.942 | 0.853 | 0.860 | 35 / 43 |
| bm25 | 0.885 | 0.913 | 0.819 | 0.832 | 4 / 5 |
| hybrid | 0.904 | 0.923 | 0.884 | 0.870 | 45 / 48 |
| rerank | 0.971 | 1.000 | 0.955 | 0.955 | 2364 / 2705 |

### Drafted (n = 34) vs. hand-written (n = 18) questions, clean queries

| Retriever | Drafted Recall@5 | Drafted MRR@10 | Hand-written Recall@5 | Hand-written MRR@10 |
|---|---|---|---|---|
| Dense | 0.985 | 0.941 | 0.833 | 0.861 |
| BM25 | 0.971 | 0.941 | 0.889 | 0.847 |
| Hybrid (RRF) | 0.956 | 0.960 | 0.889 | 0.880 |

All three retrievers score lower on the 18 hand-written questions than on the 34 drafted ones — drafted questions reuse the passage wording, which flatters clean-query scores. With only 18 hand-written questions, one question is worth 0.056, so the retrievers can't be meaningfully ranked on that subset alone.

### Dense vs. BM25 vs. hybrid: paired bootstrap, 95% interval

| Comparison | Queries | Metric | Difference | 95% CI |
|---|---|---|---|---|
| Hybrid - Dense | clean | MRR@10 | +0.019 | -0.043 to +0.077 |
| Hybrid - BM25 | clean | MRR@10 | +0.024 | -0.045 to +0.090 |
| Dense - BM25 | clean | MRR@10 | +0.005 | -0.096 to +0.106 |
| Hybrid - Dense | with typos | MRR@10 | +0.031 | -0.044 to +0.108 |
| Hybrid - BM25 | with typos | MRR@10 | +0.065 | -0.010 to +0.142 |
| Dense - BM25 | with typos | Recall@5 | +0.038 | -0.058 to +0.144 |

On this question set the three base retrievers can't be told apart — every interval includes zero, clean or with typos. Point estimates favor hybrid on MRR@10, but not with statistical confidence.

### Rerank vs. hybrid: paired bootstrap, 95% interval

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

### Rerank vs. hybrid by question source (descriptive, no intervals)

| subset | mode | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| hand-written (18) | hybrid | 0.889 | 0.889 | 0.880 | 0.848 |
| hand-written (18) | rerank | 0.944 | 0.972 | 1.000 | 0.965 |
| drafted (34) | hybrid | 0.956 | 1.000 | 0.960 | 0.957 |
| drafted (34) | rerank | 1.000 | 1.000 | 0.978 | 0.977 |

### What the evidence supports

- Reranking improves recall@10, MRR@10 and nDCG@10 over hybrid on queries with typos — all three intervals exclude 0.
- On clean queries the differences point the same way, but only nDCG@10 excludes 0, and only barely. The rest are within noise.
- Dense, BM25 and hybrid are not distinguishable from each other on this question set.
- Both of hybrid's failure cases moved to rank 1 after reranking: one question shared almost no words with its answer passage ("persisted for at least 30 min" vs. "duration of heightened brain activity"); the other had two reference-list chunks outranking the correct page.

### Limitations

- 52 questions on one small corpus and one labeller; 34 were drafted by an LLM from the passages they're tested against, which likely flatters clean-query scores (see the drafted vs. hand-written table above).
- Several questions share pages, so the confidence intervals are somewhat optimistic, and many comparisons were computed without a multiple-comparisons correction.
- Reranking costs about 50x the latency (44 ms → ~2.4 s mean, 2.8 s p95) on a CPU-only laptop.
- Reference-list chunks still occasionally outrank body text at lower ranks, because ingestion doesn't separate references from body content.

### Reproduce

```bash
python -m scripts.ingest --recreate
python -m scripts.evaluate --label baseline
python -m scripts.evaluate --typos --label typos
python -m scripts.compare_runs evaluation/results/<results file>.json --baseline hybrid
```

## Answer quality

Retrieval evaluation measures whether the right chunk was found. It says nothing about whether the LLM's *answer* stays faithful to that chunk. This layer checks three things per answer, using `rerank` retrieval (k=5):

1. **Citation validity** — every `[n]` in the answer must point at a chunk that was actually retrieved. Checked with a regex, no LLM call.
2. **Groundedness** — for questions the corpus can answer, a judge model checks whether each cited claim is actually supported by the passage it cites. The judge runs locally (Ollama, `gemma3:latest`), so grading doesn't touch the generation model's quota.
3. **Refusal correctness** — for questions the corpus *cannot* answer, the model must reply with the exact sentence: *"I don't have enough information in the provided documents."*

```bash
python -m scripts.evaluate_answers
```

### Results: 8 hand-written answerable questions, 6 deliberately unanswerable questions

| check | rate |
|---|---|
| citation validity | 1.00 (8/8) |
| groundedness | 1.00 (8/8) |
| refusal correctness | 1.00 (6/6) |

### Limitations

- 14 questions is a small sample; a perfect score here is encouraging, not proof the model never hallucinates. The retrieval evaluation above (52 questions) is the more statistically meaningful one.
- The judge model (local `gemma3:latest`) is weaker than the model being judged. A stronger judge might be more critical.
- The 6 unanswerable questions were hand-written to be clearly outside the corpus (drug names, costs, statistics not covered by these papers); real user queries near the edge of the corpus's coverage would be a harder test.

### A note on the generation model

Generation originally used `gemini-3.8-flash`. During this evaluation it repeatedly failed — a very low free-tier daily quota (20 requests/day) one day, then a "high demand" 503 upstream outage the next — both specific to that model, which was only 3 weeks old at the time. Generation now uses `gemini-3.1-flash-lite`, an established model with a larger free-tier quota, chosen over the also-available `gemini-flash-lite-latest` because it's a pinned version rather than an alias that could silently change underneath the evaluation later.
