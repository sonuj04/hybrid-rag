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