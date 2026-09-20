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

Corpus: 15 neuroplasticity papers (868 chunks). Questions: 44 (34 drafted by an LLM from random passages and reviewed by hand, 10 written by hand). A retrieved chunk counts as correct if it comes from a labelled paper and page. Latency was measured on a laptop CPU with one Elasticsearch node.

Clean questions (n = 44):

| Retriever | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Latency mean / p95 (ms) |
|---|---|---|---|---|---|
| Dense | 0.955 | 0.966 | 0.898 | 0.907 | 37 / 45 |
| BM25 | 0.966 | 0.977 | 0.877 | 0.893 | 6 / 6 |
| Hybrid (RRF) | 0.955 | 0.989 | 0.931 | 0.931 | 49 / 59 |

Same questions with typos added to the queries (n = 44):

| Retriever | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Latency mean / p95 (ms) |
|---|---|---|---|---|---|
| Dense | 0.932 | 0.966 | 0.838 | 0.857 | 39 / 45 |
| BM25 | 0.898 | 0.909 | 0.769 | 0.800 | 5 / 6 |
| Hybrid (RRF) | 0.943 | 0.966 | 0.882 | 0.885 | 48 / 55 |

Paired bootstrap comparisons over questions (n = 44; 95% confidence interval of the difference):

| Comparison | Queries | Metric | Difference | 95% CI |
|---|---|---|---|---|
| Hybrid - BM25 | with typos | MRR@10 | +0.113 | +0.041 to +0.191 |
| Hybrid - BM25 | with typos | nDCG@10 | +0.084 | +0.013 to +0.162 |
| Hybrid - Dense | with typos | MRR@10 | +0.044 | -0.045 to +0.133 |
| Hybrid - Dense | clean | MRR@10 | +0.034 | -0.031 to +0.095 |
| Hybrid - BM25 | clean | MRR@10 | +0.054 | -0.008 to +0.120 |
| Dense - BM25 | clean | MRR@10 | +0.021 | -0.081 to +0.119 |

What the results show:

- On clean questions the three retrievers cannot be told apart: every confidence interval includes zero, and all three place the right page in the top 5 for about 95-97% of questions.
- With typos in the queries, hybrid retrieval ranks the correct page significantly higher than BM25 alone (MRR@10 +0.113, nDCG@10 +0.084; both intervals exclude zero). Hybrid was nominally ahead of dense retrieval (MRR@10 +0.044), but that difference is within noise.
- Remaining failure: the question "What is the duration of the heightened brain activity following a period of being blindfolded?" has almost no key words in common with the passage, which says the signal "persisted for at least 30 min". BM25 does not retrieve the page in its top 10, dense search ranks it within its top 5, and fusion pushes it to rank 7 because pages ranked moderately by both retrievers outscore a page ranked well by only one.
- Only four questions separate the retrievers on recall@5 (q011, q075, h02, h04), which is why the clean-question differences should not be over-read.

Limits: 44 questions on one small corpus, one labeller, and many questions are drafted from the passages they are then tested against. Many comparisons were computed and the intervals ignore that several questions share pages, so the two significant results are moderate evidence, and MRR and nDCG are correlated, so they count as one finding.

## Reproduce

```bash
python -m scripts.ingest --recreate
python -m scripts.evaluate --label baseline
python -m scripts.evaluate --typos --label typos
python -m scripts.compare_runs evaluation/results/<results file>.json
```