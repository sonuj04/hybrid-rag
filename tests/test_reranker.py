from app.retrieval import reranker


class FakeModel:
    """Scores each chunk by the length of its text, so the longest chunk should rank first."""

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return [len(text) for _, text in pairs]


def make_chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "source": "a.pdf", "page": 1, "text": text, "score": 0.0}


def test_rerank_orders_by_model_score(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeModel())
    chunks = [make_chunk("a", "xx"), make_chunk("b", "xxxxxx"), make_chunk("c", "xxxx")]
    result = reranker.rerank("question", chunks, k=2)
    assert [chunk["chunk_id"] for chunk in result] == ["b", "c"]


def test_rerank_does_not_modify_its_input(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeModel())
    chunks = [make_chunk("a", "xx"), make_chunk("b", "xxxxxx")]
    reranker.rerank("question", chunks, k=2)
    assert [chunk["chunk_id"] for chunk in chunks] == ["a", "b"]
    assert chunks[0]["score"] == 0.0


def test_rerank_of_nothing_is_empty():
    assert reranker.rerank("question", []) == []