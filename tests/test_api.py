from fastapi.testclient import TestClient

from app import api, config

# No `with` here, so the startup warm-up does not run during tests.
client = TestClient(api.app)


def make_chunk(number: int) -> dict:
    return {
        "chunk_id": f"c{number}",
        "source": "a.pdf",
        "page": number,
        "text": f"text {number}",
        "score": 1.0 / number,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_the_retrieved_chunks(monkeypatch):
    monkeypatch.setattr(api, "search", lambda query, mode, k: [make_chunk(1), make_chunk(2)])
    response = client.post("/search", json={"query": "what is plasticity?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == config.DEFAULT_MODE
    assert [item["chunk_id"] for item in body["results"]] == ["c1", "c2"]


def test_search_rejects_an_unknown_mode():
    response = client.post("/search", json={"query": "x", "mode": "magic"})
    assert response.status_code == 422


def test_search_rejects_a_blank_query():
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 422


def test_search_reports_a_retrieval_failure_as_503(monkeypatch):
    def broken(query, mode, k):
        raise RuntimeError("Elasticsearch is down")

    monkeypatch.setattr(api, "search", broken)
    response = client.post("/search", json={"query": "x"})
    assert response.status_code == 503


def test_ask_returns_an_answer_with_numbered_citations(monkeypatch):
    monkeypatch.setattr(api, "search", lambda query, mode, k: [make_chunk(1), make_chunk(2)])
    monkeypatch.setattr(api, "generate", lambda messages: "an answer [1]")
    response = client.post("/ask", json={"query": "what is plasticity?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "an answer [1]"
    assert [citation["number"] for citation in body["citations"]] == [1, 2]
    assert body["citations"][0]["source"] == "a.pdf"
    assert body["citations"][0]["page"] == 1


def test_ask_with_nothing_retrieved_is_404_and_skips_the_llm(monkeypatch):
    monkeypatch.setattr(api, "search", lambda query, mode, k: [])

    def must_not_run(messages):
        raise AssertionError("the LLM must not be called without context")

    monkeypatch.setattr(api, "generate", must_not_run)
    response = client.post("/ask", json={"query": "x"})
    assert response.status_code == 404


def test_ask_reports_an_llm_failure_as_502(monkeypatch):
    monkeypatch.setattr(api, "search", lambda query, mode, k: [make_chunk(1)])

    def broken(messages):
        raise RuntimeError("timeout")

    monkeypatch.setattr(api, "generate", broken)
    response = client.post("/ask", json={"query": "x"})
    assert response.status_code == 502