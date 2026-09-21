"""HTTP API: retrieval and question answering over the indexed documents.

Run from the project root:
    python -m uvicorn app.api:app --port 8000
Interactive docs: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import config
from app.generation.llm import generate
from app.generation.prompt import build_messages, format_location
from app.retrieval.retriever import MODES, search

logger = logging.getLogger("rag.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the embedding model and the reranker once, before the first request."""
    try:
        search("warm up", mode="rerank", k=1)
    except Exception:
        logger.exception("Warm-up failed; the first request will load the models instead")
    yield


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=1000)
    mode: str = config.DEFAULT_MODE
    k: int = Field(default=config.TOP_K, ge=1, le=20)

    @field_validator("mode")
    @classmethod
    def mode_must_exist(cls, value: str) -> str:
        if value not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return value


class ChunkOut(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    text: str
    score: float


class SearchResponse(BaseModel):
    mode: str
    results: list[ChunkOut]


class Citation(BaseModel):
    number: int
    location: str
    source: str
    page: int | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    mode: str
    citations: list[Citation]


app = FastAPI(title="Hybrid RAG", version="4.0", lifespan=lifespan)


def retrieve(request: QueryRequest) -> list[dict]:
    try:
        return search(request.query, mode=request.mode, k=request.k)
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Retrieval failed. Is Elasticsearch running?")


# These are plain `def` routes on purpose: FastAPI runs them in a thread pool,
# so a slow rerank or LLM call does not block other requests.
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search_documents(request: QueryRequest) -> dict:
    return {"mode": request.mode, "results": retrieve(request)}


@app.post("/ask", response_model=AskResponse)
def ask(request: QueryRequest) -> dict:
    chunks = retrieve(request)
    if not chunks:
        raise HTTPException(status_code=404, detail="Nothing retrieved. Did you run scripts.ingest?")
    messages = build_messages(request.query, chunks)
    try:
        answer = generate(messages)
    except Exception:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail="The language model call failed.")
    citations = [
        {
            "number": number,
            "location": format_location(chunk),
            "source": chunk["source"],
            "page": chunk.get("page"),
            "score": chunk["score"],
        }
        for number, chunk in enumerate(chunks, start=1)
    ]
    return {"answer": answer, "mode": request.mode, "citations": citations}