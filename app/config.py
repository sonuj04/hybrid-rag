"""Central configuration. Every setting is read from environment variables (.env file)."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains the `app/` folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load PROJECT_ROOT/.env into the environment
load_dotenv(PROJECT_ROOT / ".env")

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")  # empty means Elasticsearch has no authentication
_ca_cert = os.getenv("ES_CA_CERT", "")
ES_CA_CERT = str(PROJECT_ROOT / _ca_cert) if _ca_cert else ""  
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "50"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "rerank")  
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # seconds to wait for one LLM reply

# Judge model for answer evaluation (Version 6): defaults to local Ollama so grading
# doesn't spend Gemini's daily quota. Change to point the judge at Gemini instead if you want.
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://localhost:11434/v1")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "ollama")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma3:latest")
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "120"))  # local CPU inference is slower than Gemini