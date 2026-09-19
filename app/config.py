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
ES_CA_CERT = str(PROJECT_ROOT / _ca_cert) if _ca_cert else ""  # empty means no custom certificate
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))