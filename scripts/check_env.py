"""Environment sanity check.

Run from the project root:
    python -m scripts.check_env
"""
import sys

from app import config
from app.es_client import get_es_client


def check_python() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")


def check_elasticsearch() -> None:
    es = get_es_client()
    info = es.info()
    print(f"Elasticsearch version: {info['version']['number']}")
    health = es.cluster.health()
    print(f"Cluster health: {health['status']}")
    indices = es.cat.indices(format="json")
    names = sorted(i["index"] for i in indices if not i["index"].startswith("."))
    print(f"Existing indices: {names}")
    print(f"Index this project will create: {config.ES_INDEX}")


def check_embedding_model() -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model {config.EMBEDDING_MODEL} (first run downloads it)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vector = model.encode("hello world")
    print(f"Embedding vector length: {len(vector)}")
    if len(vector) != config.EMBEDDING_DIM:
        raise RuntimeError(
            f"EMBEDDING_DIM in .env is {config.EMBEDDING_DIM} "
            f"but the model produces {len(vector)}"
        )


def main() -> None:
    checks = [
        ("Python", check_python),
        ("Elasticsearch", check_elasticsearch),
        ("Embedding model", check_embedding_model),
    ]
    failed = False
    for name, check in checks:
        print(f"\n--- {name} ---")
        try:
            check()
            print(f"[OK] {name}")
        except Exception as exc:
            failed = True
            print(f"[FAILED] {name}: {exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()