"""Builds the Elasticsearch connection from config."""
from elasticsearch import Elasticsearch

from app import config


def get_es_client() -> Elasticsearch:
    kwargs = {"request_timeout": 30}
    if config.ES_PASSWORD:
        kwargs["basic_auth"] = (config.ES_USER, config.ES_PASSWORD)
    if config.ES_URL.startswith("https") and config.ES_CA_CERT:
        kwargs["ca_certs"] = config.ES_CA_CERT
    return Elasticsearch(config.ES_URL, **kwargs)