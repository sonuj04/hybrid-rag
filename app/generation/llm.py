"""Calls the LLM through an OpenAI-compatible chat API (Gemini, Ollama, OpenAI, ...)."""
import time
from functools import lru_cache

import openai
from openai import OpenAI

from app import config

MAX_ATTEMPTS = 5


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is empty. Set it in your .env file.")
    if not config.LLM_MODEL:
        raise RuntimeError("LLM_MODEL is empty. Set it in your .env file.")
    # max_retries=0: we do our own retrying in generate() so we can print what is happening
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        max_retries=0,
        timeout=config.LLM_TIMEOUT,
    )


def is_daily_quota_error(exc: Exception) -> bool:
    """True if the provider says the DAILY quota is used up (waiting a few seconds won't help)."""
    return "PerDay" in str(exc)


def generate(messages: list[dict], temperature: float = 0.0) -> str:
    client = _get_client()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except (
            openai.InternalServerError,   # 5xx, such as a 503 "high demand" error
            openai.RateLimitError,        # 429
            openai.APIConnectionError,    # network problems, timeouts
        ) as exc:
            if is_daily_quota_error(exc) or attempt == MAX_ATTEMPTS:
                raise
            wait_seconds = 2 ** attempt   # waits 2, 4, 8, then 16 seconds
            print(f"[LLM] {type(exc).__name__}, retrying in {wait_seconds}s (attempt {attempt}/{MAX_ATTEMPTS})")
            time.sleep(wait_seconds)