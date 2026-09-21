from app import config


def test_llm_settings_exist():
    """generate() reads these when it runs, and the API tests replace generate, so nothing else checks them."""
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT"):
        assert hasattr(config, name), f"config.{name} is missing"