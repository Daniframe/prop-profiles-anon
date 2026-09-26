from src.propensity.common.llm_clients import get_openai_client, get_azure_client


def test_get_openai_client_defaults_to_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = get_openai_client()
    assert client.api_key == "sk-test"


def test_get_openai_client_prefers_explicit_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    client = get_openai_client(api_key="sk-explicit")
    assert client.api_key == "sk-explicit"


def test_get_azure_client_defaults_to_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    client = get_azure_client()
    assert client.api_key == "az-test"


def test_get_openai_client_accepts_custom_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = get_openai_client(base_url="http://localhost:8000/v1")
    assert str(client.base_url) == "http://localhost:8000/v1/"
