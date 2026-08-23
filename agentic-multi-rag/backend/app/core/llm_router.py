"""
Multi-provider LLM router.

Tries the primary provider first (Groq by default, fast + cheap),
then falls back to OpenRouter, then Gemini, so the agent keeps
working even if one provider is rate-limited or down.
"""
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

def get_groq_llm(model: str = "openai/gpt-oss-120b", temperature: float = 0.2):
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=model,
        temperature=temperature,
    )


def get_openrouter_llm(model: str = "meta-llama/llama-3.1-70b-instruct", temperature: float = 0.2):
    # OpenRouter exposes an OpenAI-compatible API
    return ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=temperature,
    )


def get_gemini_llm(model: str = "gemini-1.5-pro", temperature: float = 0.2):
    return ChatGoogleGenerativeAI(
        api_key=settings.gemini_api_key,
        model=model,
        temperature=temperature,
    )


PROVIDERS = {
    "groq": get_groq_llm,
    "openrouter": get_openrouter_llm,
    "gemini": get_gemini_llm,
}


def get_llm(provider: str | None = None, **kwargs):
    """Return an LLM instance for the requested provider, defaulting to
    the configured primary provider. Falls back through the other
    providers on init failure (e.g. missing key)."""
    order = [provider or settings.primary_llm_provider]
    order += [p for p in PROVIDERS if p not in order]

    last_error = None
    for name in order:
        try:
            return PROVIDERS[name](**kwargs)
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
