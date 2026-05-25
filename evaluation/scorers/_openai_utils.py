from openai import OpenAI
from env.llm_client import (
    get_openai_client,
    get_model_name,
    get_embedding_openai_client,
    get_embedding_model_name,
)


def get_client() -> OpenAI:
    """LLM judge client — may point to vLLM or OpenAI depending on eval config."""
    return get_openai_client(is_env=True)


def get_embedding_client() -> OpenAI:
    """Embedding client — reads eval config 'embedding.base_url'.

    Falls back to OpenAI API (OPENAI_API_KEY) when base_url is not set.
    Always separate from the LLM client because vLLM does not serve embeddings.
    """
    return get_embedding_openai_client(context="eval")


def get_model() -> str:
    """LLM judge model name from eval config."""
    return get_model_name(is_env=True)


def get_embed_model() -> str:
    """Embedding model name from eval config (default: text-embedding-3-small)."""
    return get_embedding_model_name(context="eval")
