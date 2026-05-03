from openai import OpenAI
from env.llm_client import get_openai_client, get_model_name

def get_client() -> OpenAI:
    return get_openai_client(is_env=True)

def get_model() -> str:
    return get_model_name(is_env=True)
