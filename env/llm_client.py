import os
import re
import time
from openai import OpenAI, RateLimitError
from env.config_loader import is_vllm_enabled, get_vllm_base_url, get_config

def get_openai_client(is_env: bool = False) -> OpenAI:
    """Returns an OpenAI-compatible client. 
    Switches to configured endpoint if 'api' config is provided, 
    otherwise checks for vllm legacy config.
    """
    context = "eval" if is_env else "runtime"
    conf = get_config(context)
    
    # 1. Generalized API config
    api_conf = conf.get("api", {})
    if api_conf:
        base_url = api_conf.get("base_url")
        api_key_env = api_conf.get("api_key_env", "OPENAI_API_KEY")
        
        if api_key_env == "EMPTY":
            api_key = "EMPTY"
        else:
            api_key = os.getenv(api_key_env, "").strip().strip('"')
            
        kwargs = {"timeout": 300.0}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAI(**kwargs)

    # 2. Legacy vLLM config fallback
    vllm_enabled = is_vllm_enabled(context)
    vllm_url = get_vllm_base_url(context)
    
    # Fallback to env var for backward compatibility
    env_vllm_url = os.getenv("VLLM_BASE_URL", "").strip().strip('"')
    
    if vllm_enabled or env_vllm_url:
        target_url = vllm_url if vllm_enabled else env_vllm_url
        return OpenAI(
            base_url=target_url,
            api_key="EMPTY"
        )
    else:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip().strip('"'))

def get_model_name(requested_model: str | None = None, is_env: bool = False) -> str:
    """Returns the effective model name.
    Prioritizes JSON config files over environment variables.
    """
    if is_env:
        conf = get_config("eval")
        return conf.get("eval_model", os.getenv("ENV_MODEL", "gpt-5.4-mini"))
    
    # If using vLLM, many backends don't strictly require the specific model name
    # but we'll provide the one from config if it exists
    if is_vllm_enabled("runtime"):
        return requested_model or "local-vllm-model"

    if requested_model:
        return requested_model
    
    runtime_conf = get_config("runtime")
    return runtime_conf.get("mut_model", os.getenv("MUT_MODEL", "gpt-5.4"))

def get_embedding_openai_client(context: str = "eval") -> OpenAI:
    """Returns an OpenAI-compatible client for embedding calls.

    Reads the 'embedding.base_url' field from the context config.
    If not set, falls back to the standard OpenAI API with OPENAI_API_KEY.
    Kept separate from get_openai_client() because embedding endpoints are
    always distinct from LLM completion endpoints (vLLM does not serve both).
    """
    conf = get_config(context)
    base_url = (conf.get("embedding", {}).get("base_url") or "").strip()
    if base_url:
        return OpenAI(base_url=base_url, api_key="EMPTY")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip().strip('"'))


def get_embedding_model_name(context: str = "eval") -> str:
    """Returns the embedding model name from config, defaulting to text-embedding-3-small."""
    conf = get_config(context)
    return conf.get("embedding", {}).get("model", "text-embedding-3-small")


def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip().strip('"'))


def chat_complete(client: OpenAI, **kwargs):
    """Wrapper around client.chat.completions.create with rate-limit retry.

    Parses the suggested wait time from the error message and sleeps accordingly.
    Falls back to exponential backoff if no suggestion is found.
    """
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            if attempt >= 5:
                print(f"[LLM Client] Failed after 5 retries. Raising RateLimitError: {e}")
                raise
            print(f"[LLM Client] RateLimitError: {e}. Retrying... (Attempt {attempt+1})")
            match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(e), re.I)
            if match:
                val, unit = float(match.group(1)), match.group(2)
                suggested = val / 1000 if unit == "ms" else val
                wait = min(suggested + 1, 60)
            else:
                wait = min(2 ** attempt * 5, 120)
            time.sleep(wait)
            attempt += 1
