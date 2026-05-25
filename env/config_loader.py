import json
from pathlib import Path
import os

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def get_config(name: str) -> dict:
    """Load config from JSON file."""
    if name == "eval" and os.environ.get("BENCHMARK_EVAL_CONFIG"):
        config_path = Path(os.environ["BENCHMARK_EVAL_CONFIG"])
    elif name == "runtime" and os.environ.get("BENCHMARK_RUNTIME_CONFIG"):
        config_path = Path(os.environ["BENCHMARK_RUNTIME_CONFIG"])
    else:
        config_path = _CONFIG_DIR / f"{name}.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_vllm_enabled(context: str = "runtime") -> bool:
    """Helper to check if VLLM is enabled for a specific phase."""
    conf = get_config(context)
    return conf.get("vllm", {}).get("enabled", False)

def get_vllm_base_url(context: str = "runtime") -> str:
    """Helper to get VLLM base URL."""
    conf = get_config(context)
    return conf.get("vllm", {}).get("base_url", "http://localhost:8000/v1")
