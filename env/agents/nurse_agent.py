"""Nurse agent.
LLM-backed agent that answers doctor questions about vitals, monitoring data, and bedside observations.
Loads its system prompt from prompts/agents/nurse_agent.txt.
"""

from pathlib import Path

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "agents" / "nurse_agent.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def answer_nurse_question(readview: str, query: str) -> str:
    """Return nurse's answer to the doctor's query."""
    raise NotImplementedError
