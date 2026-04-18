"""Patient agent.
LLM-backed agent that answers doctor questions about symptoms and patient-reported history.
Loads its system prompt from prompts/agents/patient_agent.txt.
"""

from pathlib import Path

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "agents" / "patient_agent.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def answer_patient_question(readview: str, question: str) -> str:
    """Return patient's answer to the doctor's question."""
    raise NotImplementedError
