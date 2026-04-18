"""Lab agent.
Returns lab results when the doctor explicitly requests a specific test by name.
Loads its system prompt from prompts/agents/lab_agent.txt.
"""

from pathlib import Path

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "agents" / "lab_agent.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def run_lab_test(readview: str, test_name: str) -> str:
    """Return lab test results for the requested test."""
    raise NotImplementedError
