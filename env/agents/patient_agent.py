"""Patient agent — LLM-backed.

Answers doctor questions in character as a patient, using only
information from the patient readview (demographics, CC, HPI, PMH).
"""

from __future__ import annotations
from pathlib import Path

from openai import OpenAI
from env.llm_client import chat_complete

_PROMPT = Path(__file__).parent.parent.parent / "prompts" / "env" / "patient_agent.txt"


def _build_context(readview: dict) -> str:
    parts = []
    age, gender = readview.get("age"), readview.get("gender")
    if age or gender:
        parts.append(f"Patient: {age or '?'} y/o {gender or '?'}")
    for key, label in [
        ("chief_complaint",    "Chief Complaint"),
        ("hpi",                "History of Present Illness"),
        ("past_medical_history", "Past Medical History"),
    ]:
        val = readview.get(key)
        if val:
            parts.append(f"{label}:\n{val}")
    return "\n\n".join(parts)


def answer(readview: dict, question: str, client: OpenAI, model: str) -> str:
    """Return the patient's answer to a doctor's question."""
    system  = _PROMPT.read_text(encoding="utf-8").strip()
    context = _build_context(readview)
    resp = chat_complete(
        client,
        model=model,
        messages=[
            {"role": "system", "content": f"{system}\n\n---\n{context}"},
            {"role": "user",   "content": question},
        ],
        temperature=0,
        max_completion_tokens=300,
    )
    msg = resp.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    return text.strip()
