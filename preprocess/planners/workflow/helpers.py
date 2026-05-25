"""
Shared low-level helpers used by all workflow phases.
"""

import os
import time
from pathlib import Path
import json

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "planner"

# Environment Variables for API Keys
# You can set these in your terminal or a .env file:
# export ANTHROPIC_API_KEY='your-key-here'
# export OPENAI_API_KEY='your-key-here'

DEFAULT_MODEL = "claude-sonnet-4-6"
STEP = 20  # sliding window size for Phase B


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def call_llm(client, messages: list, tools=None, model: str = DEFAULT_MODEL):
    """
    Unified LLM call wrapper supporting both OpenAI and Anthropic.
    Handles rate limits with exponential backoff.
    """
    is_anthropic = "claude" in model.lower()
    attempt = 0

    while True:
        try:
            if is_anthropic:
                # Convert all messages (dicts and MockMessage objects) to Anthropic format
                system_msg = ""
                anthropic_messages = []
                
                for m in messages:
                    if isinstance(m, dict):
                        role = m.get("role", "")
                        if role == "system":
                            system_msg = m["content"]
                        elif role == "tool":
                            # Tool result → Anthropic format
                            anthropic_messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": m["tool_call_id"],
                                    "content": m["content"]
                                }]
                            })
                        else:
                            anthropic_messages.append({"role": role, "content": m["content"]})
                    else:
                        # MockMessage object from a previous call_llm response
                        content_blocks = []
                        if hasattr(m, 'content') and m.content:
                            content_blocks.append({"type": "text", "text": m.content})
                        if hasattr(m, 'tool_calls') and m.tool_calls:
                            for tc in m.tool_calls:
                                content_blocks.append({
                                    "type": "tool_use",
                                    "id": tc.id,
                                    "name": tc.function.name,
                                    "input": json.loads(tc.function.arguments)
                                })
                        if content_blocks:
                            anthropic_messages.append({"role": "assistant", "content": content_blocks})
                
                # Merge consecutive same-role messages (Anthropic requires alternating roles)
                merged = []
                for msg in anthropic_messages:
                    if merged and merged[-1]["role"] == msg["role"]:
                        prev_content = merged[-1]["content"]
                        new_content = msg["content"]
                        if isinstance(prev_content, str):
                            prev_content = [{"type": "text", "text": prev_content}]
                        if isinstance(new_content, str):
                            new_content = [{"type": "text", "text": new_content}]
                        merged[-1]["content"] = prev_content + new_content
                    else:
                        merged.append(msg)
                
                kwargs = {
                    "model": model,
                    "max_tokens": 4096,
                    "messages": merged,
                    "temperature": 0,
                }
                if system_msg:
                    kwargs["system"] = system_msg
                if tools:
                    anthropic_tools = []
                    for t in tools:
                        anthropic_tools.append({
                            "name": t["function"]["name"],
                            "description": t["function"]["description"],
                            "input_schema": t["function"]["parameters"]
                        })
                    kwargs["tools"] = anthropic_tools

                resp = client.messages.create(**kwargs)
                
                # Build MockMessage for compatibility with existing code
                class MockChoice:
                    def __init__(self, message): self.message = message
                class MockMessage:
                    def __init__(self, content, tool_calls):
                        self.content = content
                        self.tool_calls = tool_calls
                class MockToolCall:
                    def __init__(self, name, args, tc_id):
                        self.function = type('obj', (object,), {'name': name, 'arguments': args})
                        self.id = tc_id

                tool_calls = []
                content = ""
                for block in resp.content:
                    if block.type == "text":
                        content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append(MockToolCall(block.name, json.dumps(block.input), block.id))
                
                return type('obj', (object,), {'choices': [MockChoice(MockMessage(content, tool_calls or None))]})

            else:
                # client is expected to be an openai.OpenAI() instance
                kwargs = dict(model=model, messages=messages, temperature=0, timeout=45.0)
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                return client.chat.completions.create(**kwargs)

        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["rate_limit", "429", "too many requests", "500", "502", "503", "timeout", "connection", "read timed out"]):
                wait = min(2 ** attempt * 10, 300)
                print(f"  [API Transient Error] waiting {wait}s... ({type(e).__name__})")
                time.sleep(wait)
                attempt += 1
            else:
                raise e


def fmt_event_label(e: dict) -> str:
    """Format one timeline event as a compact human-readable string."""
    src = e.get("source_table", "")
    if src == "hosp_procedures_icd_df":
        return (
            f"PROCEDURE: {e.get('long_title_procedure', '')} "
            f"(ICD{e.get('icd_version', '')}: {e.get('icd_code', '')})"
        )
    if src == "hosp_pharmacy_df":
        return (
            f"PHARMACY: {e.get('medication', '')} | {e.get('proc_type', '')} | "
            f"status={e.get('status', '')} | route={e.get('route', '')} | "
            f"freq={e.get('frequency', '')}"
        )
    if src == "hosp_prescriptions_df":
        drug_type = e.get("drug_type", "")
        if drug_type in ("BASE", "ADDITIVE"):
            return (
                f"PRESCRIPTION({drug_type}): {e.get('drug', '')} "
                f"{e.get('dose_val_rx', '')} {e.get('dose_unit_rx', '')}"
            )
        return (
            f"PRESCRIPTION: {e.get('drug', '')} {e.get('dose_val_rx', '')} "
            f"{e.get('dose_unit_rx', '')} {e.get('prod_strength', '')} "
            f"route={e.get('route', '')}"
        )
    if src == "hosp_labevents_df":
        val = e.get("value", e.get("valuenum", ""))
        uom = e.get("valueuom", "")
        return f"LAB: {e.get('label', '')} = {val} {uom} [{e.get('fluid', '')}/{e.get('category', '')}]"
    if src == "radiology_note":
        text = e.get("text", "")
        imp = ""
        for line in text.split("\n"):
            if "IMPRESSION" in line.upper():
                imp = line.strip()
                break
        return f"RADIOLOGY: {imp or text[:120].replace(chr(10), ' ')}"
    return str({k: v for k, v in e.items() if k not in ("index", "source_table", "event_time", "pre_admission")})


def fmt_events(events: list) -> str:
    """Format a list of timeline events for inclusion in a prompt."""
    lines = []
    for e in events:
        pre = " [pre-admission]" if e.get("pre_admission") else ""
        t = str(e.get("event_time", ""))[:16]
        lines.append(f"[{e['index']}] {t}{pre} | {fmt_event_label(e)}")
    return "\n".join(lines)
