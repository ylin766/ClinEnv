import json
from .helpers import call_llm, fmt_events, load_prompt

def scan_diagnoses(client, record, verbose=True, model=None, window_size=100, step_size=50):
    """
    Slide through the timeline and mark which ICD diagnoses are groundable.
    """
    if verbose:
        print("\n========== DIAGNOSIS SCANNING ==========")

    diagnoses = record.get_diagnoses()
    if not diagnoses:
        return []

    # Format global diagnosis list for the prompt
    diag_list_str = "\n".join([f"- [{d.get('icd_code')}] {d.get('long_title_diagnosis', d.get('long_title', ''))}" for d in diagnoses])
    
    marked_codes = set()
    total_events = len(record.timeline)
    llm_kwargs = {"model": model} if model else {}

    # Sliding window
    for start_idx in range(0, total_events, step_size):
        end_idx = min(start_idx + window_size - 1, total_events - 1)
        window_events = record.timeline[start_idx : end_idx + 1]
        
        prompt = f"""You are a clinical auditor. 
Below is a window of clinical events (indices {start_idx} to {end_idx}) and a list of discharge diagnoses.

Discharge Diagnoses:
{diag_list_str}

Clinical Events:
{fmt_events(window_events)}

Task: Identify which of the discharge diagnoses above are explicitly established, treated, or evidenced within THIS specific window of events.
Only return codes that a doctor could reasonably infer are being managed or recognized based on these specific events.

Output JSON only:
{{"marked_codes": ["code1", "code2", ...]}}
"""
        try:
            raw = call_llm(client, [{"role": "user", "content": prompt}], **llm_kwargs).choices[0].message.content.strip()
            # Basic cleaning of markdown
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            # Remove any non-json prefix
            if "{" in raw:
                raw = raw[raw.find("{"):]
            if "}" in raw:
                raw = raw[:raw.rfind("}")+1]
                
            res = json.loads(raw)
            for code in res.get("marked_codes", []):
                marked_codes.add(code)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  [Scan] JSON Decode Error at window {start_idx}-{end_idx}: {e}")
        except Exception as e:
            if verbose:
                print(f"  [Scan] Fatal Error at window {start_idx}-{end_idx}: {e}")
            raise e
        
        if end_idx == total_events - 1:
            break

    # Filter original diagnosis records
    final_diagnoses = [d for d in diagnoses if d.get('icd_code') in marked_codes]
    
    if verbose:
        print(f"  → Marked {len(final_diagnoses)}/{len(diagnoses)} diagnoses as groundable")
    
    return final_diagnoses
