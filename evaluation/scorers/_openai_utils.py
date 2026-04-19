"""Shared OpenAI client helpers for all scorers."""

import os
from openai import OpenAI


def get_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip().strip('"'))


def get_model() -> str:
    return os.getenv("ENV_MODEL", "gpt-5.4-mini-2026-03-17")
