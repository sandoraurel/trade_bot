from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class ChatModelClient:
    """
    Small adapter around an OpenAI-compatible chat endpoint.

    Parametric memory lives in the model. Non-parametric memory is injected
    separately through retrieved context and tool results.
    """

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_url = api_url or os.getenv("MODEL_API_URL") or "https://api.openai.com/v1/chat/completions"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("MODEL_API_KEY")
        self.model = model or os.getenv("MODEL_NAME") or "gpt-4o-mini"

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        if not self.is_enabled():
            raise RuntimeError("Model client is not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def plan(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        raw = self.complete(messages, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"answer": raw, "tool_calls": []}
