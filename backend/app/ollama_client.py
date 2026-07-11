import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "granite4.1:3b"
    timeout_sec: int = 45


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        json_output: bool = False,
    ) -> str | dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_output:
            payload["format"] = "json"

        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout_sec)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc

        body = resp.json()
        content = body.get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response")

        if not json_output:
            return content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return _extract_json_object(content)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        json_output: bool = False,
    ) -> str | dict[str, Any]:
        return self.chat_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            json_output=json_output,
        )