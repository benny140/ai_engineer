import json
from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI


@dataclass
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-10-21"
    timeout_sec: int = 60


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


class AzureOpenAIClient:
    def __init__(self, config: AzureOpenAIConfig):
        if not config.endpoint:
            raise ValueError("Missing Azure OpenAI endpoint")
        if not config.api_key:
            raise ValueError("Missing Azure OpenAI API key")
        if not config.deployment:
            raise ValueError("Missing Azure OpenAI deployment name")

        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
            timeout=config.timeout_sec,
        )

    def _create(
        self,
        messages: list[dict[str, str]],
        json_output: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.deployment,
            "messages": messages,
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Azure OpenAI call failed: {exc}") from exc

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Azure OpenAI returned an empty response")
        return content

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        json_output: bool = False,
    ) -> str | dict[str, Any]:
        content = self._create(messages, json_output)
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
