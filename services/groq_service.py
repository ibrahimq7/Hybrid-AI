from __future__ import annotations

from dataclasses import dataclass
import socket

import httpx

from core.config import get_settings


@dataclass(frozen=True)
class LLMResult:
    text: str | None
    error: str | None = None
    status_code: int | None = None
    provider: str | None = None


class GroqService:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model
        self.base_url = settings.groq_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.25,
        max_tokens: int = 700,
    ) -> LLMResult:
        if not self.enabled:
            return LLMResult(text=None, error="Groq is not configured.", provider="Groq")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            with httpx.Client(timeout=20.0, trust_env=False) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_completion_tokens": max_tokens,
                    },
                )
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if response.status_code >= 400:
                error = payload.get("error", {})
                message = error.get("message") if isinstance(error, dict) else None
                return LLMResult(
                    text=None,
                    error=message or "Groq request failed.",
                    status_code=response.status_code,
                    provider="Groq",
                )

            choices = payload.get("choices") or []
            if not choices:
                return LLMResult(
                    text=None,
                    error="Groq returned no choices.",
                    status_code=response.status_code,
                    provider="Groq",
                )

            text = (choices[0].get("message", {}).get("content") or "").strip()
            if not text:
                return LLMResult(
                    text=None,
                    error="Groq returned an empty response.",
                    status_code=response.status_code,
                    provider="Groq",
                )

            return LLMResult(text=text, status_code=response.status_code, provider=f"Groq / {self.model}")
        except httpx.HTTPError as error:
            return LLMResult(text=None, error=self._format_connection_error(error), provider="Groq")

    @staticmethod
    def _format_connection_error(error: httpx.HTTPError) -> str:
        message = str(error)
        lowered = message.lower()
        if "10061" in lowered or "actively refused" in lowered:
            return (
                "Groq connection was refused. Check your internet access, firewall, VPN, or local proxy settings. "
                "The app now ignores system proxy variables, so if this continues, verify that api.groq.com is reachable."
            )
        if isinstance(error, httpx.ConnectTimeout):
            return "Groq connection timed out. Check your internet connection and try again."
        if isinstance(error, httpx.ReadTimeout):
            return "Groq took too long to respond. Please try again."
        if isinstance(error, httpx.ConnectError):
            inner = error.__cause__
            if isinstance(inner, socket.gaierror):
                return "Groq could not be reached because DNS lookup failed. Check your internet or DNS settings."
            return "Groq could not be reached. Check your internet connection and firewall settings."
        return message or "Groq request failed."
