from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


@dataclass
class GenerationStats:
    tokens: int
    seconds: float
    tokens_per_second: float
    prompt_tokens: int | None = None
    prompt_seconds: float | None = None
    estimated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "prompt_tokens": self.prompt_tokens,
            "prompt_seconds": (
                round(self.prompt_seconds, 3)
                if self.prompt_seconds is not None
                else None
            ),
            "estimated": self.estimated,
        }


class BaseLLM:
    name = "base"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.last_stats: GenerationStats | None = None
        self.on_progress: Callable[[dict[str, Any]], None] | None = None

    def _emit_progress(self, **data: Any) -> None:
        if not self.on_progress:
            return
        try:
            self.on_progress(data)
        except Exception:  # noqa: BLE001
            # Ошибка отрисовки не должна ломать генерацию.
            pass

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        return strip_thinking(
            "".join(
                self.stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            )
        )

    def health(self) -> str:
        return "ok"

    def list_models(self) -> list[str]:
        return []

    def loaded_models(self) -> list[dict[str, Any]]:
        return []

    def load_model(self) -> str:
        raise LLMError("Загрузка модели поддерживается только для Ollama.")

    def unload_model(self) -> str:
        raise LLMError("Выгрузка модели поддерживается только для Ollama.")

    def warn_if_model_missing(self) -> str | None:
        models = self.list_models()
        if not models:
            return None
        wanted = self.cfg.model.split(":")[0]
        if any(m == self.cfg.model or m.split(":")[0] == wanted for m in models):
            return None
        return (
            f"Модель «{self.cfg.model}» не найдена на сервере. "
            f"Доступны: {', '.join(models[:10])}"
        )


class OllamaLLM(BaseLLM):
    name = "ollama"

    def _options(
        self,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "min_p": self.cfg.min_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "repeat_last_n": self.cfg.repeat_last_n,
            "num_predict": self.cfg.max_tokens if max_tokens is None else max_tokens,
            "num_ctx": self.cfg.num_ctx,
        }
        if self.cfg.num_thread:
            options["num_thread"] = self.cfg.num_thread
        if seed is not None:
            options["seed"] = seed
        return options

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Iterator[str]:
        self.last_stats = None
        self._emit_progress(
            stage="connecting", tokens=0, tokens_per_second=0.0
        )
        if self.cfg.no_think and messages and messages[0].get("role") == "system":
            messages = [dict(message) for message in messages]
            messages[0]["content"] = (
                messages[0]["content"].rstrip() + "\n\n/no_think"
            )
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "options": self._options(temperature, max_tokens, seed),
        }
        if self.cfg.think is not None:
            payload["think"] = self.cfg.think
        if self.cfg.keep_alive is not None:
            payload["keep_alive"] = self.cfg.keep_alive

        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.cfg.timeout, connect=5.0)
            ) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = b"".join(response.iter_bytes()).decode(
                            "utf-8", "replace"
                        )
                        raise LLMError(
                            f"Ollama вернул {response.status_code}: {body[:600]}"
                        )
                    self._emit_progress(
                        stage="prompt", tokens=0, tokens_per_second=0.0
                    )
                    started = time.perf_counter()
                    first_at: float | None = None
                    tokens = 0
                    last_emit = 0.0
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if data.get("error"):
                            raise LLMError(f"Ollama: {data['error']}")
                        message = data.get("message") or {}
                        chunk = message.get("content") or ""
                        if chunk:
                            now = time.perf_counter()
                            if first_at is None:
                                first_at = now
                            tokens += 1
                            if now - last_emit >= 0.2:
                                elapsed = max(0.001, now - first_at)
                                self._emit_progress(
                                    stage="generation",
                                    tokens=tokens,
                                    elapsed=elapsed,
                                    tokens_per_second=tokens / elapsed,
                                )
                                last_emit = now
                            yield chunk
                        if data.get("done"):
                            self.last_stats = self._stats_from_ollama(data)
                            stats = self.last_stats
                            if stats:
                                self._emit_progress(
                                    stage="done",
                                    tokens=stats.tokens,
                                    elapsed=stats.seconds,
                                    tokens_per_second=stats.tokens_per_second,
                                )
                            else:
                                elapsed = max(
                                    0.001,
                                    time.perf_counter() - (first_at or started),
                                )
                                self._emit_progress(
                                    stage="done",
                                    tokens=tokens,
                                    elapsed=elapsed,
                                    tokens_per_second=tokens / elapsed,
                                )
                            break
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Не могу подключиться к Ollama по {self.cfg.base_url}. "
                "Запусти `ollama serve` и проверь порт."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(
                "Ollama не ответила за отведённое время. "
                "Модель слишком большая для этого железа?"
            ) from exc

    @staticmethod
    def _stats_from_ollama(data: dict[str, Any]) -> GenerationStats | None:
        count = int(data.get("eval_count") or 0)
        duration = int(data.get("eval_duration") or 0)
        if not count or not duration:
            return None
        seconds = duration / 1_000_000_000
        prompt_tokens = int(data.get("prompt_eval_count") or 0) or None
        prompt_duration = int(data.get("prompt_eval_duration") or 0)
        return GenerationStats(
            tokens=count,
            seconds=seconds,
            tokens_per_second=count / seconds if seconds else 0.0,
            prompt_tokens=prompt_tokens,
            prompt_seconds=(
                prompt_duration / 1_000_000_000 if prompt_duration else None
            ),
        )

    def health(self) -> str:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self.cfg.base_url.rstrip("/") + "/api/tags")
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Ollama недоступна по {self.cfg.base_url}: {exc}"
            ) from exc
        models = [m.get("name", "") for m in data.get("models", [])]
        return f"Ollama на связи, моделей: {len(models)}"

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self.cfg.base_url.rstrip("/") + "/api/tags")
                response.raise_for_status()
                data = response.json()
        except Exception:  # noqa: BLE001
            return []
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def loaded_models(self) -> list[dict[str, Any]]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self.cfg.base_url.rstrip("/") + "/api/ps")
                response.raise_for_status()
                data = response.json()
        except Exception:  # noqa: BLE001
            return []
        models = data.get("models") or []
        return [model for model in models if isinstance(model, dict)]

    def _keep_alive(self, keep_alive: int | float | str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "keep_alive": keep_alive,
            "stream": False,
        }
        if keep_alive != 0:
            payload["options"] = {"num_ctx": self.cfg.num_ctx}
        url = self.cfg.base_url.rstrip("/") + "/api/generate"
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.cfg.timeout, connect=5.0)
            ) as client:
                response = client.post(url, json=payload)
                if response.status_code >= 400:
                    body = response.text[:600]
                    raise LLMError(
                        f"Ollama вернул {response.status_code}: {body}"
                    )
                data = response.json()
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Не могу подключиться к Ollama по {self.cfg.base_url}."
            ) from exc
        return data if isinstance(data, dict) else {}

    def load_model(self) -> str:
        self._keep_alive(-1)
        return f"{self.cfg.model} загружена в память"

    def unload_model(self) -> str:
        self._keep_alive(0)
        return f"{self.cfg.model} выгружена из памяти"


class OpenAILLM(BaseLLM):
    name = "openai"

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Iterator[str]:
        self.last_stats = None
        self._emit_progress(
            stage="connecting", tokens=0, tokens_per_second=0.0
        )
        start = time.perf_counter()
        first_token_at: float | None = None
        chars = 0
        usage_tokens: int | None = None
        last_emit = 0.0
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.cfg.timeout, connect=5.0)
            ) as client:
                with client.stream(
                    "POST", url, json=payload, headers=headers
                ) as response:
                    if response.status_code >= 400:
                        body = b"".join(response.iter_bytes()).decode(
                            "utf-8", "replace"
                        )
                        raise LLMError(
                            f"Сервер вернул {response.status_code}: {body[:600]}"
                        )
                    self._emit_progress(
                        stage="prompt", tokens=0, tokens_per_second=0.0
                    )
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("error"):
                            raise LLMError(f"Сервер: {obj['error']}")
                        usage = obj.get("usage")
                        if isinstance(usage, dict):
                            value = usage.get("completion_tokens")
                            if value:
                                usage_tokens = int(value)
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        first = choices[0]
                        delta = first.get("delta") or {}
                        chunk = delta.get("content") or first.get("text") or ""
                        if chunk:
                            now = time.perf_counter()
                            if first_token_at is None:
                                first_token_at = now
                            chars += len(chunk)
                            estimated = usage_tokens or max(
                                1, round(chars / 3)
                            )
                            if now - last_emit >= 0.2:
                                elapsed = max(
                                    0.001, now - first_token_at
                                )
                                self._emit_progress(
                                    stage="generation",
                                    tokens=estimated,
                                    elapsed=elapsed,
                                    tokens_per_second=estimated / elapsed,
                                )
                                last_emit = now
                            yield chunk
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Не могу подключиться к {self.cfg.base_url}."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError("Сервер не ответил вовремя.") from exc

        if chars:
            seconds = max(0.001, time.perf_counter() - (first_token_at or start))
            tokens = usage_tokens or max(1, round(chars / 3))
            self.last_stats = GenerationStats(
                tokens=tokens,
                seconds=seconds,
                tokens_per_second=tokens / seconds,
                estimated=usage_tokens is None,
            )
            self._emit_progress(
                stage="done",
                tokens=tokens,
                elapsed=seconds,
                tokens_per_second=tokens / seconds,
            )

    def health(self) -> str:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self.cfg.base_url.rstrip("/") + "/models")
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Сервер недоступен: {exc}") from exc
        return "OpenAI-совместимый сервер на связи"

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self.cfg.base_url.rstrip("/") + "/models")
                response.raise_for_status()
                data = response.json()
        except Exception:  # noqa: BLE001
            return []
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


class MockLLM(BaseLLM):
    """Заглушка для проверки интерфейса без модели."""

    name = "mock"

    REPLIES = [
        "ну наконец-то. я уже думала, ты опять залип в свой сервер. рассказывай.",
        "хм. звучит как план, который ты придумал за две минуты и уже считаешь гениальным.",
        "ладно, допустим. но если ты опять будешь молчать полчаса, я обижусь. серьёзно.",
        "подожди, ты серьёзно? перескажи ещё раз, только медленно.",
        "не, я не обиделась. я просто запомнила.",
        "ты опять всё усложняешь. возьми и сделай уже.",
    ]

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        self.counter = 0
        self.loaded = False

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Iterator[str]:
        self.last_stats = None
        self._emit_progress(
            stage="prompt", tokens=0, tokens_per_second=0.0
        )
        system = messages[0]["content"] if messages else ""
        if "Режим заготовок" in system:
            index = self.counter if seed is None else seed
            self.counter += 1
            beats = [
                "явился, блин",
                "ты вообще думал?",
                "кофе будешь?",
                "может, объяснишь?",
                "я не в настроении",
            ]
            text = beats[int(index) % len(beats)]
            self._mock_stats(text)
            tokens = max(1, len(text) // 4)
            self._emit_progress(
                stage="generation", tokens=tokens, tokens_per_second=20.0
            )
            yield text
            self._emit_progress(
                stage="done", tokens=tokens, tokens_per_second=20.0
            )
            return
        if "Разворот заготовки" in system:
            match = re.search(
                r"Выбранная заготовка её ответа: «(.+?)»", system
            )
            beat = match.group(1) if match else "ну и ладно"
            text = (
                f"*отводит взгляд* {beat}. "
                "*садится обратно* я уже думала, ты не придёшь."
            )
            self._mock_stats(text)
            tokens = max(1, len(text) // 4)
            self._emit_progress(
                stage="generation", tokens=tokens, tokens_per_second=20.0
            )
            yield text
            self._emit_progress(
                stage="done", tokens=tokens, tokens_per_second=20.0
            )
            return
        if "Придумай конкретную" in system and "JSON" in system:
            index = self.counter if seed is None else seed
            self.counter += 1
            cards = [
                {
                    "situation": "она нашла старый профиль в соцсетях и ревнует",
                    "relationship": "встречаетесь три месяца",
                    "her_goal": "получить объяснение, а не оправдание",
                    "tone": "серьёзный, с колкостями",
                    "restrictions": "не мириться в первые три хода",
                    "opener": "her",
                },
                {
                    "situation": "вы договорились встретиться, но он опоздал на час",
                    "relationship": "дружите давно, недавно начали встречаться",
                    "her_goal": "понять, насколько она ему важна",
                    "tone": "раздражённый, но с надеждой",
                    "restrictions": "не оправдываться работой больше двух раз",
                    "opener": "her",
                },
            ]
            text = json.dumps(
                cards[int(index) % len(cards)], ensure_ascii=False
            )
            self._mock_stats(text)
            tokens = max(1, len(text) // 4)
            self._emit_progress(
                stage="generation", tokens=tokens, tokens_per_second=20.0
            )
            yield text
            self._emit_progress(
                stage="done", tokens=tokens, tokens_per_second=20.0
            )
            return

        index = self.counter if seed is None else seed
        self.counter += 1
        text = self.REPLIES[int(index) % len(self.REPLIES)]
        prefix = f"[mock T={temperature:.2f}] " if temperature is not None else "[mock] "
        full = prefix + text
        self._mock_stats(full)
        tokens = max(1, len(full) // 4)
        self._emit_progress(
            stage="generation", tokens=tokens, tokens_per_second=20.0
        )
        for word in full.split(" "):
            yield word + " "
        self._emit_progress(
            stage="done", tokens=tokens, tokens_per_second=20.0
        )

    def _mock_stats(self, text: str) -> None:
        tokens = max(1, len(text) // 4)
        self.last_stats = GenerationStats(
            tokens=tokens,
            seconds=tokens / 20,
            tokens_per_second=20.0,
            estimated=True,
        )

    def health(self) -> str:
        return "mock-бэкенд, модель не нужна"

    def loaded_models(self) -> list[dict[str, Any]]:
        if not self.loaded:
            return []
        return [
            {
                "name": self.cfg.model,
                "size": 0,
                "size_vram": 0,
                "context_length": self.cfg.num_ctx,
                "expires_at": None,
            }
        ]

    def load_model(self) -> str:
        self.loaded = True
        return f"{self.cfg.model} загружена в память (mock)"

    def unload_model(self) -> str:
        self.loaded = False
        return f"{self.cfg.model} выгружена из памяти (mock)"


def make_llm(cfg: LLMConfig) -> BaseLLM:
    backend = cfg.backend.lower()
    if backend == "ollama":
        return OllamaLLM(cfg)
    if backend == "openai":
        return OpenAILLM(cfg)
    if backend == "mock":
        return MockLLM(cfg)
    raise ValueError(f"Неизвестный backend: {cfg.backend}")
