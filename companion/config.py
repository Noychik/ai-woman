from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    api_key: str = ""
    temperature: float = 0.85
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    repeat_last_n: int = 128
    max_tokens: int = 300
    num_ctx: int = 8192
    num_thread: int = 0
    think: bool = False
    # Для «думающих» Qwen3: добавить /no_think в системный промпт.
    no_think: bool = False
    timeout: float = 300.0
    # Сколько держать модель в памяти после запроса: -1 — всегда,
    # 0 — сразу выгрузить, или строка вида "10m".
    keep_alive: int | float | str = -1


@dataclass
class ChatConfig:
    history_messages: int = 40
    max_context_chars: int = 18000
    summarize_after: int = 36
    variants: int = 0
    variant_mode: str = "short"
    short_max_tokens: int = 60
    variant_temperatures: list[float] = field(
        default_factory=lambda: [0.7, 0.9, 1.05]
    )
    language: str = "ru"


@dataclass
class PathsConfig:
    db: str = "data/chat.db"
    preferences: str = "data/preferences.jsonl"
    persona: str = "persona.md"
    exports: str = "data/exports"


@dataclass
class MemoryConfig:
    auto_summarize: bool = True
    auto_facts: bool = True


@dataclass
class CaseConfig:
    variants: int = 3
    temperatures: list[float] = field(
        default_factory=lambda: [0.7, 0.9, 1.05]
    )
    auto_export: bool = True


@dataclass
class ContentConfig:
    adult: bool = True


@dataclass
class Config:
    root: Path
    llm: LLMConfig
    chat: ChatConfig
    paths: PathsConfig
    memory: MemoryConfig
    case: CaseConfig
    content: ContentConfig

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)


def _section(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def _find_config() -> Path:
    candidates = [
        Path.cwd() / "config.toml",
        Path(__file__).resolve().parent.parent / "config.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Не найден config.toml. Запусти из папки проекта или укажи --config."
    )


def load_config(path: Path | None = None) -> Config:
    config_path = (path or _find_config()).resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    return Config(
        root=config_path.parent,
        llm=_section(LLMConfig, raw.get("llm", {})),
        chat=_section(ChatConfig, raw.get("chat", {})),
        paths=_section(PathsConfig, raw.get("paths", {})),
        memory=_section(MemoryConfig, raw.get("memory", {})),
        case=_section(CaseConfig, raw.get("case", {})),
        content=_section(ContentConfig, raw.get("content", {})),
    )
