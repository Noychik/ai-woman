from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Config
from .llm import (
    BaseLLM,
    GenerationStats,
    LLMError,
    make_llm,
    strip_thinking,
)
from .prompt import build_messages as assemble_messages
from .prompt import (
    case_suggestion_messages,
    expand_instruction,
    parse_case_card,
    parse_short_variants,
    parse_summary,
    short_variant_instruction,
    summarization_messages,
)
from .store import Case, Message, Run, Store

OPENING_INSTRUCTION = (
    "Начни сцену. Ты уже в ситуации. Напиши первую реплику от лица персонажа. "
    "Не пересказывай ситуацию, не описывай обстановку, не задавай метавопросов."
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def persona_name(persona: str) -> str:
    match = re.search(r"^#\s+(.+)$", persona, re.MULTILINE)
    return match.group(1).strip() if match else "Она"


# Русская раскладка → латиница. Нужно, чтобы \мфк превращалось в "var".
RU_KEYS = "йцукенгшщзхъфывапролджэячсмитьбю"
LATIN_KEYS = "qwertyuiop[]asdfghjkl;'zxcvbnm,."
RU_TO_LATIN = str.maketrans(RU_KEYS, LATIN_KEYS)

LATIN_ALIASES = {
    "v": "variants",
    "var": "variants",
    "variants": "variants",
    "h": "help",
    "help": "help",
    "?": "help",
    "q": "quit",
    "quit": "quit",
    "exit": "quit",
    "n": "new",
    "new": "new",
    "s": "sessions",
    "sess": "sessions",
    "sessions": "sessions",
    "sw": "switch",
    "switch": "switch",
    "t": "temp",
    "temp": "temp",
    "m": "model",
    "model": "model",
    "p": "persona",
    "persona": "persona",
    "r": "remember",
    "remember": "remember",
    "f": "facts",
    "facts": "facts",
    "forget": "forget",
    "c": "compact",
    "compact": "compact",
    "a": "again",
    "again": "again",
    "u": "undo",
    "undo": "undo",
    "e": "edit",
    "edit": "edit",
    "l": "like",
    "like": "like",
    "d": "dislike",
    "dislike": "dislike",
    "x": "export",
    "export": "export",
    "st": "stats",
    "stats": "stats",
    "k": "case",
    "case": "case",
    "ks": "cases",
    "cases": "cases",
    "finish": "finish",
    "resume": "resume",
    "hint": "hint",
    "case_edit": "case_edit",
    "case_suggest": "case_suggest",
    "suggest": "case_suggest",
    "load": "load",
    "unload": "unload",
    "ps": "loaded",
    "loaded": "loaded",
    "+": "load",
    "-": "unload",
}

# Явные русские слова. Они имеют приоритет над транслитерацией,
# поэтому \к — это кейс, а не remember (r на русской раскладке).
RU_ALIASES = {
    "м": "variants",
    "в": "variants",
    "мфк": "variants",
    "вар": "variants",
    "варианты": "variants",
    "р": "help",
    "помощь": "help",
    "хелп": "help",
    "й": "quit",
    "выход": "quit",
    "т": "new",
    "новый": "new",
    "новая": "new",
    "ы": "sessions",
    "сессии": "sessions",
    "е": "temp",
    "темп": "temp",
    "температура": "temp",
    "ь": "model",
    "модель": "model",
    "з": "persona",
    "персонаж": "persona",
    "к": "case",
    "кейс": "case",
    "к+": "case_edit",
    "кп": "case_suggest",
    "придумай": "case_suggest",
    "предложи": "case_suggest",
    "кс": "cases",
    "ксы": "cases",
    "кейсы": "cases",
    "а": "facts",
    "факты": "facts",
    "с": "compact",
    "сжать": "compact",
    "ф": "again",
    "ещё": "again",
    "еще": "again",
    "г": "undo",
    "отменить": "undo",
    "отмена": "undo",
    "у": "edit",
    "править": "edit",
    "правка": "edit",
    "д": "like",
    "нравится": "like",
    "не": "dislike",
    "дизлайк": "dislike",
    "ч": "finish",
    "завершить": "finish",
    "п": "resume",
    "продолжить": "resume",
    "пд": "hint",
    "подсказка": "hint",
    "загрузи": "load",
    "загрузить": "load",
    "зг": "load",
    "выгрузи": "unload",
    "выгрузить": "unload",
    "вг": "unload",
    "пс": "loaded",
    "загружено": "loaded",
    "экспорт": "export",
    "ые": "stats",
    "статы": "stats",
    "статистика": "stats",
    "запомни": "remember",
    "забудь": "forget",
}


def expand_alias(line: str) -> str:
    """\\мфк → /variants, \\к 12 → /case 12, \\ч → /finish."""
    if not line.startswith("\\"):
        return line
    body = line[1:].strip()
    if not body:
        return line
    token, _, rest = body.partition(" ")
    token = token.lower()

    command = RU_ALIASES.get(token)
    if command is None:
        latin = token.translate(RU_TO_LATIN)
        command = LATIN_ALIASES.get(latin) or LATIN_ALIASES.get(token)
    if command is None:
        return line

    return "/" + command + ((" " + rest) if rest else "")


def parse_id(raw: str) -> int | None:
    raw = raw.strip().lstrip("#")
    return int(raw) if raw.isdigit() else None


def build_key_bindings() -> KeyBindings:
    kb = KeyBindings()

    # Обычный Backspace всегда удаляет один символ, независимо от того,
    # как терминал его кодирует (\x7f или \x08 = ControlH).
    @kb.add(Keys.Backspace)
    @kb.add(Keys.ControlH)
    def _delete_char(event) -> None:
        buff = event.current_buffer
        if buff.document.cursor_position:
            buff.delete_before_cursor(count=1)

    @kb.add(Keys.ControlLeft)
    def _word_left(event) -> None:
        buff = event.current_buffer
        offset = buff.document.find_previous_word_beginning(count=1)
        if offset is not None:
            buff.cursor_position += offset

    @kb.add(Keys.ControlRight)
    def _word_right(event) -> None:
        buff = event.current_buffer
        offset = buff.document.find_next_word_beginning(count=1)
        if offset is not None:
            buff.cursor_position += offset

    @kb.add("c-w")
    def _delete_word_left_cw(event) -> None:
        buff = event.current_buffer
        offset = buff.document.find_previous_word_beginning(count=1)
        if offset is not None:
            buff.delete_before_cursor(count=-offset)

    # Ctrl+Backspace во многих терминалах приходит как ESC + DEL (0x7f).
    @kb.add("escape", Keys.Backspace)
    def _delete_word_left_esc(event) -> None:
        buff = event.current_buffer
        offset = buff.document.find_previous_word_beginning(count=1)
        if offset is not None:
            buff.delete_before_cursor(count=-offset)

    @kb.add(Keys.ControlDelete)
    def _delete_word_right(event) -> None:
        buff = event.current_buffer
        offset = buff.document.find_next_word_beginning(count=1)
        if offset is not None:
            buff.delete(count=offset)

    @kb.add(Keys.Home)
    def _home(event) -> None:
        buff = event.current_buffer
        buff.cursor_position += buff.document.get_start_of_line_position()

    @kb.add(Keys.End)
    def _end(event) -> None:
        buff = event.current_buffer
        buff.cursor_position += buff.document.get_end_of_line_position()

    @kb.add(Keys.ControlHome)
    def _buffer_home(event) -> None:
        event.current_buffer.cursor_position = 0

    @kb.add(Keys.ControlEnd)
    def _buffer_end(event) -> None:
        event.current_buffer.cursor_position = len(event.current_buffer.text)

    return kb


def wrap_words(text: str, width: int) -> str:
    """Перенос по словам; длинные слова не разрезаются без необходимости."""
    if width <= 10:
        return text
    lines: list[str] = []
    for line in text.splitlines() or [""]:
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        lines.extend(wrapped or [""])
    return "\n".join(lines)


class GenerationProgress:
    """Живой статус: этап, токены и скорость прямо во время генерации."""

    STAGE_LABELS = {
        "connecting": "подключение",
        "load": "загрузка модели и обработка контекста",
        "prompt": "обработка контекста",
        "generation": "генерация",
        "done": "готово",
    }

    def __init__(
        self,
        console: Console,
        title: str = "",
        show_answer: bool = False,
        transient: bool = True,
    ):
        self.console = console
        self.title = title
        self.show_answer = show_answer
        self.transient = transient
        self.enabled = console.is_terminal
        self.live: Live | None = None
        self.answer = ""
        self.stage = "connecting"
        self.tokens = 0
        self.tps = 0.0
        self.started = time.perf_counter()
        self.first_token_at: float | None = None
        self.last_render = 0.0

    def start(self) -> None:
        self.started = time.perf_counter()
        if not self.enabled:
            return
        self.live = Live(
            get_renderable=self._render,
            console=self.console,
            refresh_per_second=8,
            transient=self.transient,
            vertical_overflow="visible",
        )
        self.live.start()
        self.last_render = time.perf_counter()

    def on_event(self, event: dict) -> None:
        stage = event.get("stage")
        if stage:
            self.stage = str(stage)
        if event.get("tokens") is not None:
            self.tokens = int(event["tokens"])
        if event.get("tokens_per_second") is not None:
            self.tps = float(event["tokens_per_second"])
        if self.stage == "generation" and self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        self._refresh(force=self.stage == "done")

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self.answer += chunk
        if not self.enabled:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            return
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        self._refresh()

    def finish(self, stats: GenerationStats | None) -> None:
        if stats:
            self.tokens = stats.tokens
            self.tps = stats.tokens_per_second
        self.stage = "done"
        if self.live:
            self.live.update(self._render())
            self.live.stop()
            self.live = None
            if self.show_answer:
                self.console.print()
        elif self.show_answer:
            self.console.print()
            if stats:
                tokens = (
                    f"{self.tokens} "
                    f"{plural(self.tokens, 'токен', 'токена', 'токенов')}"
                )
                self.console.print(
                    f"[dim]⏱ {self.tps:.1f} ток/с · {tokens}[/]"
                )

    def _refresh(self, force: bool = False) -> None:
        if not self.live:
            return
        now = time.perf_counter()
        if not force and now - self.last_render < 0.12:
            return
        self.last_render = now
        self.live.update(self._render())

    def _tps(self) -> float:
        if self.tps:
            return self.tps
        if self.first_token_at and self.tokens:
            elapsed = max(0.001, time.perf_counter() - self.first_token_at)
            return self.tokens / elapsed
        return 0.0

    def _render(self) -> Text:
        text = Text()
        if self.show_answer:
            text.append(f"{self.title} › ", style="bold magenta")
            text.append(
                wrap_words(self.answer, max(20, self.console.width - 2))
            )
            text.append("\n")
        elapsed = time.perf_counter() - self.started
        if self.stage in ("connecting", "load", "prompt"):
            label = self.STAGE_LABELS.get(self.stage, self.stage)
            text.append(f"⏳ {label} · {elapsed:.0f} с", style="dim")
        elif self.stage == "error":
            text.append("✗ ошибка", style="red")
        else:
            tps = self._tps()
            if self.stage == "generation" and (
                self.tokens < 3 or tps > 500
            ):
                text.append("⚡ генерация…", style="dim")
            else:
                prefix = (
                    "⚡ генерация"
                    if self.stage == "generation"
                    else "✓ готово"
                )
                text.append(
                    f"{prefix} · {tps:.1f} ток/с · "
                    f"{self.tokens} токенов",
                    style="dim",
                )
        return text


def plural(count: int, one: str, few: str, many: str) -> str:
    tail = count % 10
    if tail == 1 and count % 100 != 11:
        return one
    if 2 <= tail <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def human_size(size: int | float) -> str:
    value = float(size or 0)
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} ГБ"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f} МБ"
    if value >= 1024:
        return f"{value / 1024:.0f} КБ"
    return f"{int(value)} Б"


def format_expires(value: str | None) -> str:
    if not value:
        return "—"
    try:
        expires = datetime.fromisoformat(value)
    except ValueError:
        return value
    if expires.year >= 2100:
        return "бессрочно"
    return expires.astimezone().strftime("%d.%m %H:%M")


class App:
    def __init__(self, cfg: Config, session_id: int | None = None):
        self.cfg = cfg
        self.console = Console()
        self.store = Store(cfg.path(cfg.paths.db))
        self.llm: BaseLLM = make_llm(cfg.llm)
        self.persona_path = cfg.path(cfg.paths.persona)
        self.persona = self._load_persona()
        self.name = persona_name(self.persona)
        self.variants = max(0, cfg.chat.variants)
        self.case_variants = max(2, cfg.case.variants)
        self.variant_mode = cfg.chat.variant_mode.lower()
        if self.variant_mode not in ("short", "full"):
            self.variant_mode = "short"
        self.mode = cfg.chat.mode.lower()
        if self.mode not in ("roleplay", "messaging"):
            self.mode = "roleplay"
        self.adult = cfg.content.adult
        self.temperature = cfg.llm.temperature
        self.seed_base = 0

        self.session_id = self._pick_session(session_id)
        self.active_run: Run | None = self.store.active_run_for_session(
            self.session_id
        )
        self.interactive = sys.stdin.isatty()
        self.session = self._make_prompt_session()

    # ── запуск ────────────────────────────────────────────────────────────

    def _pick_session(self, session_id: int | None) -> int:
        if session_id is not None:
            if not self.store.get_session(session_id):
                raise SystemExit(f"Сессии {session_id} нет.")
            return session_id
        latest = self.store.latest_session()
        if latest is not None:
            return latest
        return self.store.create_session("первый разговор")

    def _make_prompt_session(self) -> PromptSession | None:
        if not self.interactive:
            return None
        history_path = self.cfg.path("data/history.txt")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        style = Style.from_dict({"prompt": "ansimagenta bold"})
        return PromptSession(
            history=FileHistory(str(history_path)),
            style=style,
            key_bindings=build_key_bindings(),
            wrap_lines=True,
        )

    def ask(self, prompt: str) -> str:
        if self.session is not None:
            return self.session.prompt([("class:prompt", prompt)])
        sys.stdout.write(prompt)
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\n")

    def _status(self, text: str):
        return self.console.status(text) if self.interactive else nullcontext()

    def _complete_with_progress(
        self, messages: list[dict[str, str]], **kwargs
    ) -> str:
        progress = GenerationProgress(
            self.console, show_answer=False, transient=True
        )
        self.llm.on_progress = progress.on_event
        progress.start()
        try:
            return self.llm.complete(messages, **kwargs)
        finally:
            self.llm.on_progress = None
            progress.finish(self.llm.last_stats)

    def _print_speed(self, stats: GenerationStats | None) -> None:
        if not stats or stats.tokens_per_second <= 0:
            return
        prefix = "~" if stats.estimated else ""
        tokens = (
            f"{stats.tokens} "
            f"{plural(stats.tokens, 'токен', 'токена', 'токенов')}"
        )
        self.console.print(
            f"[dim]⏱ {prefix}{stats.tokens_per_second:.1f} ток/с · {tokens}[/]"
        )

    def run(self) -> None:
        self._banner()
        self._health()
        while True:
            try:
                line = self.ask(self._input_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            if not line:
                continue
            if line.startswith("\\"):
                expanded = expand_alias(line)
                if expanded == line:
                    self.console.print("[yellow]Не знаю такую команду. /help[/]")
                    continue
                line = expanded
            if line.startswith("/"):
                try:
                    if not self.handle_command(line):
                        break
                except LLMError as exc:
                    self.console.print(f"[red]{exc}[/]")
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]прервано[/]")
                continue
            try:
                if self.active_run:
                    self.case_turn(line)
                else:
                    self.chat(line)
            except LLMError as exc:
                self.console.print(f"[red]{exc}[/]")
            except KeyboardInterrupt:
                self.console.print("\n[yellow]прервано[/]")
        self.store.close()

    def _input_prompt(self) -> str:
        if not self.active_run:
            return "ты › "
        case = self.store.get_case(self.active_run.case_id)
        case_id = case.id if case else "?"
        turn = self.store.count_turns(self.active_run.session_id) + 1
        return f"сцена {case_id} · ход {turn} › "

    def _banner(self) -> None:
        session = self.store.get_session(self.session_id)
        started = session["started_at"][:16].replace("T", " ") if session else "?"
        lines = [
            f"[bold]companion[/] · [cyan]{self.llm.cfg.model}[/] "
            f"([dim]{self.llm.cfg.backend}[/])",
            f"сессия {self.session_id} · начата {started} · "
            f"сообщений {self.store.count_messages(self.session_id, include_hidden=False)} · "
            f"фактов {len(self.store.facts())} · "
            f"кейсов {len(self.store.cases())}",
        ]
        if self.active_run:
            case = self.store.get_case(self.active_run.case_id)
            title = case.situation if case else "сцена"
            lines.append(
                f"[green]прогон #{self.active_run.id}[/] · кейс #{self.active_run.case_id} · "
                f"{escape(title[:70])}"
            )
        else:
            pending = self.store.latest_active_run()
            if pending:
                lines.append(f"[dim]есть незакрытый прогон #{pending.id}; \\п — продолжить[/]")
        if self.llm.cfg.backend == "ollama":
            loaded = self.llm.loaded_models()
            if loaded:
                total = sum(int(model.get("size") or 0) for model in loaded)
                lines.append(
                    f"[dim]в памяти: {len(loaded)} · {human_size(total)} · "
                    f"\\- выгрузить[/]"
                )
            else:
                lines.append("[dim]модель не загружена · \\+ загрузить[/]")
        lines.append("[dim]/help — команды, \\м 3 — режим выбора вариантов[/]")
        self.console.print(Panel("\n".join(lines), border_style="magenta", expand=False))

    def _health(self) -> None:
        try:
            self.console.print(f"[dim]{self.llm.health()}[/]")
        except LLMError as exc:
            self.console.print(f"[red]{exc}[/]")
            self.console.print(
                "[yellow]Команды работают, но генерация будет падать, "
                "пока бэкенд не поднят.[/]"
            )
            return
        warning = self.llm.warn_if_model_missing()
        if warning:
            self.console.print(f"[yellow]{warning}[/]")

    def _load_model(self) -> None:
        try:
            with self._status("загружаю модель"):
                message = self.llm.load_model()
        except LLMError as exc:
            self.console.print(f"[red]{exc}[/]")
            return
        self.console.print(f"[green]✓ {escape(message)}[/]")
        self._loaded_table(quiet=True)

    def _unload_model(self) -> None:
        try:
            with self._status("выгружаю модель"):
                message = self.llm.unload_model()
        except LLMError as exc:
            self.console.print(f"[red]{exc}[/]")
            return
        self.console.print(f"[green]✓ {escape(message)}[/]")
        self.console.print(
            "[dim]следующее сообщение загрузит её заново[/]"
        )

    def _loaded_table(self, quiet: bool = False) -> None:
        models = self.llm.loaded_models()
        if not models:
            if not quiet:
                self.console.print(
                    "[dim]модель не загружена. \\+ или /load — загрузить[/]"
                )
            return
        table = Table(border_style="magenta")
        table.add_column("модель")
        table.add_column("размер", justify="right")
        table.add_column("контекст", justify="right")
        table.add_column("до")
        for model in models:
            name = str(model.get("name") or "?")
            size = int(model.get("size") or 0)
            vram = int(model.get("size_vram") or 0)
            size_text = human_size(size)
            if vram:
                size_text += f" (VRAM {human_size(vram)})"
            context = model.get("context_length") or "—"
            table.add_row(
                escape(name),
                size_text,
                str(context),
                format_expires(model.get("expires_at")),
            )
        self.console.print(table)

    # ── диалог ────────────────────────────────────────────────────────────

    def _load_persona(self) -> str:
        if not self.persona_path.exists():
            self.console.print(f"[red]Нет файла персонажа: {self.persona_path}[/]")
            return "Ты — собеседник. Отвечай по-русски."
        return self.persona_path.read_text(encoding="utf-8")

    def _messages_for_model(self) -> list[dict[str, str]]:
        summary = self.store.latest_summary(self.session_id)
        after = summary.upto_message_id if summary else 0
        history = self.store.messages(
            self.session_id,
            after_id=after,
            limit=self.cfg.chat.history_messages,
            roles=("user", "assistant"),
            include_hidden=True,
        )
        facts = [fact.text for fact in self.store.facts()]
        scenario = self.active_run.scenario if self.active_run else None
        return assemble_messages(
            persona=self.persona,
            facts=facts,
            summary=summary.text if summary else None,
            history=history,
            max_chars=self.cfg.chat.max_context_chars,
            language=self.cfg.chat.language,
            scenario=scenario,
            adult=self.adult,
            mode=self.mode,
        )

    def _context_for_log(self) -> list[dict[str, str]]:
        return [
            message
            for message in self._messages_for_model()
            if message["content"] != OPENING_INSTRUCTION
        ]

    def chat(self, text: str) -> None:
        user_message = self.store.add_message(self.session_id, "user", text)
        if self.variants > 0:
            if self.variant_mode == "short":
                self._short_variant_turn(user_message)
            else:
                self._variant_turn(user_message)
        else:
            self._stream_turn()
        self._maybe_summarize()

    def _stream_turn(self) -> None:
        messages = self._messages_for_model()
        reply = self._stream_reply(messages)
        if reply:
            self.store.add_message(self.session_id, "assistant", reply)

    def _stream_reply(
        self, messages: list[dict[str, str]], warn_empty: bool = True
    ) -> str:
        progress = GenerationProgress(
            self.console,
            title=self.name,
            show_answer=True,
            transient=False,
        )
        if not progress.enabled:
            sys.stdout.write(f"{self.name} › ")
            sys.stdout.flush()
        self.llm.on_progress = progress.on_event
        progress.start()
        parts: list[str] = []
        buffer = ""
        started = False
        try:
            for chunk in self.llm.stream(
                messages, temperature=self.temperature
            ):
                buffer += chunk
                if not started:
                    if "</think>" in buffer:
                        buffer = buffer.split("</think>", 1)[1]
                        started = True
                    elif "<think>" not in buffer and len(buffer) > 16:
                        started = True
                    else:
                        continue
                if not parts:
                    buffer = buffer.lstrip()
                if buffer:
                    parts.append(buffer)
                    progress.append(buffer)
                    buffer = ""
            if not started and buffer:
                buffer = buffer.lstrip()
                parts.append(buffer)
                progress.append(buffer)
        finally:
            self.llm.on_progress = None
            progress.finish(self.llm.last_stats)
        reply = strip_thinking("".join(parts)).strip()
        if not reply and warn_empty:
            self.console.print("[yellow]Пустой ответ. Попробуй /again.[/]")
        return reply

    def _variant_turn(self, user_message: Message) -> None:
        while True:
            try:
                candidates = self._generate_candidates()
            except LLMError:
                self.store.delete_message(user_message.id)
                raise

            if not candidates:
                self.store.delete_message(user_message.id)
                return

            self._show_candidates(candidates)
            choice = self._choose(candidates)
            if choice is None:
                self.store.delete_message(user_message.id)
                self.console.print("[dim]пропущено[/]")
                return
            if choice == "retry":
                continue

            index, final_text, edited = choice
            stats = candidates[index][2]
            context = self._context_for_log()
            self.store.add_message(self.session_id, "assistant", final_text)
            self._log(
                {
                    "type": "choice",
                    "kind": "chat",
                    "session_id": self.session_id,
                    "model": self.llm.cfg.model,
                    "context": context,
                    "candidates": [
                        {
                            "text": text,
                            "temperature": temp,
                            "stats": stats.as_dict() if stats else None,
                        }
                        for text, temp, stats in candidates
                    ],
                    "chosen": index,
                    "edited": edited,
                    "final": final_text,
                }
            )
            self.console.print(
                f"[green]✓ сохранено[/] [dim](вариант {index + 1})[/]"
            )
            self._print_speed(stats)
            return

    def _with_instruction(
        self, messages: list[dict[str, str]], instruction: str
    ) -> list[dict[str, str]]:
        result = [dict(message) for message in messages]
        result[0]["content"] = (
            result[0]["content"].rstrip() + "\n\n" + instruction
        )
        return result

    def _generate_short_candidates(self, count: int) -> list[str]:
        count = max(1, count)
        temps = self.cfg.chat.variant_temperatures or [0.9]
        base = self.seed_base
        self.seed_base += count
        beats: list[str] = []
        for index in range(count):
            messages = self._with_instruction(
                self._messages_for_model(), short_variant_instruction()
            )
            temperature = temps[index % len(temps)]
            raw = self._complete_with_progress(
                messages,
                temperature=temperature,
                max_tokens=self.cfg.chat.short_max_tokens,
                seed=base + index,
            )
            parsed = parse_short_variants(raw, 1)
            if parsed:
                beats.append(parsed[0])

        result: list[str] = []
        seen: set[str] = set()
        for beat in beats:
            key = beat.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(beat)
        return result[:count]

    def _show_beats(self, beats: list[str]) -> None:
        self.console.print("[bold]короткие заготовки:[/]")
        for index, beat in enumerate(beats, start=1):
            self.console.print(f"  [cyan]{index}.[/] {escape(beat)}")

    def _choose_beat(
        self, beats: list[str]
    ) -> tuple[int, str, bool] | str | None:
        count = len(beats)
        while True:
            raw = self.ask(
                f"выбор [1-{count}, e=править, r=заново, s=пропустить] › "
            ).strip().lower()
            if raw in ("", "s", "skip"):
                return None
            if raw in ("r", "retry"):
                return "retry"
            if raw == "e":
                number = self.ask("номер заготовки › ").strip()
                if not number.isdigit():
                    continue
                index = int(number) - 1
                if 0 <= index < count:
                    edited = self.ask(f"заготовка [{beats[index]}] › ").strip()
                    if edited:
                        return index, edited, edited != beats[index]
                    return index, beats[index], False
                continue
            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < count:
                    return index, beats[index], False

    def _expand_beat(self, beat: str) -> str:
        messages = self._with_instruction(
            self._messages_for_model(),
            expand_instruction(beat, mode=self.mode),
        )
        return self._stream_reply(messages, warn_empty=False)

    def _short_variant_turn(self, user_message: Message) -> None:
        while True:
            try:
                beats = self._generate_short_candidates(self.variants)
                short_stats = self.llm.last_stats
            except LLMError:
                self.store.delete_message(user_message.id)
                raise
            if not beats:
                self.store.delete_message(user_message.id)
                self.console.print("[yellow]Модель не дала заготовок.[/]")
                return
            self._show_beats(beats)
            choice = self._choose_beat(beats)
            if choice is None:
                self.store.delete_message(user_message.id)
                self.console.print("[dim]пропущено[/]")
                return
            if choice == "retry":
                continue

            index, beat, edited = choice
            while True:
                try:
                    reply = self._expand_beat(beat)
                except LLMError:
                    self.store.delete_message(user_message.id)
                    raise
                if reply:
                    break
                try:
                    action = self.ask(
                        "пустой разворот. [r=повторить, s=отменить] › "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    action = "s"
                if action in ("", "r", "retry", "повторить"):
                    continue
                self.store.delete_message(user_message.id)
                self.console.print("[dim]ход отменён[/]")
                return
            context = self._context_for_log()
            stats = self.llm.last_stats
            self.store.add_message(self.session_id, "assistant", reply)
            self._log(
                {
                    "type": "choice",
                    "kind": "chat_short",
                    "session_id": self.session_id,
                    "model": self.llm.cfg.model,
                    "context": context,
                    "short_candidates": beats,
                    "chosen_short": index,
                    "short_edited": edited,
                    "short_stats": (
                        short_stats.as_dict() if short_stats else None
                    ),
                    "final": reply,
                    "stats": stats.as_dict() if stats else None,
                }
            )
            self.console.print("[green]✓ сохранено[/]")
            return

    def _generate_candidates(
        self,
        count: int | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> list[tuple[str, float, GenerationStats | None]]:
        messages = messages or self._messages_for_model()
        if count is None:
            count = self.case_variants if self.active_run else self.variants
        count = max(1, count)
        if self.active_run:
            temps = self.cfg.case.temperatures or [self.temperature]
        else:
            temps = self.cfg.chat.variant_temperatures or [self.temperature]
        base = self.seed_base
        self.seed_base += count
        candidates: list[tuple[str, float, GenerationStats | None]] = []
        for index in range(count):
            temperature = temps[index % len(temps)]
            text = self._complete_with_progress(
                messages,
                temperature=temperature,
                seed=base + index,
            )
            if text:
                candidates.append((text, temperature, self.llm.last_stats))
        return candidates

    def _show_candidates(
        self, candidates: list[tuple[str, float, GenerationStats | None]]
    ) -> None:
        for index, (text, temperature, _stats) in enumerate(candidates, start=1):
            self.console.print(
                Panel(
                    Markdown(text),
                    title=f"[{index}] T={temperature:.2f}",
                    border_style="cyan",
                    expand=False,
                )
            )

    def _choose(
        self, candidates: list[tuple[str, float, GenerationStats | None]]
    ) -> tuple[int, str, bool] | str | None:
        count = len(candidates)
        while True:
            raw = self.ask(
                f"выбор [1-{count}, e=править, r=заново, s=пропустить] › "
            ).strip().lower()
            if raw in ("", "s", "skip"):
                return None
            if raw in ("r", "retry"):
                return "retry"
            if raw == "e":
                raw = self.ask("номер варианта › ").strip()
                if not raw.isdigit():
                    continue
                index = int(raw) - 1
                if 0 <= index < count:
                    final = self.edit_text(candidates[index][0])
                    return index, final, final != candidates[index][0]
                continue
            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < count:
                    return index, candidates[index][0], False

    def edit_text(self, text: str) -> str:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            for candidate in ("nano", "vi", "vim"):
                if shutil.which(candidate):
                    editor = candidate
                    break
        if not editor:
            self.console.print(
                "[yellow]Редактор не найден. Введи текст, пустая строка — конец.[/]"
            )
            lines: list[str] = []
            while True:
                line = self.ask("… ")
                if not line:
                    break
                lines.append(line)
            return "\n".join(lines).strip() or text

        with tempfile.NamedTemporaryFile(
            "w+", suffix=".md", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(text)
            path = handle.name
        try:
            subprocess.run([editor, path], check=False)
            edited = Path(path).read_text(encoding="utf-8").strip()
            return edited or text
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def retry(self) -> None:
        if self.active_run:
            self._case_retry()
            return
        last = self.store.last_message(self.session_id)
        if last and last.role == "assistant":
            self.store.delete_message(last.id)
        last = self.store.last_message(self.session_id)
        if not last or last.role != "user":
            self.console.print("[yellow]Нечего повторять.[/]")
            return
        if self.variants > 0:
            if self.variant_mode == "short":
                self._short_variant_turn(last)
            else:
                self._variant_turn(last)
        else:
            self._stream_turn()
        self._maybe_summarize()

    # ── память ────────────────────────────────────────────────────────────

    def _maybe_summarize(self) -> None:
        if not self.cfg.memory.auto_summarize:
            return
        threshold = min(
            self.cfg.chat.summarize_after, self.cfg.chat.history_messages
        )
        if self.store.count_since_summary(self.session_id) > threshold:
            self.summarize()

    def summarize(self, force: bool = False) -> None:
        summary = self.store.latest_summary(self.session_id)
        after = summary.upto_message_id if summary else 0
        messages = self.store.messages(
            self.session_id,
            after_id=after,
            roles=("user", "assistant"),
            include_hidden=False,
        )
        keep = max(6, self.cfg.chat.history_messages // 2)
        if len(messages) <= keep:
            if force:
                self.console.print("[dim]Пока нечего сжимать.[/]")
            return
        to_summarize = messages[:-keep]
        if len(to_summarize) < 4:
            if force:
                self.console.print("[dim]Слишком мало сообщений для сводки.[/]")
            return

        transcript = "\n".join(
            f"{'Он' if message.role == 'user' else 'Она'}: {message.content}"
            for message in to_summarize
        )
        self.console.print(
            f"[dim]сжимаю {len(to_summarize)} сообщений в память…[/]"
        )
        try:
            raw = self._complete_with_progress(
                summarization_messages(
                    transcript,
                    previous_summary=summary.text if summary else None,
                ),
                temperature=0.3,
                max_tokens=700,
            )
        except LLMError as exc:
            self.console.print(f"[red]Не удалось сжать: {exc}[/]")
            return

        text, facts = parse_summary(raw)
        if not text:
            self.console.print("[red]Модель вернула пустую сводку.[/]")
            return

        upto = to_summarize[-1].id
        self.store.add_summary(self.session_id, upto, text)
        added = 0
        if self.cfg.memory.auto_facts:
            before = len(self.store.facts())
            for fact in facts:
                self.store.add_fact(fact)
            added = len(self.store.facts()) - before
        self.console.print(
            f"[green]✓ память обновлена[/] [dim](сводка + {added} фактов)[/]"
        )

    def _facts(self) -> list[str]:
        return [fact.text for fact in self.store.facts()]

    # ── лог предпочтений ──────────────────────────────────────────────────

    def _log(self, record: dict) -> None:
        path = self.cfg.path(self.cfg.paths.preferences)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": now_iso(), **record}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _context_before_last_assistant(self) -> list[dict[str, str]]:
        summary = self.store.latest_summary(self.session_id)
        after = summary.upto_message_id if summary else 0
        history = self.store.messages(
            self.session_id,
            after_id=after,
            roles=("user", "assistant"),
            include_hidden=True,
        )
        while history and history[-1].role == "assistant":
            history.pop()
        scenario = self.active_run.scenario if self.active_run else None
        messages = assemble_messages(
            persona=self.persona,
            facts=self._facts(),
            summary=summary.text if summary else None,
            history=history,
            max_chars=self.cfg.chat.max_context_chars,
            language=self.cfg.chat.language,
            scenario=scenario,
            adult=self.adult,
            mode=self.mode,
        )
        return [
            message
            for message in messages
            if message["content"] != OPENING_INSTRUCTION
        ]

    # ── кейсы ─────────────────────────────────────────────────────────────

    def _case(self, raw: str) -> None:
        raw = raw.strip()
        if not raw:
            self._case_mode()
            return
        if raw.startswith("+"):
            self._case_edit(raw[1:].strip())
            return

        action, _, argument = raw.partition(" ")
        action_lower = action.lower()
        if action.isdigit():
            self._case_start(action)
        elif action_lower in ("list", "список"):
            self._cases_table(
                include_archived=argument.lower() in ("all", "архив", "archive")
            )
        elif action_lower in ("+", "edit", "редактировать"):
            self._case_edit(argument)
        elif action_lower in ("play", "играть", "сыграть"):
            self._case_start(argument)
        elif action_lower in ("finish", "завершить", "конец"):
            self._case_finish()
        elif action_lower in ("resume", "продолжить", "дальше"):
            self._case_resume()
        elif action_lower in ("hint", "подсказка"):
            self._case_hint()
        elif action_lower in ("del", "delete", "удалить"):
            self._case_delete(argument)
        elif action_lower in ("done", "готово", "закрыт"):
            self._case_status(argument, "done")
        elif action_lower in ("export", "экспорт"):
            self._case_export()
        elif action_lower in ("suggest", "придумай", "предложи", "идея"):
            self._case_suggest(argument)
        elif action_lower in ("purge", "hard-delete", "удалитьнавсегда"):
            self._case_purge(argument)
        else:
            self._case_add(raw)

    def _case_add(self, text: str) -> None:
        try:
            case = self.store.add_case(text)
        except ValueError as exc:
            self.console.print(f"[yellow]{exc}[/]")
            return
        self.console.print(
            f"[green]✓ кейс #{case.id}[/] "
            f"[dim](всего {len(self.store.cases())}; \\к+ {case.id} — дополнить)[/]"
        )

    def _case_suggest(self, raw: str) -> None:
        seed = raw.strip() or None
        existing = [
            case.situation
            for case in self.store.cases(include_archived=True)
        ]
        while True:
            messages = case_suggestion_messages(
                self.persona, seed, existing, adult=self.adult
            )
            try:
                reply = self._complete_with_progress(
                    messages, temperature=0.9, max_tokens=500
                )
            except LLMError as exc:
                self.console.print(f"[red]{exc}[/]")
                return

            card = parse_case_card(reply)
            if not card:
                self.console.print(
                    "[red]Модель вернула не JSON. Сырой ответ:[/]"
                )
                self.console.print(escape(reply[:1500]))
                return

            while True:
                self._show_card(card)
                action = self.ask(
                    "сохранить? [Y/n/e/r] › "
                ).strip().lower()
                if action in ("", "y", "yes", "д", "да"):
                    situation = card["situation"]
                    fields = {
                        key: value
                        for key, value in card.items()
                        if key != "situation"
                    }
                    try:
                        case = self.store.add_case(situation, **fields)
                    except ValueError as exc:
                        self.console.print(f"[yellow]{exc}[/]")
                        return
                    self.console.print(
                        f"[green]✓ кейс #{case.id}[/] "
                        f"[dim](\\к {case.id} — играть, \\к+ {case.id} — править)[/]"
                    )
                    return
                if action in ("n", "no", "н", "нет"):
                    self.console.print("[dim]отменено[/]")
                    return
                if action in ("e", "edit", "у", "править"):
                    card = self._edit_card(card)
                    continue
                if action in ("r", "retry", "заново"):
                    existing.append(card["situation"])
                    break

    def _show_card(self, card: dict[str, str]) -> None:
        lines = ["[bold]предложение кейса[/]", escape(card["situation"])]
        fields = [
            ("отношения", "relationship"),
            ("её цель", "her_goal"),
            ("тон", "tone"),
            ("нельзя", "restrictions"),
        ]
        for label, key in fields:
            value = card.get(key)
            if value:
                lines.append(f"[dim]{label}:[/] {escape(value)}")
        self.console.print(
            Panel("\n".join(lines), border_style="magenta", expand=False)
        )

    def _edit_card(self, card: dict[str, str]) -> dict[str, str]:
        updated = dict(card)
        fields = [
            ("situation", "ситуация"),
            ("relationship", "отношения"),
            ("her_goal", "её цель"),
            ("tone", "тон"),
            ("restrictions", "нельзя"),
            ("opener", "кто начинает (her/me)"),
        ]
        for key, label in fields:
            current = updated.get(key, "")
            try:
                value = self.ask(f"{label} [{current}] › ").strip()
            except (EOFError, KeyboardInterrupt):
                return updated
            if value:
                updated[key] = value
        return updated

    def _case_mode(self) -> None:
        before = len(self.store.cases())
        self.console.print(
            "[dim]Пиши по одному кейсу в строку. Пустая строка — выход. "
            "Поля можно дописать позже: \\к+ ID[/]"
        )
        while True:
            try:
                line = self.ask("кейс › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            if line.startswith(("/", "\\")):
                self.console.print("[dim]выход из режима кейсов[/]")
                break
            try:
                case = self.store.add_case(line)
            except ValueError:
                self.console.print("[yellow]слишком коротко[/]")
                continue
            self.console.print(f"[green]✓ #{case.id}[/]")
        after = len(self.store.cases())
        self.console.print(f"[green]добавлено {after - before}, всего {after}[/]")

    def _cases_table(self, include_archived: bool = False) -> None:
        cases = self.store.cases(include_archived=include_archived)
        if not cases:
            self.console.print(
                "[dim]кейсов нет. \\к — добавить, \\к текст — один кейс[/]"
            )
            return
        table = Table(border_style="magenta")
        table.add_column("ID", justify="right")
        table.add_column("статус")
        table.add_column("прогоны", justify="right")
        table.add_column("лучшая", justify="right")
        table.add_column("кейс")
        for case in cases:
            stats = self.store.run_stats(case.id)
            text = case.situation
            if len(text) > 80:
                text = text[:77] + "…"
            table.add_row(
                str(case.id),
                case.status,
                str(stats["total"]),
                str(stats["best"] or "—"),
                escape(text),
            )
        self.console.print(table)
        if not include_archived:
            self.console.print("[dim]/cases all — показать архив[/]")

    def _case_edit(self, raw: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self.console.print("[yellow]Нужен ID кейса: \\к+ 12[/]")
            return
        case = self.store.get_case(case_id)
        if not case:
            self.console.print("[yellow]Не нашёл такой кейс.[/]")
            return

        self.console.print(
            f"[cyan]кейс #{case.id}[/] · пустой ввод — оставить как есть"
        )
        fields = [
            ("situation", "ситуация"),
            ("relationship", "отношения"),
            ("her_goal", "её цель"),
            ("tone", "тон"),
            ("restrictions", "нельзя"),
            ("opener", "кто начинает (her/me)"),
            ("tags", "теги"),
        ]
        updates: dict[str, str] = {}
        for key, label in fields:
            current = getattr(case, key) or ""
            try:
                value = self.ask(f"{label} [{current}] › ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if value:
                updates[key] = value

        if not updates:
            self.console.print("[dim]без изменений[/]")
            return
        self.store.update_case(case.id, **updates)
        self.console.print(f"[green]✓ кейс #{case.id} обновлён[/]")

    def _case_start(self, raw: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self.console.print("[yellow]Нужен ID кейса: \\к 12[/]")
            return
        if self.active_run:
            self.console.print(
                f"[yellow]Уже идёт прогон #{self.active_run.id}. "
                f"Сначала \\ч, чтобы закрыть.[/]"
            )
            return
        case = self.store.get_case(case_id)
        if not case:
            self.console.print("[yellow]Не нашёл такой кейс.[/]")
            return

        session_id = self.store.create_session(
            f"кейс #{case.id}: {case.situation[:50]}"
        )
        run = self.store.start_run(case, session_id)
        self.session_id = session_id
        self.active_run = run
        self._show_case(case)
        if case.opener == "me":
            self.console.print("[dim]Начинай ты.[/]")
            return
        hidden = self.store.add_message(
            session_id,
            "user",
            OPENING_INSTRUCTION,
            meta={"hidden": True, "kind": "opening"},
        )
        self._case_reply(kind="case_opening", cleanup_on_skip=hidden.id)

    def _show_case(self, case: Case) -> None:
        self._show_scenario(case.snapshot(), case.id)

    def _show_scenario(self, scenario: dict, case_id: int) -> None:
        lines = [f"[bold]кейс #{case_id}[/]"]
        situation = str(scenario.get("situation") or "")
        if situation:
            lines.append(escape(situation))
        fields = [
            ("отношения", scenario.get("relationship")),
            ("её цель", scenario.get("her_goal")),
            ("тон", scenario.get("tone")),
            ("нельзя", scenario.get("restrictions")),
        ]
        for label, value in fields:
            if value:
                lines.append(f"[dim]{label}:[/] {escape(str(value))}")
        lines.append(
            f"[dim]прогон #{self.active_run.id if self.active_run else '?'} · "
            f"\\ч — завершить · \\пд — подсказка[/]"
        )
        self.console.print(Panel("\n".join(lines), border_style="magenta", expand=False))

    def _case_reply(self, kind: str, cleanup_on_skip: int | None) -> None:
        assert self.active_run is not None
        if self.variant_mode == "short":
            self._case_short_reply(kind, cleanup_on_skip)
            return
        while True:
            try:
                candidates = self._generate_candidates(count=self.case_variants)
            except LLMError:
                if cleanup_on_skip:
                    self.store.delete_message(cleanup_on_skip)
                raise

            if not candidates:
                if cleanup_on_skip:
                    self.store.delete_message(cleanup_on_skip)
                self.console.print("[yellow]Модель не дала вариантов.[/]")
                return

            self._show_candidates(candidates)
            choice = self._choose(candidates)
            if choice is None:
                if cleanup_on_skip:
                    self.store.delete_message(cleanup_on_skip)
                self.console.print("[dim]пропущено[/]")
                return
            if choice == "retry":
                continue

            index, final_text, edited = choice
            stats = candidates[index][2]
            context = self._context_for_log()
            self.store.add_message(self.session_id, "assistant", final_text)
            self._log(
                {
                    "type": "choice",
                    "kind": kind,
                    "session_id": self.session_id,
                    "case_id": self.active_run.case_id,
                    "run_id": self.active_run.id,
                    "model": self.llm.cfg.model,
                    "context": context,
                    "candidates": [
                        {
                            "text": text,
                            "temperature": temp,
                            "stats": stats.as_dict() if stats else None,
                        }
                        for text, temp, stats in candidates
                    ],
                    "chosen": index,
                    "edited": edited,
                    "final": final_text,
                }
            )
            self.console.print(
                f"[green]✓ сохранено[/] [dim](вариант {index + 1})[/]"
            )
            self._print_speed(stats)
            return

    def _case_short_reply(
        self, kind: str, cleanup_on_skip: int | None
    ) -> None:
        assert self.active_run is not None
        while True:
            try:
                beats = self._generate_short_candidates(self.case_variants)
                short_stats = self.llm.last_stats
            except LLMError:
                if cleanup_on_skip:
                    self.store.delete_message(cleanup_on_skip)
                raise

            if not beats:
                if cleanup_on_skip:
                    self.store.delete_message(cleanup_on_skip)
                self.console.print("[yellow]Модель не дала заготовок.[/]")
                return

            self._show_beats(beats)
            choice = self._choose_beat(beats)
            if choice is None:
                if cleanup_on_skip:
                    self.store.delete_message(cleanup_on_skip)
                self.console.print("[dim]пропущено[/]")
                return
            if choice == "retry":
                continue

            index, beat, edited = choice
            while True:
                try:
                    reply = self._expand_beat(beat)
                except LLMError:
                    if cleanup_on_skip:
                        self.store.delete_message(cleanup_on_skip)
                    raise
                if reply:
                    break
                try:
                    action = self.ask(
                        "пустой разворот. [r=повторить, s=отменить] › "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    action = "s"
                if action in ("", "r", "retry", "повторить"):
                    continue
                if cleanup_on_skip:
                    self.store.delete_message(cleanup_on_skip)
                self.console.print("[dim]ход отменён[/]")
                return

            context = self._context_for_log()
            stats = self.llm.last_stats
            self.store.add_message(self.session_id, "assistant", reply)
            self._log(
                {
                    "type": "choice",
                    "kind": kind,
                    "session_id": self.session_id,
                    "case_id": self.active_run.case_id,
                    "run_id": self.active_run.id,
                    "model": self.llm.cfg.model,
                    "context": context,
                    "short_candidates": beats,
                    "chosen_short": index,
                    "short_edited": edited,
                    "short_stats": (
                        short_stats.as_dict() if short_stats else None
                    ),
                    "final": reply,
                    "stats": stats.as_dict() if stats else None,
                }
            )
            self.console.print("[green]✓ сохранено[/]")
            return

    def case_turn(self, text: str) -> None:
        assert self.active_run is not None
        user_message = self.store.add_message(self.session_id, "user", text)
        self._case_reply(kind="case_turn", cleanup_on_skip=user_message.id)

    def _case_retry(self) -> None:
        if not self.active_run:
            return
        last = self.store.last_message(self.session_id)
        if last and last.role == "assistant":
            self.store.delete_message(last.id)
        last = self.store.last_message(self.session_id)
        if not last or last.role != "user":
            self.console.print("[yellow]Нечего повторять.[/]")
            return
        self._case_reply(kind="case_retry", cleanup_on_skip=last.id)

    def _case_finish(self) -> None:
        if not self.active_run:
            self.console.print("[yellow]Нет активного прогона.[/]")
            return
        run = self.store.get_run(self.active_run.id)
        if not run:
            self.active_run = None
            self.console.print("[yellow]Прогон не найден.[/]")
            return

        rating = self._ask_rating()
        notes = self._ask_notes()
        self.store.finish_run(run.id, rating, notes)
        turns = self.store.count_turns(run.session_id)
        replies = len(
            self.store.messages(
                run.session_id,
                roles=("user", "assistant"),
                include_hidden=False,
            )
        )
        self._log(
            {
                "type": "run_finish",
                "run_id": run.id,
                "case_id": run.case_id,
                "session_id": run.session_id,
                "rating": rating,
                "notes": notes,
                "turns": turns,
                "replies": replies,
            }
        )
        if self.cfg.case.auto_export:
            self._export_run(run.id)
        self.active_run = None
        self.session_id = self.store.create_session(f"после кейса #{run.case_id}")
        self.console.print(
            f"[green]✓ прогон #{run.id} закрыт[/] · {replies} "
            f"{plural(replies, 'реплика', 'реплики', 'реплик')} · {rating}/5"
        )
        self.console.print(f"[dim]новая сессия {self.session_id}[/]")

    def _ask_rating(self) -> int:
        try:
            raw = self.ask("оценка 1–5 › ").strip()
        except (EOFError, KeyboardInterrupt):
            return 3
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        if raw:
            self.console.print("[dim]беру 3[/]")
        return 3

    def _ask_notes(self) -> str:
        try:
            return self.ask("заметка › ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def _export_run(self, run_id: int) -> Path | None:
        run = self.store.get_run(run_id)
        if not run:
            return None
        messages = self.store.messages(
            run.session_id, roles=("user", "assistant"), include_hidden=False
        )
        data = {
            "run_id": run.id,
            "case_id": run.case_id,
            "case": run.scenario,
            "status": run.status,
            "rating": run.rating,
            "notes": run.notes,
            "created_at": run.created_at,
            "finished_at": run.finished_at,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                }
                for message in messages
            ],
        }
        directory = self.cfg.path("data/runs")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"run_{run.id}.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.console.print(f"[green]✓ {path}[/]")
        return path

    def _case_resume(self) -> None:
        run = self.active_run or self.store.latest_active_run()
        if not run:
            self.console.print("[yellow]Незакрытых прогонов нет.[/]")
            return
        case = self.store.get_case(run.case_id)
        self.session_id = run.session_id
        self.active_run = run
        self._show_scenario(run.scenario, run.case_id)
        if not case:
            self.console.print(
                f"[yellow]Кейс #{run.case_id} удалён, но прогон #{run.id} жив.[/]"
            )
        messages = self.store.messages(
            run.session_id, roles=("user", "assistant"), include_hidden=False
        )
        for message in messages[-4:]:
            who = "ты" if message.role == "user" else self.name
            self.console.print(f"[bold]{who}:[/] {escape(message.content)}")

    def _case_hint(self) -> None:
        if not self.active_run:
            self.console.print("[yellow]Подсказка работает только в прогоне.[/]")
            return
        scenario_data = self.active_run.scenario
        messages = self.store.messages(
            self.session_id, roles=("user", "assistant"), include_hidden=False
        )
        transcript = "\n".join(
            f"{'Он' if message.role == 'user' else 'Она'}: {message.content}"
            for message in messages[-12:]
        )
        scenario_lines = []
        if scenario_data.get("situation"):
            scenario_lines.append(f"Ситуация: {scenario_data['situation']}")
        if scenario_data.get("tone"):
            scenario_lines.append(f"Тон: {scenario_data['tone']}")
        if scenario_data.get("her_goal"):
            scenario_lines.append(f"Её цель: {scenario_data['her_goal']}")
        scenario = "\n".join(scenario_lines) + ("\n" if scenario_lines else "")
        prompt = [
            {
                "role": "system",
                "content": (
                    "Ты помогаешь придумать реплику в ролевой сцене. "
                    "Верни ровно три коротких варианта того, что мог бы сказать он. "
                    "Каждый вариант — с новой строки, без нумерации, без кавычек. "
                    "Пиши по-русски, живо и по-разному."
                ),
            },
            {
                "role": "user",
                "content": f"{scenario}\nДиалог:\n{transcript}\n\nЧто мог бы сказать он?",
            },
        ]
        try:
            raw = self._complete_with_progress(
                prompt, temperature=0.9, max_tokens=300
            )
        except LLMError as exc:
            self.console.print(f"[red]{exc}[/]")
            return
        lines = self._parse_hints(raw)
        if not lines:
            self.console.print("[yellow]Не удалось разобрать подсказки.[/]")
            return
        for index, line in enumerate(lines, start=1):
            self.console.print(
                Panel(escape(line), title=f"[{index}]", border_style="dim")
            )

    @staticmethod
    def _parse_hints(raw: str) -> list[str]:
        lines: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
            line = re.sub(r"^[-•*]\s*", "", line)
            line = line.strip(" «»\"'")
            if line:
                lines.append(line)
        return lines[:3]

    def _case_status(self, raw: str, status: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self.console.print("[yellow]Нужен ID кейса. /cases[/]")
            return
        if self.store.set_case_status(case_id, status):
            self.console.print(f"[green]✓ кейс #{case_id}: {status}[/]")
        else:
            self.console.print("[yellow]Не нашёл такой кейс.[/]")

    def _case_delete(self, raw: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self.console.print("[yellow]Нужен ID кейса. /cases[/]")
            return
        if self.active_run and self.active_run.case_id == case_id:
            self.console.print(
                "[yellow]Этот кейс сейчас играется. Сначала \\ч.[/]"
            )
            return
        case = self.store.get_case(case_id)
        if not case:
            self.console.print("[yellow]Не нашёл такой кейс.[/]")
            return
        stats = self.store.run_stats(case_id)
        if self.store.archive_case(case_id):
            self.console.print(
                f"[green]✓ кейс #{case_id} архивирован[/] · "
                f"прогонов сохранено: {stats['total']}"
            )
            self.console.print(
                "[dim]история прогонов не удалена; /cases all — показать архив[/]"
            )
        else:
            self.console.print("[yellow]Кейс уже в архиве.[/]")

    def _case_purge(self, raw: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self.console.print("[yellow]Нужен ID кейса: /case purge 12[/]")
            return
        if self.active_run and self.active_run.case_id == case_id:
            self.console.print(
                "[yellow]Этот кейс сейчас играется. Сначала \\ч.[/]"
            )
            return
        case = self.store.get_case(case_id)
        if not case:
            self.console.print("[yellow]Не нашёл такой кейс.[/]")
            return
        stats = self.store.run_stats(case_id)
        self.console.print(
            f"[red]Жёсткое удаление кейса #{case_id}[/] и "
            f"{stats['total']} прогонов. Это необратимо."
        )
        try:
            answer = self.ask("удалить навсегда? [y/N] › ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("y", "yes", "д", "да"):
            self.console.print("[dim]отменено[/]")
            return
        if self.store.purge_case(case_id):
            self.console.print(f"[green]✓ кейс #{case_id} удалён навсегда[/]")
        else:
            self.console.print("[yellow]Не нашёл такой кейс.[/]")

    def _case_export(self) -> None:
        cases = self.store.cases(include_archived=True)
        path = self.cfg.path("data/cases.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Кейсы", "", f"Всего: {len(cases)}", ""]
        for case in cases:
            stats = self.store.run_stats(case.id)
            lines.append(f"## #{case.id} — {case.situation}")
            fields = [
                ("Отношения", case.relationship),
                ("Её цель", case.her_goal),
                ("Тон", case.tone),
                ("Нельзя", case.restrictions),
                ("Теги", case.tags),
            ]
            for label, value in fields:
                if value:
                    lines.append(f"- **{label}:** {value}")
            lines.append(
                f"- **Прогоны:** {stats['total']} · "
                f"**Лучшая:** {stats['best'] or '—'}"
            )
            lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.console.print(f"[green]✓ {path}[/]")

    # ── команды ───────────────────────────────────────────────────────────

    def handle_command(self, line: str) -> bool:
        name, _, rest = line.partition(" ")
        name = name.lower()
        rest = rest.strip()

        if name in ("/quit", "/exit", "/q"):
            return False
        if name in ("/help", "/h", "/?"):
            self._help()
        elif name == "/new":
            self._new_session(rest)
        elif name == "/sessions":
            self._sessions()
        elif name == "/switch":
            self._switch(rest)
        elif name == "/variants":
            self._set_variants(rest)
        elif name == "/temp":
            self._set_temp(rest)
        elif name == "/model":
            if rest:
                self.llm.cfg.model = rest
                self.console.print(f"[green]✓ модель: {rest}[/]")
            else:
                self.console.print(f"модель: {self.llm.cfg.model}")
        elif name == "/load":
            self._load_model()
        elif name == "/unload":
            self._unload_model()
        elif name in ("/loaded", "/ps"):
            self._loaded_table()
        elif name == "/persona":
            self._persona(rest)
        elif name == "/remember":
            if rest:
                fact_id = self.store.add_fact(rest)
                self.console.print(
                    f"[green]✓ факт #{fact_id}[/]" if fact_id else "[yellow]пусто[/]"
                )
            else:
                self.console.print("[yellow]Что запомнить? /remember текст[/]")
        elif name == "/facts":
            self._facts_table()
        elif name == "/forget":
            self._forget(rest)
        elif name == "/compact":
            self.summarize(force=True)
        elif name == "/case":
            self._case(rest)
        elif name == "/cases":
            self._cases_table(
                include_archived=rest.lower() in ("all", "архив", "archive")
            )
        elif name == "/case_edit":
            self._case_edit(rest)
        elif name == "/case_suggest":
            self._case_suggest(rest)
        elif name == "/finish":
            self._case_finish()
        elif name == "/resume":
            self._case_resume()
        elif name == "/hint":
            self._case_hint()
        elif name == "/abandon":
            self._abandon()
        elif name == "/again":
            self.retry()
        elif name == "/undo":
            deleted = self.store.delete_last_exchange(self.session_id)
            self.console.print(f"[green]✓ удалено сообщений: {deleted}[/]")
        elif name == "/edit":
            self._edit_last()
        elif name in ("/like", "/dislike"):
            self._rate(1 if name == "/like" else -1)
        elif name == "/export":
            self._export(rest or "md")
        elif name == "/stats":
            self._stats()
        else:
            self.console.print(f"[yellow]Не знаю команду {name}. /help[/]")
        return True

    def _new_session(self, title: str) -> None:
        if self.active_run:
            self.store.abandon_run(self.active_run.id)
            self.console.print(
                f"[dim]прогон #{self.active_run.id} помечен как abandoned[/]"
            )
            self.active_run = None
        self.session_id = self.store.create_session(title or None)
        self.console.print(f"[green]✓ новая сессия {self.session_id}[/]")

    def _abandon(self) -> None:
        if not self.active_run:
            self.console.print("[yellow]Нет активного прогона.[/]")
            return
        self.store.abandon_run(self.active_run.id)
        self.console.print(f"[green]✓ прогон #{self.active_run.id} брошен[/]")
        self.active_run = None

    def _help(self) -> None:
        table = Table(title="команды", border_style="magenta", show_lines=False)
        table.add_column("команда", style="cyan", no_wrap=True)
        table.add_column("что делает")
        rows = [
            ("/case текст", "добавить кейс; /case без аргумента — режим набора"),
            ("/case 12", "начать прогон кейса"),
            ("/case + 12", "дополнить поля кейса"),
            ("/case suggest [идея]", "модель придумает условия кейса"),
            ("/case del ID", "архивировать кейс (прогоны сохраняются)"),
            ("/case purge ID", "удалить кейс и прогоны навсегда"),
            ("/cases", "список кейсов; /cases all — с архивом"),
            ("/finish", "завершить прогон: оценка 1–5 и заметка"),
            ("/resume", "продолжить незакрытый прогон"),
            ("/hint", "3 подсказки для твоей реплики"),
            ("/abandon", "бросить прогон без оценки"),
            ("/load", "загрузить модель в память"),
            ("/unload", "выгрузить модель из памяти"),
            ("/loaded", "что сейчас загружено и до каких пор"),
            ("/variants N", "N вариантов в обычном чате (0 — поток)"),
            ("/again", "перегенерировать последний ответ"),
            ("/undo", "удалить последний обмен"),
            ("/edit", "отредактировать последний ответ"),
            ("/like", "оценить последний ответ как удачный"),
            ("/dislike", "оценить последний ответ как неудачный"),
            ("/remember текст", "добавить факт в память"),
            ("/facts", "показать факты"),
            ("/forget ID", "убрать факт"),
            ("/compact", "сжать старые сообщения в сводку"),
            ("/persona", "показать карточку; /persona edit — править"),
            ("/new [название]", "новая сессия"),
            ("/sessions", "список сессий"),
            ("/switch ID", "переключиться на сессию"),
            ("/model имя", "сменить модель"),
            ("/temp X", "температура 0–2"),
            ("/export [md|jsonl]", "выгрузить диалог"),
            ("/stats", "статистика"),
            ("/quit", "выход"),
        ]
        for command, description in rows:
            table.add_row(escape(command), escape(description))
        self.console.print(table)
        self.console.print(
            "[bold]Русская раскладка:[/] "
            "\\м/\\мфк — варианты · \\к — кейс · \\кп — придумать · "
            "\\ксы — кейсы · \\к 12 — играть · \\к+ 12 — править · \\ч — завершить · "
            "\\п — продолжить · \\пд — подсказка · \\р — помощь · \\й — выход"
        )
        self.console.print(
            "[bold]Память модели:[/] \\+ загрузить · \\- выгрузить · "
            "\\пс что загружено"
        )
        self.console.print(
            "[dim]1/2/3 — выбрать короткую заготовку, после выбора она "
            "развернётся в полную реплику; e — править заготовку, "
            "r — заново, s — пропустить.[/]"
        )

    def _sessions(self) -> None:
        table = Table(border_style="magenta")
        table.add_column("ID", justify="right")
        table.add_column("начата")
        table.add_column("сообщений", justify="right")
        table.add_column("прогон", justify="right")
        table.add_column("название")
        for row in self.store.sessions():
            table.add_row(
                str(row["id"]),
                row["started_at"][:16].replace("T", " "),
                str(row["n"]),
                str(row["run_id"] or ""),
                escape(row["title"] or ""),
            )
        self.console.print(table)

    def _switch(self, raw: str) -> None:
        if not raw.isdigit() or not self.store.get_session(int(raw)):
            self.console.print("[yellow]Нужен существующий ID. /sessions[/]")
            return
        self.session_id = int(raw)
        self.active_run = self.store.active_run_for_session(self.session_id)
        self.console.print(f"[green]✓ сессия {self.session_id}[/]")
        if self.active_run:
            self.console.print(
                f"[dim]в ней идёт прогон #{self.active_run.id}; \\ч — завершить[/]"
            )

    def _set_variants(self, raw: str) -> None:
        if not raw:
            self.console.print(
                f"вариантов в чате: {self.variants} · в кейсах: {self.case_variants}"
            )
            return
        try:
            count = int(raw)
        except ValueError:
            self.console.print("[yellow]Нужно число 0–8.[/]")
            return
        self.variants = max(0, min(8, count))
        if count > 0:
            self.case_variants = max(2, min(8, count))
        if self.variants == 0:
            self.console.print(
                f"[green]✓ обычный чат потоком, в кейсах "
                f"{self.case_variants} "
                f"{plural(self.case_variants, 'заготовка', 'заготовки', 'заготовок')}[/]"
            )
        elif self.variant_mode == "short":
            self.console.print(
                f"[green]✓ {self.case_variants} "
                f"{plural(self.case_variants, 'заготовка', 'заготовки', 'заготовок')} "
                f"на ответ[/] [dim](выбранная разворачивается в полную реплику)[/]"
            )
        else:
            self.console.print(
                f"[green]✓ {self.case_variants} "
                f"{plural(self.case_variants, 'вариант', 'варианта', 'вариантов')} "
                f"на ответ[/] [dim](CPU будет дольше)[/]"
            )

    def _set_temp(self, raw: str) -> None:
        if not raw:
            self.console.print(f"температура: {self.temperature:.2f}")
            return
        try:
            value = float(raw)
        except ValueError:
            self.console.print("[yellow]Нужно число 0–2.[/]")
            return
        self.temperature = max(0.0, min(2.0, value))
        self.console.print(f"[green]✓ температура: {self.temperature:.2f}[/]")

    def _persona(self, raw: str) -> None:
        if raw in ("edit", "редактировать"):
            edited = self.edit_text(self.persona)
            self.persona_path.write_text(edited, encoding="utf-8")
            self.persona = edited
            self.name = persona_name(edited)
            self.console.print(f"[green]✓ персонаж обновлён: {self.name}[/]")
            return
        if raw in ("reload", "перечитать"):
            self.persona = self._load_persona()
            self.name = persona_name(self.persona)
            self.console.print(f"[green]✓ перечитано: {self.name}[/]")
            return
        self.console.print(
            f"[cyan]{self.persona_path}[/] · {len(self.persona)} символов · "
            f"имя: {self.name}"
        )
        self.console.print(
            Panel(
                Markdown(self.persona),
                border_style="dim",
                title="persona.md",
            )
        )

    def _facts_table(self) -> None:
        facts = self.store.facts()
        if not facts:
            self.console.print("[dim]фактов пока нет[/]")
            return
        table = Table(border_style="magenta")
        table.add_column("ID", justify="right")
        table.add_column("факт")
        for fact in facts:
            table.add_row(str(fact.id), escape(fact.text))
        self.console.print(table)

    def _forget(self, raw: str) -> None:
        if not raw.isdigit():
            self.console.print("[yellow]Нужен ID факта. /facts[/]")
            return
        if self.store.deactivate_fact(int(raw)):
            self.console.print(f"[green]✓ факт {raw} убран[/]")
        else:
            self.console.print("[yellow]Не нашёл такой активный факт.[/]")

    def _edit_last(self) -> None:
        message = self.store.last_message(self.session_id, "assistant")
        if not message:
            self.console.print("[yellow]Нет ответа для правки.[/]")
            return
        context = self._context_before_last_assistant()
        edited = self.edit_text(message.content)
        if edited == message.content:
            self.console.print("[dim]без изменений[/]")
            return
        self.store.update_message(message.id, edited)
        self._log(
            {
                "type": "edit",
                "kind": "case_turn" if self.active_run else "chat",
                "session_id": self.session_id,
                "case_id": self.active_run.case_id if self.active_run else None,
                "run_id": self.active_run.id if self.active_run else None,
                "model": self.llm.cfg.model,
                "context": context,
                "before": message.content,
                "after": edited,
            }
        )
        self.console.print("[green]✓ ответ исправлен и записан в датасет[/]")

    def _rate(self, rating: int) -> None:
        message = self.store.last_message(self.session_id, "assistant")
        if not message:
            self.console.print("[yellow]Нет ответа для оценки.[/]")
            return
        self._log(
            {
                "type": "rating",
                "kind": "case_turn" if self.active_run else "chat",
                "session_id": self.session_id,
                "case_id": self.active_run.case_id if self.active_run else None,
                "run_id": self.active_run.id if self.active_run else None,
                "model": self.llm.cfg.model,
                "context": self._context_before_last_assistant(),
                "response": message.content,
                "rating": rating,
            }
        )
        self.console.print(
            "[green]✓ записано[/]"
            if rating > 0
            else "[green]✓ записано как неудачный[/]"
        )

    def _export(self, raw: str) -> None:
        fmt = raw.lower().strip()
        if fmt not in ("md", "jsonl"):
            self.console.print("[yellow]Формат: md или jsonl.[/]")
            return
        messages = self.store.messages(
            self.session_id, roles=("user", "assistant"), include_hidden=False
        )
        exports = self.cfg.path(self.cfg.paths.exports)
        exports.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = exports / f"session_{self.session_id}_{stamp}.{fmt}"
        if fmt == "jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for message in messages:
                    handle.write(
                        json.dumps(
                            {
                                "role": message.role,
                                "content": message.content,
                                "created_at": message.created_at,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        else:
            lines = [f"# Сессия {self.session_id}", f"_{stamp}_", ""]
            for message in messages:
                who = "Ты" if message.role == "user" else self.name
                lines.append(f"**{who}:** {message.content}")
                lines.append("")
            path.write_text("\n".join(lines), encoding="utf-8")
        self.console.print(f"[green]✓ {path}[/]")

    def _stats(self) -> None:
        summary = self.store.latest_summary(self.session_id)
        runs = self.store.runs()
        finished = [run for run in runs if run.rating is not None]
        average = (
            sum(run.rating for run in finished) / len(finished) if finished else None
        )
        table = Table(border_style="magenta")
        table.add_column("параметр")
        table.add_column("значение")
        table.add_row("модель", self.llm.cfg.model)
        table.add_row("бэкенд", self.llm.cfg.backend)
        loaded = self.llm.loaded_models()
        if loaded:
            total = sum(int(model.get("size") or 0) for model in loaded)
            table.add_row("в памяти", f"{len(loaded)} · {human_size(total)}")
        else:
            table.add_row("в памяти", "нет")
        table.add_row("keep_alive", str(self.llm.cfg.keep_alive))
        table.add_row("сессия", str(self.session_id))
        table.add_row(
            "сообщений",
            str(self.store.count_messages(self.session_id, include_hidden=False)),
        )
        table.add_row("фактов", str(len(self.store.facts())))
        table.add_row("кейсов", str(len(self.store.cases())))
        table.add_row("прогонов", f"{len(runs)} (активных {sum(1 for r in runs if r.status == 'active')})")
        table.add_row(
            "средняя оценка",
            f"{average:.1f}" if average is not None else "—",
        )
        table.add_row(
            "сводка",
            f"до сообщения {summary.upto_message_id}" if summary else "нет",
        )
        table.add_row("вариантов в чате", str(self.variants))
        table.add_row("вариантов в кейсах", str(self.case_variants))
        table.add_row("температура", f"{self.temperature:.2f}")
        table.add_row(
            "база",
            f"{self.store.path.stat().st_size / 1024:.0f} КиБ",
        )
        self.console.print(table)
