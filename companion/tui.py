from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from .cli import (
    OPENING_INSTRUCTION,
    build_key_bindings,
    expand_alias,
    parse_id,
    persona_name,
)
from .config import Config
from .llm import BaseLLM, LLMError, make_llm, strip_thinking
from .prompt import (
    build_messages as assemble_messages,
    case_suggestion_messages,
    expand_instruction,
    parse_case_card,
    parse_short_variants,
    parse_summary,
    short_variant_instruction,
    summarization_messages,
)
from .store import Run, Store


STYLE = Style.from_dict(
    {
        "status": "bg:#1f2937 fg:#9ca3af",
        "sidebar": "bg:#111827",
        "sidebar.active": "bg:#374151 bold",
        "transcript": "bg:default",
        "divider": "fg:#6b7280",
        "input": "bg:#1f2937",
        "help": "fg:#6b7280",
        "error": "fg:#f87171",
        "modal.header": "bold",
    }
)

TRANSCRIPT_LIMIT = 60
STAGE_LABELS = {
    "connecting": "подключение",
    "load": "загрузка модели",
    "prompt": "обработка контекста",
    "generation": "генерация",
    "done": "готово",
}


class TuiApp:
    """Полноэкранный интерфейс: сессии, история, потоковая генерация."""

    def __init__(self, cfg: Config, session_id: int | None = None):
        self.cfg = cfg
        self.store = Store(cfg.path(cfg.paths.db))
        self.llm: BaseLLM = make_llm(cfg.llm)
        self.persona = cfg.path(cfg.paths.persona).read_text(encoding="utf-8")
        self.name = persona_name(self.persona)
        self.mode = cfg.chat.mode.lower() or "roleplay"
        self.adult = cfg.content.adult
        self.temperature = cfg.llm.temperature
        self.variants = max(0, cfg.chat.variants)
        self.case_variants = max(2, cfg.case.variants)
        self.variant_mode = cfg.chat.variant_mode.lower()
        if self.variant_mode not in ("short", "full"):
            self.variant_mode = "short"
        self.seed_base = 0

        self.session_id = self._pick_session(session_id)
        # Если открыли закрытую сессию явно (--session), покажем её в сайдбаре.
        self.show_closed_sessions = self.store.is_session_closed(
            self.session_id
        )
        self.active_run: Run | None = self.store.active_run_for_session(
            self.session_id
        )
        self.ask_kind: str | None = None
        self.ask_prompt = ""
        self.ask_data: dict[str, Any] = {}
        self.sessions: list[dict[str, Any]] = []
        self.sel_index = 0
        self.busy = False
        self.at_bottom = True
        self.history_limit = TRANSCRIPT_LIMIT
        self.edit_target_id: int | None = None

        self.stage = "ready"
        self.tps = 0.0
        self.tokens = 0
        self.streaming = ""
        self.streaming_active = False
        self.error = ""
        self.hint = ""

        self.modal_open = False
        self.modal_title = ""
        self.modal_items: list[tuple[Any, ...]] = []
        self.modal_filter = ""
        self.modal_sel = 0
        self.modal_hint = ""
        self.modal_on_escape: Any = None
        self.modal_control = FormattedTextControl(
            self._modal_text, focusable=True
        )

        self.transcript_buffer = Buffer(read_only=True, multiline=True)
        self.status_control = FormattedTextControl(self._status_text)
        self.session_control = FormattedTextControl(
            self._sessions_text, focusable=True
        )
        self.input_area = TextArea(
            prompt="ты › ",
            multiline=True,
            height=Dimension(min=1, max=3),
            style="class:input",
        )
        self.transcript_window = Window(
            content=BufferControl(buffer=self.transcript_buffer, focusable=True),
            wrap_lines=True,
            style="class:transcript",
            get_vertical_scroll=self._transcript_scroll,
        )

        self.app = Application(
            layout=Layout(self._build_layout()),
            key_bindings=self._bindings(),
            style=STYLE,
            full_screen=True,
            mouse_support=True,
        )

        self._refresh_sessions()
        self._render_transcript()

    # ── запуск ────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self.app.run(pre_run=self._focus_input)
        finally:
            self.close()

    def close(self) -> None:
        self.store.close()

    def _focus_input(self) -> None:
        self.app.layout.focus(self.input_area)

    def _pick_session(self, session_id: int | None) -> int:
        if session_id is not None:
            if not self.store.get_session(session_id):
                raise SystemExit(f"Сессии {session_id} нет.")
            return session_id
        latest = self.store.latest_open_session()
        if latest is not None:
            return latest
        return self.store.create_session("первый разговор")

    def _switch_session(self, session_id: int) -> None:
        if self._busy_guard():
            return
        if not self.store.get_session(session_id):
            return
        self.session_id = session_id
        self.active_run = self.store.active_run_for_session(session_id)
        self.at_bottom = True
        self.streaming_active = False
        self.streaming = ""
        self._refresh_sessions()
        self._render_transcript()
        if self.store.is_session_closed(session_id):
            self._flash(f"сессия {session_id} закрыта; /reopen {session_id}")
        self.invalidate()

    # ── layout ────────────────────────────────────────────────────────────

    def _build_layout(self) -> Any:
        sidebar = Window(
            content=self.session_control,
            width=Dimension(min=26, max=34),
            style="class:sidebar",
            wrap_lines=False,
        )
        status = Window(
            content=self.status_control,
            height=Dimension.exact(1),
            style="class:status",
        )
        right = HSplit(
            [
                self.transcript_window,
                Window(height=Dimension.exact(1), char="─", style="class:divider"),
                self.input_area,
                status,
            ]
        )
        root = VSplit([sidebar, right])
        modal = ConditionalContainer(
            content=Window(
                content=self.modal_control,
                style="class:sidebar",
                dont_extend_width=True,
                wrap_lines=True,
            ),
            filter=Condition(lambda: self.modal_open),
        )
        return FloatContainer(
            content=root,
            floats=[
                Float(
                    content=modal,
                    left=6,
                    right=6,
                    top=3,
                    bottom=3,
                )
            ],
        )

    # ── текст ─────────────────────────────────────────────────────────────

    def _status_text(self) -> Any:
        if self.busy:
            if self.stage in ("connecting", "load", "prompt"):
                state = STAGE_LABELS.get(self.stage, self.stage)
            elif self.stage == "error":
                state = "ошибка"
            else:
                state = f"{self.tps:.1f} ток/с"
        else:
            state = "готова"
        parts: list[Any] = [
            ("class:status", f" {self.llm.cfg.model} · {self.mode} · "),
            ("class:status", f"сессия {self.session_id} · "),
        ]
        if self.active_run:
            parts.append(
                ("class:status", f"кейс #{self.active_run.case_id} · ")
            )
        if self.store.is_session_closed(self.session_id):
            parts.append(("class:error", "закрыта · "))
        style = "class:error" if self.stage == "error" else "class:status"
        parts.append((style, state))
        if self.stage == "error" and self.error:
            parts.append(("class:error", f" · {self.error[:60]}"))
        elif self.hint:
            parts.append(("class:help", f" · {self.hint}"))
        return parts

    def _sessions_text(self) -> Any:
        if not self.sessions:
            return [("class:help", " сессий нет")]
        lines: list[Any] = [("class:help", " сессии (Tab, ↑↓, Enter)\n")]
        for index, session in enumerate(self.sessions):
            mark = ">" if index == self.sel_index else " "
            closed = session.get("closed")
            style = "class:sidebar.active" if index == self.sel_index else ""
            if closed and index != self.sel_index:
                style = "class:help"
            title = (session.get("title") or "разговор")[:18]
            prefix = "✓ " if closed else ""
            preview = (session.get("preview") or "—")[:16]
            lines.append(
                (style, f"{mark} {session['id']:>3} {prefix}{title}\n")
            )
            lines.append(("class:help", f"       {preview}\n"))
        return lines

    def _refresh_sessions(self) -> None:
        rows = self.store.sessions(
            include_closed=self.show_closed_sessions
        )
        self.sessions = [
            {
                "id": row["id"],
                "title": row["title"] or "разговор",
                "n": row["n"],
                "run_id": row["run_id"],
                "closed": bool(row["ended_at"]),
                "preview": self._session_preview(row["id"]),
            }
            for row in rows
        ]
        if self.sel_index >= len(self.sessions):
            self.sel_index = max(0, len(self.sessions) - 1)

    def _session_preview(self, session_id: int) -> str:
        last = self.store.messages(
            session_id,
            roles=("user", "assistant"),
            include_hidden=False,
            limit=1,
        )
        if not last:
            return ""
        text = " ".join(last[0].content.split())
        return text[:40]

    # ── transcript ────────────────────────────────────────────────────────

    def _history_messages(self) -> list[tuple[str, str]]:
        messages = self.store.messages(
            self.session_id,
            roles=("user", "assistant"),
            include_hidden=False,
        )
        lines: list[tuple[str, str]] = []
        for message in messages[-self.history_limit:]:
            role = "user" if message.role == "user" else "assistant"
            who = "ты" if message.role == "user" else self.name
            lines.append((role, f"{who} › {message.content}"))
        return lines

    def _render_transcript(self) -> None:
        lines = self._history_messages()
        text = "\n".join(content for _, content in lines)
        if self.streaming_active:
            if text:
                text += "\n"
            text += f"{self.name} › {self.streaming}"
        self.transcript_buffer.set_document(
            Document(text, cursor_position=len(text)),
            bypass_readonly=True,
        )

    def _transcript_scroll(self, window: Any) -> int:
        info = window.render_info
        if info is None:
            return 0
        if self.at_bottom:
            return info.max_scroll
        return info.vertical_scroll

    def invalidate(self) -> None:
        try:
            self.app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    # ── сообщения для модели ──────────────────────────────────────────────

    def _messages(
        self,
        session_id: int | None = None,
        scenario: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        session_id = self.session_id if session_id is None else session_id
        summary = self.store.latest_summary(session_id)
        after = summary.upto_message_id if summary else 0
        history = self.store.messages(
            session_id,
            after_id=after,
            limit=self.cfg.chat.history_messages,
            roles=("user", "assistant"),
            include_hidden=True,
        )
        facts = [fact.text for fact in self.store.facts()]
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

    # ── ввод и генерация ──────────────────────────────────────────────────

    def _submit(self, text: str) -> None:
        self.hint = ""
        text = text.strip()
        if self.ask_kind is not None:
            self._answer_ask(text)
            return
        if not text:
            return
        if text.startswith("/") or text.startswith("\\"):
            self.edit_target_id = None
            self._run_command(text)
            return
        if self.edit_target_id is not None:
            self.store.update_message(self.edit_target_id, text)
            self.edit_target_id = None
            self._render_transcript()
            self.invalidate()
            self._flash("ответ исправлен")
            return
        if self._closed_guard():
            return
        if self.busy:
            self.hint = "подожди, идёт генерация"
            self.invalidate()
            return
        session_id = self.session_id
        user_message = self.store.add_message(session_id, "user", text)
        self._refresh_sessions()
        self._render_transcript()
        if self.active_run:
            self.app.create_background_task(
                self._case_reply("case_turn", user_message.id)
            )
        elif self.variants > 0:
            self.app.create_background_task(
                self._generate_variants(session_id)
            )
        else:
            self.app.create_background_task(self._generate(session_id))

    def _flash(self, message: str) -> None:
        self.hint = message
        self.invalidate()

    def _busy_guard(self) -> bool:
        if self.busy:
            self._flash("подожди, идёт генерация")
            return True
        return False

    # ── модальный список ──────────────────────────────────────────────────

    def _open_modal(
        self,
        title: str,
        items: list[tuple[Any, ...]],
        hint: str = "↑/↓ — выбор · Enter — выбрать · Esc — назад",
        on_escape: Any = None,
    ) -> None:
        self.modal_title = title
        self.modal_items = items
        self.modal_hint = hint
        self.modal_on_escape = on_escape
        self.modal_filter = ""
        self.modal_sel = 0
        self.modal_open = True
        self._jump_to_first_selectable()
        self.app.layout.focus(self.modal_control)
        self.invalidate()

    def _close_modal(self) -> None:
        callback = self.modal_on_escape
        self.modal_on_escape = None
        self.modal_open = False
        self.modal_items = []
        self.modal_filter = ""
        self._focus_input()
        if callable(callback):
            callback()
        self.invalidate()

    def _filtered_items(self) -> list[tuple[Any, ...]]:
        needle = self.modal_filter.lower()
        if not needle:
            return self.modal_items
        return [
            item
            for item in self.modal_items
            if needle in item[0].lower()
        ]

    def _is_selectable(self, item: tuple[Any, ...]) -> bool:
        kind = item[2] if len(item) > 2 else "item"
        return kind not in ("header", "info")

    def _jump_to_first_selectable(self) -> None:
        items = self._filtered_items()
        for index, item in enumerate(items):
            if self._is_selectable(item):
                self.modal_sel = index
                return
        self.modal_sel = 0

    def _modal_viewport(self) -> int:
        try:
            rows = self.app.output.get_size().rows
        except Exception:  # noqa: BLE001
            rows = 24
        # Поля Float (top/bottom), заголовок и строка подсказки.
        return max(6, min(rows - 8, 32))

    def _modal_text(self) -> Any:
        lines: list[Any] = [("class:sidebar.active", f" {self.modal_title}\n"), ("", "\n")]
        items = self._filtered_items()
        if not items:
            lines.append(("class:help", "  (ничего не найдено)\n"))
            lines.append(("class:help", "\n Esc — назад"))
            return lines
        viewport = self._modal_viewport()
        sel = min(self.modal_sel, len(items) - 1)
        start = max(0, min(sel - viewport // 2, len(items) - viewport))
        end = min(len(items), start + viewport)
        if start > 0:
            lines.append(("class:help", "  … выше …\n"))
        for index in range(start, end):
            item = items[index]
            label = item[0]
            kind = item[2] if len(item) > 2 else "item"
            if kind == "header":
                if index > start:
                    lines.append(("", "\n"))
                lines.append(("class:modal.header", f"  {label}\n"))
                continue
            if kind == "info":
                lines.append(("class:help", f"   {label}\n"))
                continue
            mark = ">" if index == sel else " "
            style = "class:sidebar.active" if index == sel else ""
            lines.append((style, f" {mark} {label}\n"))
        if end < len(items):
            lines.append(("class:help", "  … ниже …\n"))
        lines.append(("class:help", f"\n {self.modal_hint}"))
        return lines

    def _modal_run(self) -> None:
        items = self._filtered_items()
        if not items:
            return
        index = min(self.modal_sel, len(items) - 1)
        item = items[index]
        if not self._is_selectable(item):
            return
        action = item[1]
        self.modal_on_escape = None
        self.modal_open = False
        self._focus_input()
        if callable(action):
            action()
        self.invalidate()

    def _modal_filter_char(self, char: str) -> None:
        if char and char.isprintable():
            self.modal_filter += char
            self._jump_to_first_selectable()
            self.invalidate()

    def _modal_backspace(self) -> None:
        self.modal_filter = self.modal_filter[:-1]
        self._jump_to_first_selectable()
        self.invalidate()

    def _modal_move(self, delta: int) -> None:
        items = self._filtered_items()
        if not items:
            return
        selectable = [
            index
            for index, item in enumerate(items)
            if self._is_selectable(item)
        ]
        if not selectable:
            return
        if self.modal_sel in selectable:
            position = selectable.index(self.modal_sel)
        else:
            position = -1 if delta > 0 else 0
        self.modal_sel = selectable[(position + delta) % len(selectable)]
        self.invalidate()

    # ── палитра команд ────────────────────────────────────────────────────

    def _command_items(self) -> list[tuple[str, Any]]:
        return [
            ("новая сессия", self._new_session),
            ("сессии (сайдбар)", lambda: self.app.layout.focus(self.session_control)),
            ("сессии: все", lambda: self._show_sessions(True)),
            ("сессии: открытые", lambda: self._show_sessions(False)),
            ("закрыть сессию", lambda: self._close_session("")),
            ("открыть сессию…", lambda: self._insert_input("/reopen ")),
            ("выбор модели…", self._model_menu),
            (f"режим: {self.mode} (переключить)", self._toggle_mode),
            ("загрузить модель", self._load_model),
            ("выгрузить модель", self._unload_model),
            ("статистика", self._stats_hint),
            ("экспорт markdown", lambda: self._export("md")),
            ("экспорт jsonl", lambda: self._export("jsonl")),
            ("кейсы (список)", self._cases_modal),
            ("предложить кейс…", lambda: self._insert_input("/case suggest ")),
            ("завершить кейс…", self._case_finish),
            ("продолжить кейс", self._case_resume),
            ("подсказка в кейсе", self._case_hint),
            ("помощь", self._open_help),
            ("выход", self.app.exit),
        ]

    def _open_palette(self) -> None:
        self._open_modal("команды", self._command_items())

    def _insert_input(self, text: str) -> None:
        self.input_area.text = text
        self.input_area.buffer.cursor_position = len(text)
        self.app.layout.focus(self.input_area)
        self.invalidate()

    def _help_items(self) -> list[tuple[Any, ...]]:
        def insert(text: str) -> Any:
            return lambda: self._insert_input(text)

        return [
            ("----- клавиши -----", None, "header"),
            ("Enter — отправить сообщение", None),
            ("Alt+Enter — перенос строки во вводе", None),
            ("Tab — фокус: ввод → история → сессии", None),
            ("↑/↓ + Enter — выбрать сессию в сайдбаре", None),
            ("Ctrl+N — новая сессия", None),
            ("Ctrl+P — палитра команд", None),
            ("PageUp/PageDown, Home/End — прокрутка истории", None),
            ("Ctrl+←/→, Ctrl+W — по словам", None),
            ("Ctrl+C / Ctrl+Q — выход", None),
            ("----- чат -----", None, "header"),
            ("/new [название] — новая сессия", insert("/new ")),
            ("/sessions — открытые сессии", insert("/sessions")),
            ("/sessions all — вместе с закрытыми", insert("/sessions all")),
            ("/switch ID — переключиться на сессию", insert("/switch ")),
            ("/close [ID] — закрыть сессию", insert("/close")),
            ("/reopen ID — открыть закрытую сессию", insert("/reopen ")),
            ("/again — перегенерировать последний ответ", insert("/again")),
            ("/undo — удалить последний обмен", insert("/undo")),
            ("/edit — править последний ответ", insert("/edit")),
            ("/like — лайк последнему ответу", insert("/like")),
            ("/dislike — дизлайк последнему ответу", insert("/dislike")),
            ("/variants N — N вариантов (0 — обычный поток)", insert("/variants ")),
            ("/temp X — температура 0–2", insert("/temp ")),
            ("/model [имя] — сменить или выбрать модель", insert("/model ")),
            ("/history N — сколько сообщений показывать", insert("/history ")),
            ("/export [md|jsonl] — выгрузить диалог", insert("/export ")),
            ("/stats — статистика", insert("/stats")),
            ("/quit — выход", insert("/quit")),
            ("----- память и персонаж -----", None, "header"),
            ("/remember текст — добавить факт", insert("/remember ")),
            ("/facts — показать факты", insert("/facts")),
            ("/forget ID — убрать факт", insert("/forget ")),
            ("/compact — сжать историю в сводку", insert("/compact")),
            ("/persona — карточка персонажа", insert("/persona")),
            ("/persona edit — править карточку", insert("/persona edit")),
            ("/persona reload — перечитать карточку", insert("/persona reload")),
            ("----- модель -----", None, "header"),
            ("/load — загрузить модель в память", insert("/load")),
            ("/unload — выгрузить модель", insert("/unload")),
            ("/loaded (/ps) — что загружено", insert("/loaded")),
            ("----- кейсы -----", None, "header"),
            ("/case текст — добавить и сразу играть", insert("/case ")),
            ("/case add текст — добавить без запуска", insert("/case add ")),
            ("/case show ID — полная карточка кейса", insert("/case show ")),
            ("/case ID — начать прогон", insert("/case ")),
            ("/case suggest [идея] — модель предложит кейс", insert("/case suggest ")),
            ("/case + ID — дополнить поля кейса", insert("/case + ")),
            ("/case del ID — архивировать кейс", insert("/case del ")),
            ("/case purge ID — удалить кейс навсегда", insert("/case purge ")),
            ("/case export — выгрузить data/cases.md", insert("/case export")),
            ("/cases — список кейсов", insert("/cases")),
            ("/cases all — список с архивом", insert("/cases all")),
            ("/finish — завершить прогон: оценка и заметка", insert("/finish")),
            ("/resume — продолжить незакрытый прогон", insert("/resume")),
            ("/hint — 3 подсказки для твоей реплики", insert("/hint")),
            ("/abandon — бросить прогон без оценки", insert("/abandon")),
        ]

    def _open_help(self) -> None:
        self._open_modal(
            "помощь",
            self._help_items(),
            hint="↑/↓ — выбор · Enter — вставить в ввод · Esc — назад",
        )

    def _model_menu(self) -> None:
        self.app.create_background_task(self._model_menu_async())

    async def _model_menu_async(self) -> None:
        try:
            models = await asyncio.to_thread(self.llm.list_models)
        except LLMError as exc:
            self._flash(str(exc)[:80])
            return
        if not models:
            models = [self.llm.cfg.model]
        items = [
            (name, lambda name=name: self._set_model(name)) for name in models
        ]
        self._open_modal("модели", items)

    def _set_model(self, name: str) -> None:
        self.llm.cfg.model = name
        self._flash(f"модель: {name}")

    def _toggle_mode(self) -> None:
        self.mode = "messaging" if self.mode == "roleplay" else "roleplay"
        self.cfg.chat.mode = self.mode
        self._flash(f"режим: {self.mode}")

    def _new_session(self, title: str | None = None) -> None:
        if self._busy_guard():
            return
        if self.active_run:
            self.store.abandon_run(self.active_run.id)
            self.active_run = None
        self.session_id = self.store.create_session(title)
        self._refresh_sessions()
        self._render_transcript()

    def _closed_guard(self) -> bool:
        if self.store.is_session_closed(self.session_id):
            self._flash(
                f"сессия {self.session_id} закрыта; "
                f"/reopen {self.session_id}"
            )
            return True
        return False

    def _show_sessions(self, include_closed: bool) -> None:
        self.show_closed_sessions = include_closed
        self._refresh_sessions()
        self.app.layout.focus(self.session_control)
        self.invalidate()

    def _close_session(self, raw: str) -> None:
        if self._busy_guard():
            return
        session_id = parse_id(raw) if raw else self.session_id
        if session_id is None:
            self._flash("/close [ID]")
            return
        if not self.store.get_session(session_id):
            self._flash("не нашёл такую сессию")
            return
        if self.store.is_session_closed(session_id):
            self._flash("сессия уже закрыта")
            return
        if self.active_run and self.active_run.session_id == session_id:
            self._flash(
                f"в сессии идёт прогон #{self.active_run.id}; "
                "/finish или /abandon"
            )
            return

        open_count = len(self.store.sessions(include_closed=False))
        self.store.close_session(session_id)

        if session_id != self.session_id:
            self._refresh_sessions()
            self._render_transcript()
            self.invalidate()
            self._flash(f"сессия {session_id} закрыта")
            return

        if open_count <= 1:
            self.session_id = self.store.create_session("новая сессия")
            self.active_run = None
            self._flash("последняя открытая закрыта; создана новая")
        else:
            next_id = self.store.latest_open_session()
            if next_id is None:
                next_id = self.store.create_session("новая сессия")
            self.session_id = next_id
            self.active_run = self.store.active_run_for_session(next_id)
            self._flash(f"сессия закрыта; открыта {next_id}")
        self._refresh_sessions()
        self._render_transcript()
        self.invalidate()

    def _reopen_session(self, raw: str) -> None:
        if self._busy_guard():
            return
        session_id = parse_id(raw) if raw else self.session_id
        if session_id is None:
            self._flash("/reopen ID")
            return
        if not self.store.get_session(session_id):
            self._flash("не нашёл такую сессию")
            return
        if self.store.reopen_session(session_id):
            self.session_id = session_id
            self.active_run = self.store.active_run_for_session(session_id)
            self._refresh_sessions()
            self._render_transcript()
            self._flash(f"сессия {session_id} открыта")
        else:
            self._flash("сессия уже открыта")

    def _load_model(self) -> None:
        if self._busy_guard():
            return
        self.app.create_background_task(self._load_model_async())

    async def _load_model_async(self) -> None:
        try:
            await asyncio.to_thread(self.llm.load_model)
            self._flash("модель загружена")
        except LLMError as exc:
            self._flash(str(exc)[:80])

    def _unload_model(self) -> None:
        if self._busy_guard():
            return
        self.app.create_background_task(self._unload_model_async())

    async def _unload_model_async(self) -> None:
        try:
            await asyncio.to_thread(self.llm.unload_model)
            self._flash("модель выгружена")
        except LLMError as exc:
            self._flash(str(exc)[:80])

    def _loaded_hint(self) -> None:
        self.app.create_background_task(self._loaded_hint_async())

    async def _loaded_hint_async(self) -> None:
        try:
            loaded = await asyncio.to_thread(self.llm.loaded_models)
        except LLMError as exc:
            self._flash(str(exc)[:80])
            return
        if loaded:
            total = sum(int(model.get("size") or 0) for model in loaded)
            self._flash(f"в памяти: {len(loaded)} · {total / 1024**3:.1f} ГБ")
        else:
            self._flash("модель не загружена")

    def _stats_hint(self) -> None:
        runs = self.store.runs()
        self._flash(
            f"сессий {len(self.store.sessions())} · "
            f"фактов {len(self.store.facts())} · "
            f"кейсов {len(self.store.cases())} · "
            f"прогонов {len(runs)}"
        )

    def _export(self, fmt: str) -> None:
        messages = self.store.messages(
            self.session_id, roles=("user", "assistant"), include_hidden=False
        )
        directory = self.cfg.path(self.cfg.paths.exports)
        directory.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = directory / f"session_{self.session_id}_{stamp}.{fmt}"
        if fmt == "jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for message in messages:
                    handle.write(
                        json.dumps(
                            {"role": message.role, "content": message.content},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        else:
            lines = [f"# Сессия {self.session_id}", ""]
            for message in messages:
                who = "ты" if message.role == "user" else self.name
                lines.append(f"**{who}:** {message.content}")
                lines.append("")
            path.write_text("\n".join(lines), encoding="utf-8")
        self._flash(f"экспорт: {path}")

    async def _generate(self, session_id: int | None = None) -> None:
        session_id = self.session_id if session_id is None else session_id
        await self._stream_and_save(self._messages(session_id), session_id)

    def _with_instruction(
        self, messages: list[dict[str, str]], instruction: str
    ) -> list[dict[str, str]]:
        result = [dict(message) for message in messages]
        result[0]["content"] = (
            result[0]["content"].rstrip() + "\n\n" + instruction
        )
        return result

    async def _generate_variants(self, session_id: int | None = None) -> None:
        session_id = self.session_id if session_id is None else session_id
        self.busy = True
        self.streaming = ""
        self.streaming_active = False
        self.stage = "generation"
        self.tps = 0.0
        self.tokens = 0
        self.invalidate()

        messages = self._with_instruction(
            self._messages(session_id), short_variant_instruction()
        )
        loop = asyncio.get_running_loop()
        self.llm.on_progress = lambda event: loop.call_soon_threadsafe(
            self._on_progress, event
        )
        temps = self.cfg.chat.variant_temperatures or [self.temperature]
        beats: list[str] = []
        try:
            for index in range(max(1, self.variants)):
                raw = await asyncio.to_thread(
                    self.llm.complete,
                    messages,
                    temperature=temps[index % len(temps)],
                    max_tokens=self.cfg.chat.short_max_tokens,
                )
                parsed = parse_short_variants(raw, 1)
                if parsed:
                    beats.append(parsed[0])
        except Exception as exc:  # noqa: BLE001
            self.busy = False
            self.stage = "error"
            self.error = str(exc)
            self.invalidate()
            return
        finally:
            self.llm.on_progress = None

        self.busy = False
        self.stage = "ready"
        self.invalidate()
        if not beats:
            self._flash("модель не дала заготовок")
            return
        items = [
            (
                f"{index + 1}. {beat}",
                lambda beat=beat: self.app.create_background_task(
                    self._expand(beat, session_id)
                ),
            )
            for index, beat in enumerate(beats)
        ]
        self._open_modal("выберите заготовку", items)

    async def _expand(
        self, beat: str, session_id: int | None = None
    ) -> None:
        session_id = self.session_id if session_id is None else session_id
        messages = self._with_instruction(
            self._messages(session_id),
            expand_instruction(beat, mode=self.mode),
        )
        await self._stream_and_save(messages, session_id)

    async def _stream_and_save(
        self,
        messages: list[dict[str, str]],
        session_id: int | None = None,
    ) -> str:
        session_id = self.session_id if session_id is None else session_id
        self.busy = True
        self.streaming = ""
        self.streaming_active = True
        self.stage = "connecting"
        self.tps = 0.0
        self.tokens = 0
        self.invalidate()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(self._on_progress, event)

        self.llm.on_progress = on_progress

        def producer() -> None:
            try:
                for chunk in self.llm.stream(
                    messages, temperature=self.temperature
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        producer_task = asyncio.create_task(asyncio.to_thread(producer))

        error: str | None = None
        while True:
            item = await queue.get()
            if item[0] == "token":
                self.streaming += item[1]
                if self.session_id == session_id:
                    self._render_transcript()
                self.invalidate()
            elif item[0] == "error":
                error = str(item[1])
                break
            else:
                break

        await producer_task
        self.llm.on_progress = None

        if error is not None:
            self.streaming_active = False
            self.streaming = ""
            self.error = error
            self.stage = "error"
            self.busy = False
            if self.session_id == session_id:
                self._render_transcript()
            self.invalidate()
            return ""

        reply = strip_thinking(self.streaming).strip()
        self.streaming_active = False
        self.streaming = ""
        if reply:
            self.store.add_message(session_id, "assistant", reply)
            self._refresh_sessions()
        self.busy = False
        self.stage = "ready"
        if self.session_id == session_id:
            self._render_transcript()
        self.invalidate()
        return reply

    def _on_progress(self, event: dict[str, Any]) -> None:
        if event.get("stage"):
            self.stage = str(event["stage"])
        if event.get("tokens") is not None:
            self.tokens = int(event["tokens"])
        if event.get("tokens_per_second") is not None:
            self.tps = float(event["tokens_per_second"])
        self.invalidate()

    # ── команды (базовые) ────────────────────────────────────────────────

    def _again(self) -> None:
        if self._busy_guard():
            return
        if self._closed_guard():
            return
        last = self.store.last_message(self.session_id)
        if last and last.role == "assistant":
            self.store.delete_message(last.id)
        last = self.store.last_message(self.session_id)
        if not last or last.role != "user":
            self._flash("нечего повторять")
            return
        self._render_transcript()
        self.invalidate()
        if self.active_run:
            kind = (
                "case_opening"
                if last.content == OPENING_INSTRUCTION
                else "case_retry"
            )
            self.app.create_background_task(
                self._case_reply(kind, last.id)
            )
            return
        session_id = self.session_id
        if self.variants > 0:
            self.app.create_background_task(
                self._generate_variants(session_id)
            )
        else:
            self.app.create_background_task(self._generate(session_id))

    def _edit_persona(self, path: Any) -> None:
        if self._busy_guard():
            return
        self.app.create_background_task(self._edit_persona_async(path))

    async def _edit_persona_async(self, path: Any) -> None:
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        try:
            await self.app.run_system_command(
                f"{editor} {shlex.quote(str(path))}",
                wait_for_enter=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._flash(f"не удалось открыть редактор: {exc}"[:100])
            return
        try:
            self.persona = path.read_text(encoding="utf-8")
            self.name = persona_name(self.persona)
            self._render_transcript()
            self._flash(f"персонаж обновлён: {self.name}")
        except OSError as exc:
            self._flash(str(exc)[:80])

    def _case_command(self, arg: str) -> None:
        arg = arg.strip()
        if not arg:
            self._flash(
                "/case текст — добавить · /case ID — играть · "
                "/case suggest — предложить"
            )
            return
        if arg.isdigit():
            self._case_start(arg)
            return
        action, _, rest = arg.partition(" ")
        action_lower = action.lower()
        if action_lower in ("add", "добавить"):
            self._case_add_only(rest.strip())
            return
        if action_lower in ("show", "показать", "open", "открыть"):
            case_id = parse_id(rest)
            if case_id is None:
                self._flash("/case show ID")
                return
            self._show_case_detail(case_id, include_archived=True)
            return
        if action_lower in ("suggest", "предложи", "придумай", "идея", "gen"):
            self._case_suggest(rest.strip())
            return
        if action_lower in ("play", "играть", "сыграть"):
            self._case_start(rest.strip())
            return
        if action_lower in ("+", "edit", "редактировать", "правка"):
            self._case_edit(rest.strip())
            return
        if action_lower in ("finish", "завершить", "конец"):
            self._case_finish()
            return
        if action_lower in ("resume", "продолжить", "дальше"):
            self._case_resume()
            return
        if action_lower in ("hint", "подсказка"):
            self._case_hint()
            return
        if action_lower in ("del", "delete", "удалить"):
            self._case_delete(rest.strip())
            return
        if action_lower in ("purge", "hard-delete", "удалитьнавсегда"):
            self._case_purge(rest.strip())
            return
        if action_lower in ("done", "готово", "закрыт"):
            self._case_status(rest.strip(), "done")
            return
        if action_lower in ("export", "экспорт"):
            self._case_export()
            return
        if action_lower in ("list", "список"):
            self._cases_modal()
            return
        try:
            case = self.store.add_case(arg)
        except ValueError as exc:
            self._flash(str(exc)[:100])
            return
        self._start_new_case(case)

    def _case_suggest(self, seed: str) -> None:
        if self._busy_guard():
            return
        self.app.create_background_task(
            self._case_suggest_async(seed or None)
        )

    async def _case_suggest_async(
        self, seed: str | None, exclude: list[str] | None = None
    ) -> None:
        self.busy = True
        self.stage = "generation"
        self.tps = 0.0
        self.tokens = 0
        self.invalidate()

        existing = [
            case.situation for case in self.store.cases(include_archived=True)
        ]
        if exclude:
            existing.extend(exclude)
        messages = case_suggestion_messages(
            self.persona, seed, existing, adult=self.adult
        )
        loop = asyncio.get_running_loop()
        self.llm.on_progress = lambda event: loop.call_soon_threadsafe(
            self._on_progress, event
        )
        try:
            raw = await asyncio.to_thread(
                self.llm.complete,
                messages,
                temperature=0.9,
                max_tokens=500,
            )
        except Exception as exc:  # noqa: BLE001
            self.busy = False
            self.stage = "error"
            self.error = str(exc)
            self.invalidate()
            return
        finally:
            self.llm.on_progress = None

        card = parse_case_card(raw)
        self.busy = False
        self.stage = "ready"
        self.invalidate()
        if not card:
            self._flash("модель вернула не JSON — попробуй ещё раз")
            return
        self._show_case_card(card, seed)

    def _show_case_card(
        self, card: dict[str, str], seed: str | None
    ) -> None:
        def save() -> None:
            fields = {key: value for key, value in card.items() if key != "situation"}
            try:
                case = self.store.add_case(card["situation"], **fields)
            except ValueError as exc:
                self._flash(str(exc)[:100])
                return
            self._start_new_case(case)

        def retry() -> None:
            self.app.create_background_task(
                self._case_suggest_async(seed, exclude=[card["situation"]])
            )

        items: list[tuple[Any, ...]] = [
            ("ситуация: " + card["situation"], None, "info"),
        ]
        for label, key in (
            ("отношения", "relationship"),
            ("её цель", "her_goal"),
            ("тон", "tone"),
            ("нельзя", "restrictions"),
        ):
            value = card.get(key)
            if value:
                items.append((f"{label}: {value}", None, "info"))
        items.append(("----- действия -----", None, "header"))
        items.append(("сохранить кейс", save))
        items.append(("заново", retry))
        items.append(("отмена", None))
        self._open_modal(
            "предложение кейса",
            items,
            hint="↑/↓ — выбор · Enter — действие · Esc — назад",
        )

    # ── кейсы: прогоны и завершение ───────────────────────────────────────

    def _case_scenario(self) -> dict[str, Any] | None:
        if not self.active_run:
            return None
        return self.active_run.scenario

    def _case_start(
        self, raw: str, reuse_empty: bool = False
    ) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self._flash("/case ID — начать прогон")
            return
        if self._busy_guard():
            return
        pending = self.active_run or self.store.latest_active_run()
        if pending:
            self.active_run = pending
            self._flash(
                f"уже идёт прогон #{pending.id}; /resume или /finish"
            )
            return
        case = self.store.get_case(case_id)
        if not case:
            self._flash("не нашёл такой кейс")
            return
        title = f"кейс #{case.id}: {case.situation[:50]}"
        can_reuse = (
            reuse_empty
            and not self.store.is_session_closed(self.session_id)
            and not self.store.messages(
                self.session_id, include_hidden=True
            )
        )
        if can_reuse:
            session_id = self.session_id
            self.store.set_session_title(session_id, title)
        else:
            session_id = self.store.create_session(title)
        run = self.store.start_run(case, session_id)
        self.session_id = session_id
        self.active_run = run
        self._refresh_sessions()
        self._render_transcript()
        self._flash(f"кейс #{case.id}: {case.situation[:50]}")
        if case.opener == "me":
            self._flash(f"кейс #{case.id}: начинай ты")
            return
        hidden = self.store.add_message(
            session_id,
            "user",
            OPENING_INSTRUCTION,
            meta={"hidden": True, "kind": "opening"},
        )
        self.app.create_background_task(
            self._case_reply("case_opening", hidden.id)
        )

    def _start_new_case(self, case: Any) -> None:
        pending = self.active_run or self.store.latest_active_run()
        if pending:
            self.active_run = pending
            self._flash(
                f"кейс #{case.id} добавлен; есть прогон #{pending.id} — "
                "/finish или /abandon"
            )
            return
        self._case_start(str(case.id), reuse_empty=True)

    def _case_add_only(self, text: str) -> None:
        try:
            case = self.store.add_case(text)
        except ValueError as exc:
            self._flash(str(exc)[:100])
            return
        self._flash(
            f"кейс #{case.id} добавлен; /case {case.id} — играть"
        )

    def _archive_case(
        self, case_id: int, include_archived: bool
    ) -> None:
        if self.active_run and self.active_run.case_id == case_id:
            self._flash("этот кейс сейчас играется; /finish сначала")
            return
        if self.store.archive_case(case_id):
            self._flash(f"кейс #{case_id} архивирован")
            self._cases_modal(include_archived)
        else:
            self._flash("кейс уже в архиве")

    def _restore_case(
        self, case_id: int, include_archived: bool
    ) -> None:
        if self.store.set_case_status(case_id, "new"):
            self._flash(f"кейс #{case_id} восстановлен")
            self._cases_modal(include_archived)
        else:
            self._flash("не нашёл такой кейс")

    async def _case_reply(
        self, kind: str, cleanup_on_skip: int | None
    ) -> None:
        if self.variant_mode == "short":
            await self._case_short_reply(kind, cleanup_on_skip)
        else:
            await self._case_full_reply(kind, cleanup_on_skip)

    def _cleanup_skip(self, cleanup_on_skip: int | None, message: str) -> None:
        if cleanup_on_skip:
            self.store.delete_message(cleanup_on_skip)
            self._render_transcript()
        self._flash(message)

    async def _case_short_reply(
        self, kind: str, cleanup_on_skip: int | None
    ) -> None:
        beats = await self._generate_short_candidates_async(self.case_variants)
        if not beats:
            self._cleanup_skip(cleanup_on_skip, "модель не дала заготовок")
            return
        self._show_case_beats(beats, kind, cleanup_on_skip)

    def _show_case_beats(
        self,
        beats: list[str],
        kind: str,
        cleanup_on_skip: int | None,
    ) -> None:
        items: list[tuple[Any, ...]] = []
        for index, beat in enumerate(beats):
            items.append(
                (
                    f"{index + 1}. {beat}",
                    lambda beat=beat: self.app.create_background_task(
                        self._case_expand(beat, beats, kind, cleanup_on_skip)
                    ),
                )
            )
        items.append(("----- действия -----", None, "header"))
        items.append(
            (
                "заново",
                lambda: self.app.create_background_task(
                    self._case_reply(kind, cleanup_on_skip)
                ),
            )
        )
        items.append(
            (
                "пропустить",
                lambda: self._cleanup_skip(cleanup_on_skip, "ход пропущен"),
            )
        )
        self._open_modal(
            "выберите реплику",
            items,
            hint="↑/↓ — выбор · Enter — развернуть · Esc — пропустить",
            on_escape=lambda: self._cleanup_skip(
                cleanup_on_skip, "ход пропущен"
            ),
        )

    async def _case_expand(
        self,
        beat: str,
        beats: list[str],
        kind: str,
        cleanup_on_skip: int | None,
    ) -> None:
        context = self._context_for_log()
        messages = self._with_instruction(
            self._messages(scenario=self._case_scenario()),
            expand_instruction(beat, mode=self.mode),
        )
        reply = await self._stream_and_save(messages)
        if not reply:
            self._show_case_beats(beats, kind, cleanup_on_skip)
            return
        self._log_choice(
            kind,
            final=reply,
            context=context,
            beats=beats,
            chosen_short=beats.index(beat),
        )

    async def _case_full_reply(
        self, kind: str, cleanup_on_skip: int | None
    ) -> None:
        candidates = await self._generate_candidates_async(self.case_variants)
        if not candidates:
            self._cleanup_skip(cleanup_on_skip, "модель не дала вариантов")
            return
        self._show_case_candidates(candidates, kind, cleanup_on_skip)

    def _show_case_candidates(
        self,
        candidates: list[tuple[str, float, Any]],
        kind: str,
        cleanup_on_skip: int | None,
    ) -> None:
        items: list[tuple[Any, ...]] = []
        for index, (text, _temp, _stats) in enumerate(candidates):
            items.append(
                (
                    f"{index + 1}. {' '.join(text.split())[:60]}",
                    lambda text=text, index=index: self.app.create_background_task(
                        self._case_save_candidate(
                            text, candidates, index, kind, cleanup_on_skip
                        )
                    ),
                )
            )
        items.append(("----- действия -----", None, "header"))
        items.append(
            (
                "заново",
                lambda: self.app.create_background_task(
                    self._case_reply(kind, cleanup_on_skip)
                ),
            )
        )
        items.append(
            (
                "пропустить",
                lambda: self._cleanup_skip(cleanup_on_skip, "ход пропущен"),
            )
        )
        self._open_modal(
            "выберите вариант",
            items,
            hint="↑/↓ — выбор · Enter — сохранить · Esc — пропустить",
            on_escape=lambda: self._cleanup_skip(
                cleanup_on_skip, "ход пропущен"
            ),
        )

    async def _case_save_candidate(
        self,
        text: str,
        candidates: list[tuple[str, float, Any]],
        index: int,
        kind: str,
        cleanup_on_skip: int | None,
    ) -> None:
        context = self._context_for_log()
        self.store.add_message(self.session_id, "assistant", text)
        self._refresh_sessions()
        self._render_transcript()
        self._log_choice(
            kind,
            final=text,
            context=context,
            candidates=candidates,
            chosen=index,
        )

    def _context_for_log(self) -> list[dict[str, str]]:
        return [
            message
            for message in self._messages(scenario=self._case_scenario())
            if message["content"] != OPENING_INSTRUCTION
        ]

    def _log_choice(
        self,
        kind: str,
        final: str,
        context: list[dict[str, str]],
        beats: list[str] | None = None,
        chosen_short: int | None = None,
        candidates: list[tuple[str, float, Any]] | None = None,
        chosen: int | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "type": "choice",
            "kind": kind,
            "session_id": self.session_id,
            "model": self.llm.cfg.model,
            "context": context,
            "final": final,
        }
        if self.active_run:
            record["case_id"] = self.active_run.case_id
            record["run_id"] = self.active_run.id
        if beats is not None:
            record["short_candidates"] = beats
            record["chosen_short"] = chosen_short
        if candidates is not None:
            record["candidates"] = [
                {"text": text, "temperature": temperature}
                for text, temperature, _stats in candidates
            ]
            record["chosen"] = chosen
        self._log(record)

    async def _generate_short_candidates_async(
        self, count: int
    ) -> list[str]:
        count = max(1, count)
        temps = self.cfg.chat.variant_temperatures or [self.temperature]
        base = self.seed_base
        self.seed_base += count
        loop = asyncio.get_running_loop()
        self.llm.on_progress = lambda event: loop.call_soon_threadsafe(
            self._on_progress, event
        )
        self.busy = True
        self.stage = "generation"
        self.tps = 0.0
        self.tokens = 0
        self.invalidate()
        beats: list[str] = []
        error: str | None = None
        try:
            for index in range(count):
                messages = self._with_instruction(
                    self._messages(scenario=self._case_scenario()),
                    short_variant_instruction(),
                )
                raw = await asyncio.to_thread(
                    self.llm.complete,
                    messages,
                    temperature=temps[index % len(temps)],
                    max_tokens=self.cfg.chat.short_max_tokens,
                    seed=base + index,
                )
                parsed = parse_short_variants(raw, 1)
                if parsed:
                    beats.append(parsed[0])
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            self.llm.on_progress = None
            self.busy = False
            self.stage = "ready" if error is None else "error"
            if error is not None:
                self.error = error
            self.invalidate()
        if error is not None:
            self._flash(f"ошибка генерации: {error[:80]}")
            return []
        result: list[str] = []
        seen: set[str] = set()
        for beat in beats:
            key = beat.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(beat)
        return result[:count]

    async def _generate_candidates_async(
        self, count: int
    ) -> list[tuple[str, float, Any]]:
        count = max(1, count)
        temps = self.cfg.case.temperatures or [self.temperature]
        base = self.seed_base
        self.seed_base += count
        loop = asyncio.get_running_loop()
        self.llm.on_progress = lambda event: loop.call_soon_threadsafe(
            self._on_progress, event
        )
        self.busy = True
        self.stage = "generation"
        self.tps = 0.0
        self.tokens = 0
        self.invalidate()
        candidates: list[tuple[str, float, Any]] = []
        error: str | None = None
        try:
            for index in range(count):
                text = await asyncio.to_thread(
                    self.llm.complete,
                    self._messages(scenario=self._case_scenario()),
                    temperature=temps[index % len(temps)],
                    seed=base + index,
                )
                if text:
                    candidates.append(
                        (text, temps[index % len(temps)], self.llm.last_stats)
                    )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            self.llm.on_progress = None
            self.busy = False
            self.stage = "ready" if error is None else "error"
            if error is not None:
                self.error = error
            self.invalidate()
        if error is not None:
            self._flash(f"ошибка генерации: {error[:80]}")
            return []
        return candidates

    # ── завершение / возобновление / подсказка / правка ──────────────────

    def _case_finish(self) -> None:
        if not self.active_run:
            self._flash("нет активного прогона")
            return
        self.ask_data = {}
        self._ask("rating", "оценка 1–5 (Enter — 3)")

    def _finish_run(self, rating: int, notes: str | None) -> None:
        run = self.active_run
        if not run:
            self._flash("нет активного прогона")
            return
        self.store.finish_run(run.id, rating, notes)
        turns = self.store.count_turns(run.session_id)
        replies = len(
            self.store.messages(
                run.session_id, roles=("user", "assistant"), include_hidden=False
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
        self.session_id = self.store.create_session(
            f"после кейса #{run.case_id}"
        )
        self._refresh_sessions()
        self._render_transcript()
        self._flash(f"прогон #{run.id} закрыт · {rating}/5")

    def _export_run(self, run_id: int) -> None:
        run = self.store.get_run(run_id)
        if not run:
            return
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

    def _case_resume(self) -> None:
        if self._busy_guard():
            return
        run = self.active_run or self.store.latest_active_run()
        if not run:
            self._flash("незакрытых прогонов нет")
            return
        self.session_id = run.session_id
        self.active_run = run
        self._refresh_sessions()
        self._render_transcript()
        self._flash(f"прогон #{run.id} · кейс #{run.case_id}")

    def _case_hint(self) -> None:
        if not self.active_run:
            self._flash("подсказка работает только в прогоне")
            return
        if self._busy_guard():
            return
        self.app.create_background_task(self._case_hint_async())

    async def _case_hint_async(self) -> None:
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
                    "Верни ровно три коротких варианта того, что мог бы "
                    "сказать он. Каждый вариант — с новой строки, без "
                    "нумерации, без кавычек. Пиши по-русски, живо."
                ),
            },
            {
                "role": "user",
                "content": f"{scenario}\nДиалог:\n{transcript}\n\n"
                "Что мог бы сказать он?",
            },
        ]
        self.busy = True
        self.stage = "generation"
        self.invalidate()
        loop = asyncio.get_running_loop()
        self.llm.on_progress = lambda event: loop.call_soon_threadsafe(
            self._on_progress, event
        )
        try:
            raw = await asyncio.to_thread(
                self.llm.complete, prompt, temperature=0.9, max_tokens=300
            )
        except Exception as exc:  # noqa: BLE001
            self.busy = False
            self.stage = "error"
            self.error = str(exc)
            self.invalidate()
            return
        finally:
            self.llm.on_progress = None
        self.busy = False
        self.stage = "ready"
        self.invalidate()
        lines = self._parse_hints(raw)
        if not lines:
            self._flash("не удалось разобрать подсказки")
            return
        self._open_modal(
            "подсказки",
            [(line, None) for line in lines],
            hint="↑/↓ — просмотр · Esc — назад",
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

    def _abandon(self) -> None:
        if not self.active_run:
            self._flash("нет активного прогона")
            return
        self.store.abandon_run(self.active_run.id)
        self._flash(f"прогон #{self.active_run.id} брошен")
        self.active_run = None

    def _case_edit(self, raw: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self._flash("/case + ID — дополнить поля")
            return
        case = self.store.get_case(case_id)
        if not case:
            self._flash("не нашёл такой кейс")
            return
        fields = [
            ("situation", "ситуация"),
            ("relationship", "отношения"),
            ("her_goal", "её цель"),
            ("tone", "тон"),
            ("restrictions", "нельзя"),
            ("opener", "кто начинает (her/me)"),
            ("tags", "теги"),
        ]
        self.ask_data = {
            "case_id": case_id,
            "fields": fields,
            "index": 0,
            "updates": {},
            "case": case,
        }
        self._ask_case_edit_field()

    def _ask_case_edit_field(self) -> None:
        data = self.ask_data
        index = data["index"]
        if index >= len(data["fields"]):
            self._finish_case_edit()
            return
        key, label = data["fields"][index]
        current = getattr(data["case"], key) or ""
        self._ask("case_edit", f"{label} [{current}] › пустой ввод — оставить")

    def _case_edit_next(self, text: str) -> None:
        data = self.ask_data
        index = data["index"]
        key, _label = data["fields"][index]
        if text.strip():
            data["updates"][key] = text.strip()
        data["index"] = index + 1
        self._ask_case_edit_field()

    def _finish_case_edit(self) -> None:
        data = self.ask_data
        updates = data["updates"]
        case_id = data["case_id"]
        self.ask_data = {}
        if updates:
            self.store.update_case(case_id, **updates)
            self._flash(f"кейс #{case_id} обновлён")
        else:
            self._flash("без изменений")

    def _case_delete(self, raw: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self._flash("/case del ID")
            return
        if self.active_run and self.active_run.case_id == case_id:
            self._flash("этот кейс сейчас играется; /finish сначала")
            return
        case = self.store.get_case(case_id)
        if not case:
            self._flash("не нашёл такой кейс")
            return
        if case.status == "archived":
            self._flash("кейс уже в архиве")
            return
        if self.store.archive_case(case_id):
            self._flash(f"кейс #{case_id} архивирован")

    def _case_purge(self, raw: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self._flash("/case purge ID")
            return
        if self.active_run and self.active_run.case_id == case_id:
            self._flash("этот кейс сейчас играется; /finish сначала")
            return
        if not self.store.get_case(case_id):
            self._flash("не нашёл такой кейс")
            return
        self.ask_data = {"case_id": case_id}
        self._ask("confirm_purge", f"удалить кейс #{case_id} навсегда? [y/N]")

    def _case_status(self, raw: str, status: str) -> None:
        case_id = parse_id(raw)
        if case_id is None:
            self._flash("/case done ID")
            return
        if self.store.set_case_status(case_id, status):
            self._flash(f"кейс #{case_id}: {status}")
        else:
            self._flash("не нашёл такой кейс")

    def _case_export(self) -> None:
        cases = self.store.cases(include_archived=True)
        path = self.cfg.path("data/cases.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Кейсы", "", f"Всего: {len(cases)}", ""]
        for case in cases:
            stats = self.store.run_stats(case.id)
            lines.append(f"## #{case.id} — {case.situation}")
            for label, value in (
                ("Отношения", case.relationship),
                ("Её цель", case.her_goal),
                ("Тон", case.tone),
                ("Нельзя", case.restrictions),
                ("Теги", case.tags),
            ):
                if value:
                    lines.append(f"- **{label}:** {value}")
            lines.append(
                f"- **Прогоны:** {stats['total']} · "
                f"**Лучшая:** {stats['best'] or '—'}"
            )
            lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._flash(f"экспорт: {path}")

    # ── ввод с вопросом (оценка, заметка, подтверждение, правка) ──────────

    def _ask(self, kind: str, prompt: str) -> None:
        self.ask_kind = kind
        self.ask_prompt = prompt
        self.input_area.buffer.reset()
        self._flash(prompt)
        self.app.layout.focus(self.input_area)

    def _cancel_ask(self) -> None:
        self.ask_kind = None
        self.ask_prompt = ""
        self.ask_data = {}
        self._flash("отменено")

    def _answer_ask(self, text: str) -> None:
        kind = self.ask_kind
        self.ask_kind = None
        self.ask_prompt = ""
        if kind == "rating":
            rating = 3
            if text.isdigit() and 1 <= int(text) <= 5:
                rating = int(text)
            self.ask_data["rating"] = rating
            self._ask("notes", "заметка (Enter — без заметки)")
        elif kind == "notes":
            rating = self.ask_data.get("rating", 3)
            notes = text.strip() or None
            self.ask_data = {}
            self._finish_run(rating, notes)
        elif kind == "confirm_purge":
            case_id = self.ask_data.get("case_id")
            self.ask_data = {}
            if text.strip().lower() in ("y", "yes", "д", "да"):
                if self.store.purge_case(case_id):
                    self._flash(f"кейс #{case_id} удалён навсегда")
                else:
                    self._flash("не нашёл такой кейс")
            else:
                self._flash("отменено")
        elif kind == "case_edit":
            self._case_edit_next(text)

    def _begin_edit_last(self) -> None:
        message = self.store.last_message(self.session_id, "assistant")
        if not message:
            self._flash("нет ответа для правки")
            return
        self.edit_target_id = message.id
        self.input_area.text = message.content
        self.input_area.buffer.cursor_position = len(message.content)
        self.app.layout.focus(self.input_area)
        self._flash("Enter — сохранить, Esc — отмена")

    def _cancel_edit(self) -> None:
        self.edit_target_id = None
        self.input_area.buffer.reset()
        self._flash("отменено")

    def _rate(self, value: int) -> None:
        message = self.store.last_message(self.session_id, "assistant")
        if not message:
            self._flash("нет ответа для оценки")
            return
        self._log(
            {
                "type": "rating",
                "session_id": self.session_id,
                "model": self.llm.cfg.model,
                "response": message.content,
                "rating": value,
            }
        )
        self._flash("оценка записана")

    def _log(self, record: dict[str, Any]) -> None:
        path = self.cfg.path(self.cfg.paths.preferences)
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        record = {"ts": datetime.now().astimezone().isoformat(), **record}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _remember(self, text: str) -> None:
        if not text:
            self._flash("что запомнить?")
            return
        fact_id = self.store.add_fact(text)
        self._flash(f"факт #{fact_id}" if fact_id else "пусто")

    def _facts_modal(self) -> None:
        facts = self.store.facts()
        if not facts:
            self._flash("фактов нет")
            return
        self._open_modal(
            "факты", [(f"#{fact.id}  {fact.text}", None) for fact in facts]
        )

    def _cases_modal(self, include_archived: bool = False) -> None:
        cases = self.store.cases(include_archived=include_archived)
        if not cases:
            self._flash("кейсов нет")
            return
        items: list[tuple[Any, ...]] = []
        for case in cases:
            stats = self.store.run_stats(case.id)
            label = (
                f"#{case.id}  {case.status}  "
                f"{' '.join(case.situation.split())[:44]}"
            )
            if stats["total"]:
                label += (
                    f"  · {stats['total']} прогонов"
                    f" · лучшая {stats['best'] or '—'}"
                )
            items.append(
                (
                    label,
                    lambda case_id=case.id, inc=include_archived: (
                        self._show_case_detail(case_id, inc)
                    ),
                )
            )
        self._open_modal(
            "кейсы",
            items,
            hint="↑/↓ — выбор · Enter — открыть · Esc — назад",
        )

    def _show_case_detail(
        self, case_id: int, include_archived: bool = False
    ) -> None:
        case = self.store.get_case(case_id)
        if not case:
            self._flash("не нашёл такой кейс")
            return
        stats = self.store.run_stats(case_id)
        average = stats["average"]
        average_text = f"{average:.1f}" if average is not None else "—"
        items: list[tuple[Any, ...]] = [
            (f"статус: {case.status}", None, "info"),
            (f"ситуация: {case.situation}", None, "info"),
        ]
        for label, value in (
            ("отношения", case.relationship),
            ("её цель", case.her_goal),
            ("тон", case.tone),
            ("нельзя", case.restrictions),
            ("теги", case.tags),
        ):
            if value:
                items.append((f"{label}: {value}", None, "info"))
        items.append(
            (
                f"начинает: {'она' if case.opener == 'her' else 'ты'}",
                None,
                "info",
            )
        )
        items.append(
            (
                f"прогоны: {stats['total']} · "
                f"завершено {stats['finished']} · "
                f"лучшая {stats['best'] or '—'} · "
                f"средняя {average_text}",
                None,
                "info",
            )
        )
        items.append(("----- действия -----", None, "header"))
        items.append(
            (
                "начать прогон",
                lambda: self._case_start(str(case_id)),
            )
        )
        items.append(
            (
                "редактировать",
                lambda: self._case_edit(str(case_id)),
            )
        )
        if case.status == "archived":
            items.append(
                (
                    "восстановить",
                    lambda: self._restore_case(
                        case_id, include_archived
                    ),
                )
            )
        else:
            items.append(
                (
                    "заархивировать",
                    lambda: self._archive_case(
                        case_id, include_archived
                    ),
                )
            )
        items.append(
            (
                "удалить навсегда",
                lambda: self._case_purge(str(case_id)),
            )
        )
        items.append(
            (
                "назад к списку",
                lambda: self._cases_modal(include_archived),
            )
        )
        self._open_modal(
            f"кейс #{case.id}",
            items,
            hint="↑/↓ — выбор · Enter — действие · Esc — назад",
            on_escape=lambda: self._cases_modal(include_archived),
        )

    def _forget(self, text: str) -> None:
        fact_id = parse_id(text)
        if fact_id is None:
            self._flash("/forget ID")
            return
        if self.store.deactivate_fact(fact_id):
            self._flash(f"факт #{fact_id} убран")
        else:
            self._flash("не нашёл факт")

    async def _compact(self) -> None:
        session_id = self.session_id
        summary = self.store.latest_summary(session_id)
        after = summary.upto_message_id if summary else 0
        messages = self.store.messages(
            session_id,
            after_id=after,
            roles=("user", "assistant"),
            include_hidden=False,
        )
        keep = max(6, self.cfg.chat.history_messages // 2)
        if len(messages) <= keep:
            self._flash("нечего сжимать")
            return
        to_summarize = messages[:-keep]
        if len(to_summarize) < 4:
            self._flash("слишком мало сообщений")
            return
        transcript = "\n".join(
            f"{'Он' if message.role == 'user' else 'Она'}: {message.content}"
            for message in to_summarize
        )
        prompt = summarization_messages(
            transcript,
            previous_summary=summary.text if summary else None,
        )
        self.busy = True
        self.stage = "prompt"
        self.invalidate()
        loop = asyncio.get_running_loop()
        self.llm.on_progress = lambda event: loop.call_soon_threadsafe(
            self._on_progress, event
        )
        try:
            raw = await asyncio.to_thread(
                self.llm.complete,
                prompt,
                temperature=0.3,
                max_tokens=700,
            )
        except Exception as exc:  # noqa: BLE001
            self.busy = False
            self.stage = "error"
            self.error = str(exc)
            self.invalidate()
            return
        finally:
            self.llm.on_progress = None
        text, facts = parse_summary(raw)
        if text:
            self.store.add_summary(
                session_id, to_summarize[-1].id, text
            )
            for fact in facts:
                self.store.add_fact(fact)
            self._flash("память обновлена")
        else:
            self._flash("модель вернула пустую сводку")
        self.busy = False
        self.stage = "ready"
        self.invalidate()

    def _run_command(self, text: str) -> None:
        if text.startswith("\\"):
            text = expand_alias(text)
        name = text.split(" ", 1)[0].lower()
        if name in ("/quit", "/exit", "/q", "\\й"):
            self.app.exit()
            return
        if name in ("/new", "\\т"):
            title = text.split(" ", 1)[1] if " " in text else None
            self._new_session(title)
            self.invalidate()
            return
        if name in ("/sessions", "/session", "\\ы"):
            rest = text.split(" ", 1)[1].strip().lower() if " " in text else ""
            self._show_sessions(
                rest in ("all", "архив", "archive")
            )
            return
        if name == "/close":
            self._close_session(
                text.split(" ", 1)[1].strip() if " " in text else ""
            )
            return
        if name == "/reopen":
            self._reopen_session(
                text.split(" ", 1)[1].strip() if " " in text else ""
            )
            return
        if name == "/switch":
            if self._busy_guard():
                return
            arg = text.split(" ", 1)[1] if " " in text else ""
            session_id = parse_id(arg)
            if session_id is not None:
                self._switch_session(session_id)
                self.app.layout.focus(self.input_area)
                self.invalidate()
            else:
                self._flash("/switch ID")
            return
        if name in ("/help", "\\р", "/h", "/?"):
            self._open_help()
            return

        arg = text.split(" ", 1)[1].strip() if " " in text else ""
        if name in ("/again", "\\ф"):
            self._again()
        elif name == "/undo":
            if self._busy_guard():
                return
            if self._closed_guard():
                return
            deleted = self.store.delete_last_exchange(self.session_id)
            self._render_transcript()
            self._flash(f"удалено сообщений: {deleted}")
        elif name == "/edit":
            if self._busy_guard():
                return
            if self._closed_guard():
                return
            self._begin_edit_last()
        elif name in ("/like", "\\д"):
            self._rate(1)
        elif name in ("/dislike", "\\не"):
            self._rate(-1)
        elif name == "/remember":
            self._remember(arg)
        elif name in ("/facts", "\\а"):
            self._facts_modal()
        elif name == "/forget":
            self._forget(arg)
        elif name == "/compact":
            if self._busy_guard():
                return
            if self._closed_guard():
                return
            self.app.create_background_task(self._compact())
        elif name == "/persona":
            path = self.cfg.path(self.cfg.paths.persona)
            if arg in ("reload", "перечитать"):
                try:
                    self.persona = path.read_text(encoding="utf-8")
                    self.name = persona_name(self.persona)
                    self._render_transcript()
                    self._flash(f"персонаж перечитан: {self.name}")
                except OSError as exc:
                    self._flash(str(exc)[:80])
            elif arg in ("edit", "редактировать"):
                self._edit_persona(path)
            else:
                self._flash(f"персонаж: {self.name} · {path}")
        elif name == "/temp":
            try:
                self.temperature = max(0.0, min(2.0, float(arg)))
                self._flash(f"температура: {self.temperature:.2f}")
            except ValueError:
                self._flash("/temp 0.85")
        elif name == "/variants":
            try:
                self.variants = max(0, min(8, int(arg)))
                self._flash(
                    "варианты выключены"
                    if self.variants == 0
                    else f"вариантов: {self.variants}"
                )
            except ValueError:
                self._flash("/variants 3")
        elif name == "/model":
            if arg:
                self._set_model(arg)
            else:
                self._model_menu()
        elif name == "/load":
            self._load_model()
        elif name == "/unload":
            self._unload_model()
        elif name in ("/loaded", "/ps"):
            self._loaded_hint()
        elif name == "/export":
            self._export(arg or "md")
        elif name == "/stats":
            self._stats_hint()
        elif name == "/history":
            try:
                self.history_limit = max(4, min(200, int(arg)))
                self._render_transcript()
                self._flash(f"история: {self.history_limit}")
            except ValueError:
                self._flash("/history 40")
        elif name in ("/cases", "\\ксы"):
            self._cases_modal(
                include_archived=arg.lower() in ("all", "архив", "archive")
            )
        elif name == "/case":
            self._case_command(arg)
        elif name == "/case_suggest":
            self._case_suggest(arg)
        elif name == "/case_edit":
            self._case_edit(arg)
        elif name == "/finish":
            self._case_finish()
        elif name == "/resume":
            self._case_resume()
        elif name == "/hint":
            self._case_hint()
        elif name == "/abandon":
            self._abandon()
        else:
            self._flash(f"не знаю команду {name}")

    # ── клавиши ───────────────────────────────────────────────────────────

    def _bindings(self) -> KeyBindings:
        kb = build_key_bindings()

        @kb.add(Keys.ControlC, eager=True)
        @kb.add(Keys.ControlQ, eager=True)
        def _(event) -> None:
            event.app.exit()

        @kb.add(Keys.Tab)
        def _(event) -> None:
            app = event.app
            if app.layout.has_focus(self.input_area):
                app.layout.focus(self.transcript_buffer)
            elif app.layout.has_focus(self.transcript_buffer):
                app.layout.focus(self.session_control)
            else:
                app.layout.focus(self.input_area)

        @kb.add(Keys.Up, filter=has_focus(self.session_control), eager=True)
        def _(event) -> None:
            if self.sessions:
                self.sel_index = max(0, self.sel_index - 1)
                self.invalidate()

        @kb.add(Keys.Down, filter=has_focus(self.session_control), eager=True)
        def _(event) -> None:
            if self.sessions:
                self.sel_index = min(len(self.sessions) - 1, self.sel_index + 1)
                self.invalidate()

        @kb.add("enter", filter=has_focus(self.session_control), eager=True)
        def _(event) -> None:
            if self.sessions:
                self._switch_session(self.sessions[self.sel_index]["id"])
                event.app.layout.focus(self.input_area)

        @kb.add(Keys.ControlN)
        def _(event) -> None:
            if self._busy_guard():
                return
            self._new_session()
            event.app.layout.focus(self.input_area)
            self.invalidate()

        @kb.add("enter", filter=has_focus(self.input_area), eager=True)
        def _(event) -> None:
            text = self.input_area.text
            self.input_area.buffer.reset()
            self._submit(text)

        @kb.add("escape", "enter", filter=has_focus(self.input_area))
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.PageUp, filter=has_focus(self.transcript_window))
        @kb.add(Keys.Up, filter=has_focus(self.transcript_window))
        def _(event) -> None:
            self.at_bottom = False

        @kb.add(Keys.End, filter=has_focus(self.transcript_window))
        def _(event) -> None:
            self.at_bottom = True
            self.transcript_buffer.cursor_position = len(
                self.transcript_buffer.text
            )

        modal = Condition(lambda: self.modal_open)

        @kb.add(Keys.ControlP)
        def _(event) -> None:
            if not self.modal_open:
                self._open_palette()

        @kb.add("escape", filter=modal, eager=True)
        def _(event) -> None:
            self._close_modal()

        @kb.add(Keys.Up, filter=modal, eager=True)
        def _(event) -> None:
            self._modal_move(-1)

        @kb.add(Keys.Down, filter=modal, eager=True)
        def _(event) -> None:
            self._modal_move(1)

        @kb.add("enter", filter=modal, eager=True)
        def _(event) -> None:
            self._modal_run()

        @kb.add(Keys.Backspace, filter=modal, eager=True)
        def _(event) -> None:
            self._modal_backspace()

        @kb.add("<any>", filter=modal, eager=True)
        def _(event) -> None:
            self._modal_filter_char(event.data)

        @kb.add(
            "escape",
            filter=Condition(lambda: self.edit_target_id is not None),
            eager=True,
        )
        def _cancel_edit(event) -> None:
            self._cancel_edit()

        @kb.add(
            "escape",
            filter=Condition(lambda: self.ask_kind is not None),
            eager=True,
        )
        def _cancel_ask(event) -> None:
            self._cancel_ask()

        return kb


def run_tui(cfg: Config, session_id: int | None = None) -> None:
    TuiApp(cfg, session_id).run()


if __name__ == "__main__":
    sys.stderr.write("Запускай через `python -m companion`, а не напрямую.\n")
    raise SystemExit(1)
