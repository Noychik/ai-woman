from __future__ import annotations

import asyncio
import json
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

from .cli import build_key_bindings, parse_id, persona_name
from .config import Config
from .llm import BaseLLM, LLMError, make_llm, strip_thinking
from .prompt import build_messages as assemble_messages
from .store import Store


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

        self.session_id = self._pick_session(session_id)
        self.sessions: list[dict[str, Any]] = []
        self.sel_index = 0
        self.busy = False
        self.at_bottom = True

        self.stage = "ready"
        self.tps = 0.0
        self.tokens = 0
        self.streaming = ""
        self.streaming_active = False
        self.error = ""
        self.hint = ""

        self.modal_open = False
        self.modal_title = ""
        self.modal_items: list[tuple[str, Any]] = []
        self.modal_filter = ""
        self.modal_sel = 0
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
        self.app.run(pre_run=self._focus_input)

    def _focus_input(self) -> None:
        self.app.layout.focus(self.input_area)

    def _pick_session(self, session_id: int | None) -> int:
        if session_id is not None:
            if not self.store.get_session(session_id):
                raise SystemExit(f"Сессии {session_id} нет.")
            return session_id
        latest = self.store.latest_session()
        if latest is not None:
            return latest
        return self.store.create_session("первый разговор")

    def _switch_session(self, session_id: int) -> None:
        if not self.store.get_session(session_id):
            return
        self.session_id = session_id
        self.at_bottom = True
        self.streaming_active = False
        self.streaming = ""
        self._refresh_sessions()
        self._render_transcript()
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
            style = "class:sidebar.active" if index == self.sel_index else ""
            title = (session.get("title") or "разговор")[:18]
            preview = (session.get("preview") or "—")[:16]
            lines.append((style, f"{mark} {session['id']:>3} {title}\n"))
            lines.append(("class:help", f"       {preview}\n"))
        return lines

    def _refresh_sessions(self) -> None:
        rows = self.store.sessions()
        self.sessions = [
            {
                "id": row["id"],
                "title": row["title"] or "разговор",
                "n": row["n"],
                "run_id": row["run_id"],
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
        for message in messages[-TRANSCRIPT_LIMIT:]:
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

    def _messages(self) -> list[dict[str, str]]:
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
        return assemble_messages(
            persona=self.persona,
            facts=facts,
            summary=summary.text if summary else None,
            history=history,
            max_chars=self.cfg.chat.max_context_chars,
            language=self.cfg.chat.language,
            scenario=None,
            adult=self.adult,
            mode=self.mode,
        )

    # ── ввод и генерация ──────────────────────────────────────────────────

    def _submit(self, text: str) -> None:
        self.hint = ""
        text = text.strip()
        if not text:
            return
        if text.startswith("/") or text.startswith("\\"):
            self._run_command(text)
            return
        if self.busy:
            self.hint = "подожди, идёт генерация"
            self.invalidate()
            return
        self.store.add_message(self.session_id, "user", text)
        self._refresh_sessions()
        self._render_transcript()
        self.app.create_background_task(self._generate())

    def _flash(self, message: str) -> None:
        self.hint = message
        self.invalidate()

    # ── модальный список ──────────────────────────────────────────────────

    def _open_modal(self, title: str, items: list[tuple[str, Any]]) -> None:
        self.modal_title = title
        self.modal_items = items
        self.modal_filter = ""
        self.modal_sel = 0
        self.modal_open = True
        self.app.layout.focus(self.modal_control)
        self.invalidate()

    def _close_modal(self) -> None:
        self.modal_open = False
        self.modal_items = []
        self.modal_filter = ""
        self._focus_input()
        self.invalidate()

    def _filtered_items(self) -> list[tuple[str, Any]]:
        needle = self.modal_filter.lower()
        if not needle:
            return self.modal_items
        return [
            item
            for item in self.modal_items
            if needle in item[0].lower()
        ]

    def _modal_text(self) -> Any:
        lines: list[Any] = [("class:sidebar.active", f" {self.modal_title}\n"), ("", "\n")]
        items = self._filtered_items()
        if not items:
            lines.append(("class:help", "  (ничего не найдено)\n"))
        for index, (label, _) in enumerate(items[:40]):
            mark = ">" if index == self.modal_sel else " "
            style = "class:sidebar.active" if index == self.modal_sel else ""
            lines.append((style, f" {mark} {label}\n"))
        lines.append(("class:help", "\n Enter — выбрать · Esc — назад"))
        return lines

    def _modal_run(self) -> None:
        items = self._filtered_items()
        if not items:
            return
        action = items[min(self.modal_sel, len(items) - 1)][1]
        self.modal_open = False
        self._focus_input()
        if callable(action):
            action()
        self.invalidate()

    def _modal_filter_char(self, char: str) -> None:
        if char and char.isprintable():
            self.modal_filter += char
            self.modal_sel = 0
            self.invalidate()

    def _modal_backspace(self) -> None:
        self.modal_filter = self.modal_filter[:-1]
        self.modal_sel = 0
        self.invalidate()

    def _modal_move(self, delta: int) -> None:
        items = self._filtered_items()
        if not items:
            return
        self.modal_sel = (self.modal_sel + delta) % len(items)
        self.invalidate()

    # ── палитра команд ────────────────────────────────────────────────────

    def _command_items(self) -> list[tuple[str, Any]]:
        return [
            ("новая сессия", self._new_session),
            ("сессии (сайдбар)", lambda: self.app.layout.focus(self.session_control)),
            ("выбор модели…", self._model_menu),
            (f"режим: {self.mode} (переключить)", self._toggle_mode),
            ("загрузить модель", self._load_model),
            ("выгрузить модель", self._unload_model),
            ("статистика", self._stats_hint),
            ("экспорт markdown", lambda: self._export("md")),
            ("экспорт jsonl", lambda: self._export("jsonl")),
            ("помощь", self._open_help),
            ("выход", self.app.exit),
        ]

    def _open_palette(self) -> None:
        self._open_modal("команды", self._command_items())

    def _help_items(self) -> list[tuple[str, Any]]:
        return [
            ("Enter — отправить сообщение", None),
            ("Alt+Enter — перенос строки во вводе", None),
            ("Tab — фокус: ввод → история → сессии", None),
            ("↑/↓ + Enter — выбрать сессию в сайдбаре", None),
            ("Ctrl+N — новая сессия", None),
            ("Ctrl+P — палитра команд", None),
            ("PageUp/PageDown, Home/End — прокрутка истории", None),
            ("Ctrl+←/→, Ctrl+Backspace — по словам", None),
            ("Ctrl+C / Ctrl+Q — выход", None),
            ("команды: /new, /sessions, /switch, /model, /help", None),
            ("полный список команд — /help в --no-tui", None),
        ]

    def _open_help(self) -> None:
        self._open_modal("помощь (Esc — закрыть)", self._help_items())

    def _model_menu(self) -> None:
        models = self.llm.list_models()
        if not models:
            models = [self.llm.cfg.model]
        items = [(name, lambda name=name: self._set_model(name)) for name in models]
        self._open_modal("модели", items)

    def _set_model(self, name: str) -> None:
        self.llm.cfg.model = name
        self._flash(f"модель: {name}")

    def _toggle_mode(self) -> None:
        self.mode = "messaging" if self.mode == "roleplay" else "roleplay"
        self.cfg.chat.mode = self.mode
        self._flash(f"режим: {self.mode}")

    def _new_session(self) -> None:
        self.session_id = self.store.create_session()
        self._refresh_sessions()
        self._render_transcript()

    def _load_model(self) -> None:
        try:
            self.llm.load_model()
            self._flash("модель загружена")
        except LLMError as exc:
            self._flash(str(exc)[:80])

    def _unload_model(self) -> None:
        try:
            self.llm.unload_model()
            self._flash("модель выгружена")
        except LLMError as exc:
            self._flash(str(exc)[:80])

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

    async def _generate(self) -> None:
        self.busy = True
        self.streaming = ""
        self.streaming_active = True
        self.stage = "connecting"
        self.tps = 0.0
        self.tokens = 0
        self.invalidate()

        messages = self._messages()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(self._on_progress, event)

        self.llm.on_progress = on_progress

        def producer() -> None:
            chunks: list[str] = []
            try:
                for chunk in self.llm.stream(
                    messages, temperature=self.temperature
                ):
                    chunks.append(chunk)
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
            self._render_transcript()
            self.invalidate()
            return

        reply = strip_thinking(self.streaming).strip()
        self.streaming_active = False
        self.streaming = ""
        if reply:
            self.store.add_message(self.session_id, "assistant", reply)
            self._refresh_sessions()
        self.busy = False
        self.stage = "ready"
        self._render_transcript()
        self.invalidate()

    def _on_progress(self, event: dict[str, Any]) -> None:
        if event.get("stage"):
            self.stage = str(event["stage"])
        if event.get("tokens") is not None:
            self.tokens = int(event["tokens"])
        if event.get("tokens_per_second") is not None:
            self.tps = float(event["tokens_per_second"])
        self.invalidate()

    # ── команды (базовые) ────────────────────────────────────────────────

    def _run_command(self, text: str) -> None:
        name = text.split(" ", 1)[0].lower()
        if name in ("/quit", "/exit", "/q", "\\й"):
            self.app.exit()
            return
        if name in ("/new", "\\т"):
            title = text.split(" ", 1)[1] if " " in text else None
            self.session_id = self.store.create_session(title)
            self._refresh_sessions()
            self._render_transcript()
            self.invalidate()
            return
        if name in ("/sessions", "/session", "\\ы"):
            self._refresh_sessions()
            self.app.layout.focus(self.session_control)
            self.invalidate()
            return
        if name == "/switch":
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
        self._flash(f"команда {name} пока не в TUI")

    # ── клавиши ───────────────────────────────────────────────────────────

    def _bindings(self) -> KeyBindings:
        kb = build_key_bindings()

        @kb.add(Keys.ControlC)
        @kb.add(Keys.ControlQ)
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
            self.session_id = self.store.create_session()
            self._refresh_sessions()
            self._render_transcript()
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

        return kb


def run_tui(cfg: Config, session_id: int | None = None) -> None:
    TuiApp(cfg, session_id).run()


if __name__ == "__main__":
    sys.stderr.write("Запускай через `python -m companion`, а не напрямую.\n")
    raise SystemExit(1)
