from __future__ import annotations

import sys
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from .cli import build_key_bindings, persona_name
from .config import Config
from .llm import BaseLLM, make_llm
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
    }
)

TRANSCRIPT_LIMIT = 60


class TuiApp:
    """Полноэкранный интерфейс: сайдбар сессий, история, ввод, статус."""

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

        self.transcript_buffer = Buffer(read_only=True, multiline=True)
        self.status_control = FormattedTextControl(self._status_text)
        self.session_control = FormattedTextControl(self._sessions_text)
        self.input_area = TextArea(
            prompt="ты › ",
            multiline=True,
            height=Dimension(min=1, max=3),
            style="class:input",
            history=None,
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
        return VSplit([sidebar, right])

    # ── текст ─────────────────────────────────────────────────────────────

    def _status_text(self) -> Any:
        state = "генерация" if self.busy else "готова"
        return [
            ("class:status", f" {self.llm.cfg.model} · {self.mode} · "),
            ("class:status", f"сессия {self.session_id} · {state}"),
        ]

    def _sessions_text(self) -> Any:
        if not self.sessions:
            return [("class:help", " сессий нет")]
        lines: list[Any] = [("class:help", " сессии\n")]
        for index, session in enumerate(self.sessions):
            mark = ">" if index == self.sel_index else " "
            style = "class:sidebar.active" if index == self.sel_index else ""
            title = (session.get("title") or "разговор")[:20]
            lines.append((style, f"{mark} {session['id']:>3} {title}\n"))
        return lines

    def _refresh_sessions(self) -> None:
        rows = self.store.sessions()
        self.sessions = [
            {
                "id": row["id"],
                "title": row["title"] or "разговор",
                "n": row["n"],
                "run_id": row["run_id"],
            }
            for row in rows
        ]
        if self.sel_index >= len(self.sessions):
            self.sel_index = max(0, len(self.sessions) - 1)

    # ── transcript ────────────────────────────────────────────────────────

    def _transcript_lines(self) -> list[tuple[str, str]]:
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
        lines = self._transcript_lines()
        text = "\n".join(content for _, content in lines)
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

    # ── ввод ──────────────────────────────────────────────────────────────

    def _submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.store.add_message(self.session_id, "user", text)
        self._refresh_sessions()
        self._render_transcript()

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
                app.layout.focus(self.input_area)

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

        return kb


def run_tui(cfg: Config, session_id: int | None = None) -> None:
    TuiApp(cfg, session_id).run()


if __name__ == "__main__":
    sys.stderr.write("Запускай через `python -m companion`, а не напрямую.\n")
    raise SystemExit(1)
