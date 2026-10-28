from __future__ import annotations

import asyncio
import json
import os
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

from .cli import build_key_bindings, parse_id, persona_name
from .config import Config
from .llm import BaseLLM, LLMError, make_llm, strip_thinking
from .prompt import (
    build_messages as assemble_messages,
    expand_instruction,
    parse_short_variants,
    parse_summary,
    short_variant_instruction,
    summarization_messages,
)
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

        self.session_id = self._pick_session(session_id)
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
        latest = self.store.latest_session()
        if latest is not None:
            return latest
        return self.store.create_session("первый разговор")

    def _switch_session(self, session_id: int) -> None:
        if self._busy_guard():
            return
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

    def _messages(self, session_id: int | None = None) -> list[dict[str, str]]:
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
        if self.busy:
            self.hint = "подожди, идёт генерация"
            self.invalidate()
            return
        session_id = self.session_id
        self.store.add_message(session_id, "user", text)
        self._refresh_sessions()
        self._render_transcript()
        if self.variants > 0:
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

    def _open_modal(self, title: str, items: list[tuple[Any, ...]]) -> None:
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

    def _filtered_items(self) -> list[tuple[Any, ...]]:
        needle = self.modal_filter.lower()
        if not needle:
            return self.modal_items
        return [
            item
            for item in self.modal_items
            if needle in item[0].lower()
        ]

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
            mark = ">" if index == sel else " "
            style = "class:sidebar.active" if index == sel else ""
            lines.append((style, f" {mark} {label}\n"))
        if end < len(items):
            lines.append(("class:help", "  … ниже …\n"))
        lines.append(("class:help", "\n ↑/↓ — выбор · Enter — применить · Esc — назад"))
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

    def _help_items(self) -> list[tuple[Any, ...]]:
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
            ("/new [название] — новая сессия", None),
            ("/sessions — список сессий", None),
            ("/switch ID — переключиться на сессию", None),
            ("/again — перегенерировать последний ответ", None),
            ("/undo — удалить последний обмен", None),
            ("/edit — править последний ответ (Enter — сохранить, Esc — отмена)", None),
            ("/like | /dislike — оценка последнего ответа", None),
            ("/variants N — N вариантов (0 — обычный поток)", None),
            ("/temp X — температура 0–2", None),
            ("/model [имя] — сменить или выбрать модель", None),
            ("/history N — сколько сообщений показывать", None),
            ("/export [md|jsonl] — выгрузить диалог", None),
            ("/stats — статистика", None),
            ("/quit — выход", None),
            ("----- память и персонаж -----", None, "header"),
            ("/remember текст — добавить факт", None),
            ("/facts — показать факты", None),
            ("/forget ID — убрать факт", None),
            ("/compact — сжать историю в сводку", None),
            ("/persona — карточка персонажа", None),
            ("/persona edit — править карточку", None),
            ("/persona reload — перечитать карточку", None),
            ("----- модель -----", None, "header"),
            ("/load — загрузить модель в память", None),
            ("/unload — выгрузить модель", None),
            ("/loaded — что загружено", None),
            ("----- кейсы -----", None, "header"),
            ("/case текст — добавить кейс", None),
            ("/cases, /cases all — список кейсов", None),
            ("прогон и правка кейсов — в --no-tui", None),
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
        if self._busy_guard():
            return
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
    ) -> None:
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
            return

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
        last = self.store.last_message(self.session_id)
        if last and last.role == "assistant":
            self.store.delete_message(last.id)
        last = self.store.last_message(self.session_id)
        if not last or last.role != "user":
            self._flash("нечего повторять")
            return
        self._render_transcript()
        self.invalidate()
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
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        try:
            self.app.run_system_command(
                f"{editor} {shlex.quote(str(path))}",
                wait_for_enter=True,
                display_before_text="редактор персонажа",
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
            self._flash("/case текст — добавить кейс; прогон — в --no-tui")
            return
        if arg.isdigit():
            self._flash("прогон кейса пока в --no-tui")
            return
        lowered = arg.lower()
        if lowered in ("del", "delete", "purge", "gen", "suggest"):
            self._flash("управление кейсами — в --no-tui")
            return
        try:
            case = self.store.add_case(arg)
        except ValueError as exc:
            self._flash(str(exc)[:100])
            return
        self._flash(f"кейс #{case.id} добавлен; дополнить — в --no-tui")

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
        self._open_modal(
            "кейсы",
            [
                (
                    f"#{case.id}  {case.status}  "
                    f"{' '.join(case.situation.split())[:50]}",
                    None,
                )
                for case in cases
            ],
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
        name = text.split(" ", 1)[0].lower()
        if name in ("/quit", "/exit", "/q", "\\й"):
            self.app.exit()
            return
        if name in ("/new", "\\т"):
            if self._busy_guard():
                return
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
            deleted = self.store.delete_last_exchange(self.session_id)
            self._render_transcript()
            self._flash(f"удалено сообщений: {deleted}")
        elif name == "/edit":
            if self._busy_guard():
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
            if not self._busy_guard():
                self._load_model()
        elif name == "/unload":
            if not self._busy_guard():
                self._unload_model()
        elif name == "/loaded":
            loaded = self.llm.loaded_models()
            if loaded:
                total = sum(
                    int(model.get("size") or 0) for model in loaded
                )
                self._flash(
                    f"в памяти: {len(loaded)} · {total / 1024**3:.1f} ГБ"
                )
            else:
                self._flash("модель не загружена")
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
        elif name in (
            "/case_edit",
            "/case_suggest",
            "/finish",
            "/resume",
            "/hint",
            "/abandon",
            "\\к",
            "\\ч",
            "\\п",
            "\\пд",
        ):
            self._flash("кейсы и прогоны пока в --no-tui")
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

        @kb.add(
            "escape",
            filter=Condition(lambda: self.edit_target_id is not None),
            eager=True,
        )
        def _cancel_edit(event) -> None:
            self._cancel_edit()

        return kb


def run_tui(cfg: Config, session_id: int | None = None) -> None:
    TuiApp(cfg, session_id).run()


if __name__ == "__main__":
    sys.stderr.write("Запускай через `python -m companion`, а не напрямую.\n")
    raise SystemExit(1)
