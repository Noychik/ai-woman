from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from companion.config import (
    CaseConfig,
    ChatConfig,
    Config,
    ContentConfig,
    LLMConfig,
    MemoryConfig,
    PathsConfig,
)
from companion.tui import TuiApp


class TuiConstructionTests(unittest.TestCase):
    def test_tui_constructs_with_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "persona.md").write_text("# Алиса\n\nТы — Алиса.\n", encoding="utf-8")
            cfg = Config(
                root=root,
                llm=LLMConfig(backend="mock", model="mock"),
                chat=ChatConfig(),
                paths=PathsConfig(),
                memory=MemoryConfig(),
                case=CaseConfig(),
                content=ContentConfig(),
            )
            app = TuiApp(cfg)
            self.addCleanup(app.close)
            self.assertIsNotNone(app.session_id)
            self.assertIsNotNone(app.app)

    def test_command_items_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "persona.md").write_text("# Алиса\n", encoding="utf-8")
            cfg = Config(
                root=root,
                llm=LLMConfig(backend="mock", model="mock"),
                chat=ChatConfig(),
                paths=PathsConfig(),
                memory=MemoryConfig(),
                case=CaseConfig(),
                content=ContentConfig(),
            )
            app = TuiApp(cfg)
            self.addCleanup(app.close)
            self.assertGreater(len(app._command_items()), 5)

    def test_modal_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "persona.md").write_text("# Алиса\n", encoding="utf-8")
            cfg = Config(
                root=root,
                llm=LLMConfig(backend="mock", model="mock"),
                chat=ChatConfig(),
                paths=PathsConfig(),
                memory=MemoryConfig(),
                case=CaseConfig(),
                content=ContentConfig(),
            )
            app = TuiApp(cfg)
            self.addCleanup(app.close)
            app.modal_items = [("новая сессия", None), ("экспорт", None)]
            app.modal_filter = "экспорт"
            self.assertEqual(len(app._filtered_items()), 1)

    def test_tui_chat_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "persona.md").write_text("# Алиса\n\nТы — Алиса.\n", encoding="utf-8")
            cfg = Config(
                root=root,
                llm=LLMConfig(backend="mock", model="mock"),
                chat=ChatConfig(),
                paths=PathsConfig(),
                memory=MemoryConfig(),
                case=CaseConfig(),
                content=ContentConfig(),
            )
            app = TuiApp(cfg)
            self.addCleanup(app.close)

            app._run_command("/remember любит чай")
            facts = app.store.facts()
            self.assertEqual([fact.text for fact in facts], ["любит чай"])
            app._run_command(f"/forget {facts[0].id}")
            self.assertEqual(app.store.facts(), [])

            app._run_command("/variants 3")
            self.assertEqual(app.variants, 3)
            app._run_command("/history 12")
            self.assertEqual(app.history_limit, 12)
            app._run_command("/temp 0.5")
            self.assertEqual(app.temperature, 0.5)
            app._run_command("/case кейс про чай")
            self.assertEqual(len(app.store.cases()), 1)

            session = app.session_id
            app.store.add_message(session, "user", "привет")
            app.store.add_message(session, "assistant", "ответ")
            app._run_command("/like")
            preferences = cfg.path(cfg.paths.preferences)
            self.assertIn('"rating": 1', preferences.read_text(encoding="utf-8"))

            app._run_command("/edit")
            self.assertEqual(app.input_area.text, "ответ")
            app._cancel_edit()
            app._run_command("/undo")
            self.assertEqual(app.store.last_message(session), None)


if __name__ == "__main__":
    unittest.main()
