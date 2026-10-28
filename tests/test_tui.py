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
            app.modal_items = [("новая сессия", None), ("экспорт", None)]
            app.modal_filter = "экспорт"
            self.assertEqual(len(app._filtered_items()), 1)


if __name__ == "__main__":
    unittest.main()
