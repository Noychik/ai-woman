from __future__ import annotations

import json
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

    def test_help_insert_commands(self) -> None:
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
            items = app._help_items()

            def find(prefix):
                for item in items:
                    if item[0].startswith(prefix) and callable(item[1]):
                        return item[1]
                return None

            find("/again")()
            self.assertEqual(app.input_area.text, "/again")

            find("/remember")()
            self.assertEqual(app.input_area.text, "/remember ")

            find("/switch")()
            self.assertEqual(app.input_area.text, "/switch ")

    def test_case_suggest_card_modal(self) -> None:
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

            app._show_case_card(
                {
                    "situation": "она ревнует",
                    "relationship": "встречаетесь",
                    "her_goal": "объяснение",
                    "tone": "серьёзный",
                    "restrictions": "не мириться",
                    "opener": "her",
                },
                "идея",
            )
            self.assertTrue(app.modal_open)

            save = next(
                item[1]
                for item in app.modal_items
                if item[0] == "сохранить кейс"
            )
            save()
            cases = app.store.cases()
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].situation, "она ревнует")
            self.assertEqual(cases[0].relationship, "встречаетесь")
            self.assertEqual(cases[0].opener, "her")

    def test_case_finish_and_export(self) -> None:
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

            case = app.store.add_case("тестовый кейс")
            run = app.store.start_run(case, app.session_id)
            app.active_run = run
            app._finish_run(5, "хорошо")

            self.assertIsNone(app.active_run)
            finished = app.store.get_run(run.id)
            self.assertIsNotNone(finished)
            self.assertEqual(finished.status, "finished")
            self.assertEqual(finished.rating, 5)
            self.assertEqual(finished.notes, "хорошо")

            run_file = root / "data/runs" / f"run_{run.id}.json"
            self.assertTrue(run_file.exists())
            data = json.loads(run_file.read_text(encoding="utf-8"))
            self.assertEqual(data["rating"], 5)

            prefs = (root / "data/preferences.jsonl").read_text(encoding="utf-8")
            self.assertIn('"type": "run_finish"', prefs)

    def test_case_edit_flow(self) -> None:
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

            case = app.store.add_case("старый кейс")
            app._case_edit(str(case.id))
            self.assertEqual(app.ask_kind, "case_edit")

            app._answer_ask("")
            app._answer_ask("подруги")
            for _ in range(5):
                app._answer_ask("")

            updated = app.store.get_case(case.id)
            self.assertIsNotNone(updated)
            self.assertEqual(updated.relationship, "подруги")
            self.assertIsNone(app.ask_kind)

    def test_case_purge_confirmation(self) -> None:
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

            case = app.store.add_case("удалить меня")
            app._case_purge(str(case.id))
            self.assertEqual(app.ask_kind, "confirm_purge")
            app._answer_ask("y")
            self.assertIsNone(app.store.get_case(case.id))

    def test_parse_hints(self) -> None:
        self.assertEqual(
            TuiApp._parse_hints("1. раз\n2) два\n- три\n"),
            ["раз", "два", "три"],
        )

    def test_help_has_case_commands(self) -> None:
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
            labels = [item[0] for item in app._help_items()]
            for prefix in (
                "/case ID",
                "/case + ID",
                "/case purge ID",
                "/case export",
                "/finish",
                "/resume",
                "/hint",
                "/abandon",
            ):
                self.assertTrue(
                    any(label.startswith(prefix) for label in labels),
                    prefix,
                )


if __name__ == "__main__":
    unittest.main()
