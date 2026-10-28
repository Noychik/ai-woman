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


def make_app(root: Path) -> TuiApp:
    (root / "persona.md").write_text(
        "# Алиса\n\nТы — Алиса.\n", encoding="utf-8"
    )
    cfg = Config(
        root=root,
        llm=LLMConfig(backend="mock", model="mock"),
        chat=ChatConfig(),
        paths=PathsConfig(),
        memory=MemoryConfig(),
        case=CaseConfig(),
        content=ContentConfig(),
    )
    return TuiApp(cfg)


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
            app._run_command("/case add кейс про чай")
            self.assertEqual(len(app.store.cases()), 1)
            self.assertIsNone(app.active_run)

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

            labels = [item[0] for item in items]
            for prefix in (
                "/case ID",
                "/case + ID",
                "/case purge ID",
                "/case export",
                "/finish",
                "/resume",
                "/hint",
                "/abandon",
                "/close",
                "/reopen",
                "/sessions all",
            ):
                self.assertTrue(
                    any(label.startswith(prefix) for label in labels),
                    prefix,
                )

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
                    "opener": "me",
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
            self.assertEqual(cases[0].opener, "me")
            self.assertIsNotNone(app.active_run)

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

    def test_close_session_switches_when_another_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            first = app.session_id
            second = app.store.create_session("вторая")
            app.session_id = second
            app._refresh_sessions()

            app._close_session("")

            self.assertTrue(app.store.is_session_closed(second))
            self.assertFalse(app.store.is_session_closed(first))
            self.assertEqual(app.session_id, first)

    def test_close_last_session_creates_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            first = app.session_id

            app._close_session("")

            self.assertTrue(app.store.is_session_closed(first))
            self.assertNotEqual(app.session_id, first)
            self.assertFalse(app.store.is_session_closed(app.session_id))

    def test_reopen_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            first = app.session_id
            app._close_session("")
            self.assertNotEqual(app.session_id, first)

            app._reopen_session(str(first))

            self.assertEqual(app.session_id, first)
            self.assertFalse(app.store.is_session_closed(first))

    def test_closed_session_blocks_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            session = app.session_id
            app.store.close_session(session)

            app._submit("привет")

            self.assertEqual(app.store.count_messages(session), 0)
            self.assertIn("закрыта", app.hint)

    def test_closed_sessions_hidden_from_sidebar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            first = app.session_id
            second = app.store.create_session("вторая")
            app.session_id = second
            app._refresh_sessions()

            app._close_session(str(first))
            app.show_closed_sessions = False
            app._refresh_sessions()
            ids = [session["id"] for session in app.sessions]
            self.assertNotIn(first, ids)

            app.show_closed_sessions = True
            app._refresh_sessions()
            closed = [
                session for session in app.sessions if session["id"] == first
            ]
            self.assertEqual(len(closed), 1)
            self.assertTrue(closed[0]["closed"])

    def test_new_case_autostarts_in_empty_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            session = app.session_id
            case = app.store.add_case("новая сцена", opener="me")

            app._start_new_case(case)

            self.assertEqual(app.session_id, session)
            self.assertIsNotNone(app.active_run)
            self.assertEqual(app.active_run.case_id, case.id)
            self.assertEqual(app.store.get_case(case.id).status, "used")
            self.assertEqual(
                app.store.get_session(session)["title"],
                f"кейс #{case.id}: новая сцена",
            )

    def test_new_case_creates_session_when_busy_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            old_session = app.session_id
            app.store.add_message(old_session, "user", "старое сообщение")
            case = app.store.add_case("новая сцена", opener="me")

            app._start_new_case(case)

            self.assertNotEqual(app.session_id, old_session)
            self.assertIsNotNone(app.active_run)
            self.assertEqual(app.active_run.case_id, case.id)

    def test_new_case_with_active_run_only_adds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            first = app.store.add_case("первая сцена", opener="me")
            app._start_new_case(first)
            active = app.active_run
            self.assertIsNotNone(active)

            second = app.store.add_case("вторая сцена", opener="me")
            app._start_new_case(second)

            self.assertEqual(app.active_run.id, active.id)
            self.assertEqual(len(app.store.runs(case_id=second.id)), 0)

    def test_case_detail_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            case = app.store.add_case("сцена для деталей")

            app._show_case_detail(case.id, include_archived=True)

            self.assertTrue(app.modal_open)
            labels = [item[0] for item in app.modal_items]
            self.assertIn("начать прогон", labels)
            self.assertIn("редактировать", labels)
            self.assertIn("заархивировать", labels)
            self.assertIn("удалить навсегда", labels)
            self.assertIn("назад к списку", labels)
            self.assertTrue(
                any("сцена для деталей" in label for label in labels)
            )

            archive = next(
                item[1] for item in app.modal_items
                if item[0] == "заархивировать"
            )
            archive()
            self.assertEqual(
                app.store.get_case(case.id).status, "archived"
            )

    def test_case_detail_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(Path(tmp))
            self.addCleanup(app.close)
            case = app.store.add_case("архивная сцена")
            app.store.archive_case(case.id)

            app._show_case_detail(case.id, include_archived=True)
            labels = [item[0] for item in app.modal_items]
            self.assertIn("восстановить", labels)

            restore = next(
                item[1] for item in app.modal_items
                if item[0] == "восстановить"
            )
            restore()
            self.assertEqual(app.store.get_case(case.id).status, "new")

    def test_parse_hints(self) -> None:
        self.assertEqual(
            TuiApp._parse_hints("1. раз\n2) два\n- три\n"),
            ["раз", "два", "три"],
        )


if __name__ == "__main__":
    unittest.main()
