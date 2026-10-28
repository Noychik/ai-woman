from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from companion.cli import expand_alias, wrap_words
from companion.prompt import (
    build_messages,
    case_suggestion_messages,
    expand_instruction,
    parse_case_card,
    parse_short_variants,
    short_variant_instruction,
)
from companion.store import Message, Store


def make_message(index: int, role: str, text: str) -> Message:
    return Message(
        id=index,
        session_id=1,
        role=role,
        content=text,
        created_at="2026-01-01T00:00:00+00:00",
    )


class MigrationTests(unittest.TestCase):
    def test_migrates_old_text_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.db"
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                CREATE TABLE cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    tags TEXT,
                    status TEXT NOT NULL DEFAULT 'new',
                    created_at TEXT NOT NULL,
                    used_at TEXT
                );
                INSERT INTO cases(text, created_at) VALUES ('она ревнует', '2026-01-01');
                """
            )
            conn.commit()
            conn.close()

            store = Store(path)
            try:
                case = store.cases()[0]
                self.assertEqual(case.situation, "она ревнует")
                self.assertEqual(case.opener, "her")
                self.assertTrue(case.updated_at)
                second = store.add_case("другая сцена")
                self.assertEqual(second.id, 2)
            finally:
                store.close()


class CaseAndRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "chat.db")
        self.case = self.store.add_case(
            "она нашла старый профиль",
            relationship="встречаемся три месяца",
            her_goal="получить объяснение",
            tone="серьёзный",
        )
        self.session = self.store.create_session("кейс")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_snapshot_has_no_my_goal(self) -> None:
        self.assertNotIn("my_goal", self.case.snapshot())

    def test_case_fields_and_update(self) -> None:
        self.assertEqual(self.case.relationship, "встречаемся три месяца")
        updated = self.store.update_case(self.case.id, tone="резкий", my_goal="не врать")
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.tone, "резкий")
        self.assertEqual(updated.my_goal, "не врать")

    def test_run_lifecycle(self) -> None:
        run = self.store.start_run(self.case, self.session)
        self.assertEqual(run.status, "active")
        self.assertEqual(run.scenario["situation"], "она нашла старый профиль")
        self.assertEqual(self.store.get_case(self.case.id).status, "used")
        self.assertEqual(
            self.store.active_run_for_session(self.session).id, run.id
        )

        self.assertTrue(self.store.finish_run(run.id, 4, "хорошо"))
        finished = self.store.get_run(run.id)
        assert finished is not None
        self.assertEqual(finished.status, "finished")
        self.assertEqual(finished.rating, 4)
        self.assertIsNone(self.store.active_run_for_session(self.session))

        stats = self.store.run_stats(self.case.id)
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["finished"], 1)
        self.assertEqual(stats["best"], 4)

    def test_archive_keeps_runs(self) -> None:
        run = self.store.start_run(self.case, self.session)
        self.assertTrue(self.store.archive_case(self.case.id))
        archived = self.store.get_case(self.case.id)
        assert archived is not None
        self.assertEqual(archived.status, "archived")
        self.assertEqual(self.store.cases(), [])
        self.assertEqual(len(self.store.cases(include_archived=True)), 1)
        self.assertEqual(len(self.store.runs(case_id=self.case.id)), 1)
        self.assertIsNotNone(self.store.get_run(run.id))

    def test_archived_case_revives_on_add(self) -> None:
        self.store.archive_case(self.case.id)
        revived = self.store.add_case(self.case.situation)
        self.assertEqual(revived.id, self.case.id)
        self.assertEqual(revived.status, "new")
        self.assertEqual(len(self.store.cases()), 1)

    def test_purge_deletes_runs(self) -> None:
        run = self.store.start_run(self.case, self.session)
        self.assertTrue(self.store.purge_case(self.case.id))
        self.assertIsNone(self.store.get_case(self.case.id))
        self.assertIsNone(self.store.get_run(run.id))
        self.assertEqual(self.store.runs(case_id=self.case.id), [])

    def test_hidden_messages(self) -> None:
        self.store.add_message(
            self.session, "user", "начни сцену", meta={"hidden": True}
        )
        self.store.add_message(self.session, "user", "видимая реплика")
        visible = self.store.messages(self.session, include_hidden=False)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].content, "видимая реплика")
        self.assertEqual(self.store.count_turns(self.session), 1)


class ScenarioPromptTests(unittest.TestCase):
    def test_scenario_is_in_system_prompt(self) -> None:
        messages = build_messages(
            persona="Ты — Алиса.",
            facts=[],
            summary=None,
            history=[make_message(1, "user", "привет")],
            max_chars=5000,
            scenario={
                "situation": "она нашла старый профиль",
                "tone": "серьёзный",
                "her_goal": "объяснение",
            },
        )
        system = messages[0]["content"]
        self.assertIn("## Сцена", system)
        self.assertIn("она нашла старый профиль", system)
        self.assertIn("серьёзный", system)
        self.assertIn("объяснение", system)
        self.assertEqual(messages[1]["role"], "user")

    def test_scenario_ignores_my_goal(self) -> None:
        messages = build_messages(
            persona="Ты — Алиса.",
            facts=[],
            summary=None,
            history=[make_message(1, "user", "привет")],
            max_chars=5000,
            scenario={
                "situation": "она ревнует",
                "my_goal": "не ссориться",
            },
        )
        self.assertNotIn("Его цель", messages[0]["content"])
        self.assertNotIn("не ссориться", messages[0]["content"])

    def test_no_scenario_for_normal_chat(self) -> None:
        messages = build_messages(
            persona="Ты — Алиса.",
            facts=[],
            summary=None,
            history=[make_message(1, "user", "привет")],
            max_chars=5000,
        )
        self.assertNotIn("## Сцена", messages[0]["content"])

    def test_messaging_prompt_has_no_actions(self) -> None:
        messages = build_messages(
            persona="Ты — Алиса.",
            facts=[],
            summary=None,
            history=[make_message(1, "user", "привет")],
            max_chars=5000,
            mode="messaging",
        )
        system = messages[0]["content"]
        self.assertIn("мессенджере", system)
        self.assertIn("ASCII", system)
        self.assertNotIn("Действия и мимику", system)

    def test_roleplay_prompt_has_actions(self) -> None:
        messages = build_messages(
            persona="Ты — Алиса.",
            facts=[],
            summary=None,
            history=[make_message(1, "user", "привет")],
            max_chars=5000,
            mode="roleplay",
        )
        self.assertIn("Действия и мимику", messages[0]["content"])

    def test_messaging_expand_has_no_actions(self) -> None:
        instruction = expand_instruction("ну ты и выбрал момент", mode="messaging")
        self.assertIn("мессенджере", instruction)
        self.assertNotIn("*улыбается*", instruction)
        self.assertNotIn("*отводит", instruction)
        self.assertIn("ASCII", instruction)


class ShortVariantTests(unittest.TestCase):
    def test_parse_numbered(self) -> None:
        raw = "1. явился, блин\n2) ты вообще думал?\n- кофе будешь?"
        self.assertEqual(
            parse_short_variants(raw, 3),
            ["явился, блин", "ты вообще думал?", "кофе будешь?"],
        )

    def test_dedup_and_limit(self) -> None:
        raw = "да\nда\nнет\nещё"
        self.assertEqual(parse_short_variants(raw, 2), ["да", "нет"])

    def test_semicolon_fallback(self) -> None:
        self.assertEqual(
            parse_short_variants("да; нет; может", 3),
            ["да", "нет", "может"],
        )

    def test_actions_removed(self) -> None:
        raw = (
            "1. Привет. Снова в очереди? *усмехается*\n"
            "2. *отводит взгляд* ну да"
        )
        self.assertEqual(
            parse_short_variants(raw, 2),
            ["Привет. Снова в очереди?", "ну да"],
        )

    def test_instructions(self) -> None:
        self.assertIn("одну", short_variant_instruction())
        self.assertIn("заготовка", expand_instruction("да").lower())


class CaseSuggestionTests(unittest.TestCase):
    def test_parse_card(self) -> None:
        raw = (
            '```json\n{"situation": "она ревнует", '
            '"relationship": "три месяца", "opener": "her"}\n```'
        )
        card = parse_case_card(raw)
        self.assertIsNotNone(card)
        assert card is not None
        self.assertEqual(card["situation"], "она ревнует")
        self.assertEqual(card["relationship"], "три месяца")
        self.assertEqual(card["opener"], "her")

    def test_parse_card_with_trailing_text(self) -> None:
        raw = '{"situation": "опоздал на час", "tone": "раздражённый"} вот так'
        card = parse_case_card(raw)
        self.assertIsNotNone(card)
        assert card is not None
        self.assertEqual(card["tone"], "раздражённый")

    def test_parse_card_invalid(self) -> None:
        self.assertIsNone(parse_case_card("просто текст"))
        self.assertIsNone(parse_case_card('{"tone": "без ситуации"}'))

    def test_prompt_contains_seed(self) -> None:
        messages = case_suggestion_messages(
            "Ты — Алиса.", seed="ревность", existing=["первое свидание"]
        )
        self.assertIn("ревность", messages[-1]["content"])
        self.assertIn("первое свидание", messages[-1]["content"])
        self.assertIn("JSON", messages[0]["content"])

    def test_adult_prompt_keeps_full_persona(self) -> None:
        persona = "Ты — Алиса.\n\n## Близость\n\n- Ты можешь хотеть близости."
        messages = case_suggestion_messages(
            persona, seed="сцена секса", adult=True
        )
        system = messages[0]["content"]
        self.assertIn("## Близость", system)
        self.assertIn("Взрослые сцены разрешены", system)
        self.assertIn("не смягчай", messages[-1]["content"])

    def test_sexual_seed_starts_in_media_res(self) -> None:
        messages = case_suggestion_messages(
            "Ты — Алиса.", seed="сцена секса", adult=True
        )
        user = messages[-1]["content"]
        self.assertIn("уже занимаетесь сексом", user)
        self.assertIn("Не описывай прелюдию", user)

    def test_sfw_prompt_has_no_adult_block(self) -> None:
        messages = case_suggestion_messages(
            "Ты — Алиса.", seed="ревность", adult=False
        )
        self.assertNotIn("Взрослые сцены разрешены", messages[0]["content"])

        chat = build_messages(
            persona="Ты — Алиса.",
            facts=[],
            summary=None,
            history=[make_message(1, "user", "привет")],
            max_chars=5000,
            adult=False,
        )
        self.assertNotIn("Ты взрослая женщина", chat[0]["content"])


class EditingTests(unittest.TestCase):
    def test_wrap_words_does_not_split_long_word(self) -> None:
        text = "короткое оченьдлинноесловобезпробелов конец"
        wrapped = wrap_words(text, 12)
        self.assertIn("оченьдлинноесловобезпробелов", wrapped)
        self.assertIn("\n", wrapped)


class AliasTests(unittest.TestCase):
    def test_russian_layout_aliases(self) -> None:
        self.assertEqual(expand_alias("\\м"), "/variants")
        self.assertEqual(expand_alias("\\мфк 3"), "/variants 3")
        self.assertEqual(expand_alias("\\к 12"), "/case 12")
        self.assertEqual(expand_alias("\\к+ 12"), "/case_edit 12")
        self.assertEqual(expand_alias("\\кп"), "/case_suggest")
        self.assertEqual(expand_alias("\\кп ревность"), "/case_suggest ревность")
        self.assertEqual(expand_alias("\\придумай"), "/case_suggest")
        self.assertEqual(expand_alias("\\ксы"), "/cases")
        self.assertEqual(expand_alias("\\ч"), "/finish")
        self.assertEqual(expand_alias("\\п"), "/resume")
        self.assertEqual(expand_alias("\\пд"), "/hint")
        self.assertEqual(expand_alias("\\ые"), "/stats")
        self.assertEqual(expand_alias("\\р"), "/help")
        self.assertEqual(expand_alias("\\й"), "/quit")

    def test_memory_aliases(self) -> None:
        self.assertEqual(expand_alias("\\+"), "/load")
        self.assertEqual(expand_alias("\\-"), "/unload")
        self.assertEqual(expand_alias("\\пс"), "/loaded")
        self.assertEqual(expand_alias("\\загрузи"), "/load")
        self.assertEqual(expand_alias("\\выгрузи"), "/unload")
        self.assertEqual(expand_alias("\\зг"), "/load")
        self.assertEqual(expand_alias("\\вг"), "/unload")

    def test_unknown_alias_stays(self) -> None:
        self.assertEqual(expand_alias("\\что-то"), "\\что-то")


if __name__ == "__main__":
    unittest.main()
