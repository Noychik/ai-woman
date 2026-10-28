from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from companion.config import load_config
from companion.llm import LLMConfig, MockLLM, strip_thinking
from companion.prompt import (
    build_messages,
    parse_summary,
    summarization_messages,
    trim_history,
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


class PromptTests(unittest.TestCase):
    def test_trim_keeps_last_user_message(self) -> None:
        history = [
            make_message(1, "user", "а" * 500),
            make_message(2, "assistant", "б" * 500),
            make_message(3, "user", "в" * 500),
            make_message(4, "assistant", "г" * 500),
            make_message(5, "user", "последний вопрос"),
        ]
        trimmed = trim_history(history, max_chars=1200, keep_last=2)
        self.assertEqual(trimmed[-1].content, "последний вопрос")
        self.assertEqual(trimmed[0].role, "user")

    def test_build_messages_starts_with_system_and_user(self) -> None:
        history = [
            make_message(1, "assistant", "привет"),
            make_message(2, "user", "как дела"),
            make_message(3, "assistant", "нормально"),
            make_message(4, "user", "что делаешь"),
        ]
        messages = build_messages(
            persona="Ты — Алиса.",
            facts=["его зовут Дима"],
            summary="Они познакомились в кофейне.",
            history=history,
            max_chars=4000,
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Алиса", messages[0]["content"])
        self.assertIn("Дима", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "что делаешь")

    def test_parse_summary_json(self) -> None:
        raw = '```json\n{"summary": "Они поссорились.", "facts": ["он любит кофе"]}\n```'
        summary, facts = parse_summary(raw)
        self.assertEqual(summary, "Они поссорились.")
        self.assertEqual(facts, ["он любит кофе"])

    def test_parse_summary_fallback(self) -> None:
        summary, facts = parse_summary("просто текст без json")
        self.assertEqual(summary, "просто текст без json")
        self.assertEqual(facts, [])

    def test_summarization_prompt_asks_for_json(self) -> None:
        messages = summarization_messages("Он: привет\nОна: привет")
        self.assertIn("JSON", messages[0]["content"])
        self.assertEqual(messages[-1]["role"], "user")

    def test_summarization_merges_previous_summary(self) -> None:
        messages = summarization_messages(
            "Он: я переехал в Новосибирск",
            previous_summary="Он учится в НГТУ.",
        )
        user = messages[-1]["content"]
        self.assertIn("НГТУ", user)
        self.assertIn("Новосибирск", user)
        self.assertIn("Обнови сводку", user)
        self.assertIn("не теряй", user.lower())


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "chat.db")
        self.session = self.store.create_session("тест")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_messages_roundtrip(self) -> None:
        self.store.add_message(self.session, "user", "привет")
        self.store.add_message(self.session, "assistant", "привет")
        messages = self.store.messages(self.session)
        self.assertEqual([m.role for m in messages], ["user", "assistant"])

    def test_fact_deduplication(self) -> None:
        first = self.store.add_fact("его зовут Дима")
        second = self.store.add_fact("Его зовут Дима")
        self.assertEqual(first, second)
        self.assertEqual(len(self.store.facts()), 1)

    def test_forget_then_remember_reactivates(self) -> None:
        fact_id = self.store.add_fact("он любит кофе")
        self.assertTrue(self.store.deactivate_fact(fact_id))
        again = self.store.add_fact("он любит кофе")
        self.assertEqual(fact_id, again)
        self.assertEqual(len(self.store.facts()), 1)

    def test_summary_and_count(self) -> None:
        for index in range(5):
            self.store.add_message(
                self.session, "user" if index % 2 == 0 else "assistant", f"m{index}"
            )
        messages = self.store.messages(self.session)
        self.store.add_summary(self.session, messages[2].id, "сводка")
        self.assertEqual(self.store.count_since_summary(self.session), 2)

    def test_delete_last_exchange(self) -> None:
        self.store.add_message(self.session, "user", "вопрос")
        self.store.add_message(self.session, "assistant", "ответ")
        deleted = self.store.delete_last_exchange(self.session)
        self.assertEqual(deleted, 2)
        self.assertEqual(self.store.count_messages(self.session), 0)

    def test_delete_dangling_user(self) -> None:
        self.store.add_message(self.session, "user", "вопрос без ответа")
        deleted = self.store.delete_last_exchange(self.session)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.store.count_messages(self.session), 0)


class ThinkingTests(unittest.TestCase):
    def test_strip_thinking(self) -> None:
        self.assertEqual(
            strip_thinking("<think>\nдолго думал\n</think>\n\nпривет"),
            "привет",
        )
        self.assertEqual(strip_thinking("без тегов"), "без тегов")


class MockLLMTests(unittest.TestCase):
    def test_mock_streams_text(self) -> None:
        llm = MockLLM(LLMConfig())
        text = llm.complete(
            [{"role": "user", "content": "привет"}], temperature=0.7
        )
        self.assertTrue(text)
        self.assertIn("mock", text)
        self.assertIsNotNone(llm.last_stats)


if __name__ == "__main__":
    unittest.main()
