from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Message:
    id: int
    session_id: int
    role: str
    content: str
    created_at: str
    meta: dict[str, Any] | None = None

    @property
    def hidden(self) -> bool:
        return bool((self.meta or {}).get("hidden"))


@dataclass
class Fact:
    id: int
    text: str
    created_at: str
    active: int = 1


@dataclass
class Summary:
    id: int
    session_id: int
    upto_message_id: int
    text: str
    created_at: str


@dataclass
class Case:
    id: int
    situation: str
    status: str
    created_at: str
    updated_at: str
    relationship: str | None = None
    her_goal: str | None = None
    my_goal: str | None = None
    tone: str | None = None
    restrictions: str | None = None
    opener: str = "her"
    tags: str | None = None
    used_at: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "situation": self.situation,
            "relationship": self.relationship,
            "her_goal": self.her_goal,
            "my_goal": self.my_goal,
            "tone": self.tone,
            "restrictions": self.restrictions,
            "opener": self.opener,
            "tags": self.tags,
        }


@dataclass
class Run:
    id: int
    case_id: int
    session_id: int
    scenario_snapshot: str
    status: str
    created_at: str
    rating: int | None = None
    notes: str | None = None
    finished_at: str | None = None

    @property
    def scenario(self) -> dict[str, Any]:
        try:
            data = json.loads(self.scenario_snapshot)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    title      TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta       TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    upto_message_id INTEGER NOT NULL,
    text            TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON summaries(session_id, upto_message_id);
"""

CASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    situation    TEXT NOT NULL,
    relationship TEXT,
    her_goal     TEXT,
    my_goal      TEXT,
    tone         TEXT,
    restrictions TEXT,
    opener       TEXT NOT NULL DEFAULT 'her',
    tags         TEXT,
    status       TEXT NOT NULL DEFAULT 'new',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    used_at      TEXT
);
"""

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id           INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    session_id        INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    scenario_snapshot TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active',
    rating            INTEGER,
    notes             TEXT,
    created_at        TEXT NOT NULL,
    finished_at       TEXT,
    UNIQUE(session_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_case
    ON runs(case_id, id);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs(status, id);
"""

CASE_FIELDS = (
    "situation",
    "relationship",
    "her_goal",
    "my_goal",
    "tone",
    "restrictions",
    "opener",
    "tags",
    "status",
)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(BASE_SCHEMA)
        self._migrate_cases()
        self.conn.executescript(RUNS_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ── миграция ──────────────────────────────────────────────────────────

    def _migrate_cases(self) -> None:
        exists = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'cases'"
        ).fetchone()
        if not exists:
            self.conn.executescript(CASES_SCHEMA)
            self.conn.commit()
            return

        columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(cases)")
        }

        # Черновая версия хранила только text. Пересобираем таблицу,
        # чтобы избавиться от NOT NULL на старом поле.
        if "situation" not in columns:
            self.conn.execute("DROP TABLE IF EXISTS cases_legacy")
            self.conn.execute("ALTER TABLE cases RENAME TO cases_legacy")
            self.conn.executescript(CASES_SCHEMA)
            self.conn.execute(
                "INSERT INTO cases("
                "id, situation, tags, status, created_at, updated_at, used_at"
                ") SELECT id, text, tags, status, created_at, created_at, used_at "
                "FROM cases_legacy"
            )
            self.conn.execute("DROP TABLE cases_legacy")
            self.conn.commit()
            return

        additions = {
            "relationship": "TEXT",
            "her_goal": "TEXT",
            "my_goal": "TEXT",
            "tone": "TEXT",
            "restrictions": "TEXT",
            "opener": "TEXT NOT NULL DEFAULT 'her'",
            "tags": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'new'",
            "updated_at": "TEXT",
            "used_at": "TEXT",
        }
        for name, ddl in additions.items():
            if name not in columns:
                self.conn.execute(f"ALTER TABLE cases ADD COLUMN {name} {ddl}")
        self.conn.execute(
            "UPDATE cases SET updated_at = created_at WHERE updated_at IS NULL"
        )
        self.conn.execute(
            "UPDATE cases SET opener = 'her' WHERE opener IS NULL OR opener = ''"
        )
        self.conn.commit()

    # ── сессии ────────────────────────────────────────────────────────────

    def create_session(self, title: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions(started_at, title) VALUES (?, ?)",
            (utcnow(), title),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_session(self, session_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

    def latest_session(self) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return int(row["id"]) if row else None

    def sessions(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT s.*, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n, "
            "  (SELECT r.id FROM runs r WHERE r.session_id = s.id) AS run_id "
            "FROM sessions s ORDER BY s.id DESC"
        ).fetchall()

    # ── сообщения ─────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        now = utcnow()
        cur = self.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                role,
                content,
                now,
                json.dumps(meta, ensure_ascii=False) if meta else None,
            ),
        )
        self.conn.commit()
        return Message(int(cur.lastrowid), session_id, role, content, now, meta)

    def messages(
        self,
        session_id: int,
        after_id: int = 0,
        limit: int | None = None,
        roles: tuple[str, ...] | None = None,
        include_hidden: bool = True,
    ) -> list[Message]:
        sql = "SELECT * FROM messages WHERE session_id = ? AND id > ?"
        params: list[Any] = [session_id, after_id]
        if roles:
            placeholders = ",".join("?" for _ in roles)
            sql += f" AND role IN ({placeholders})"
            params.extend(roles)
        if limit is None:
            sql += " ORDER BY id"
        else:
            sql = (
                "SELECT * FROM (" + sql + " ORDER BY id DESC LIMIT ?) ORDER BY id"
            )
            params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        result = [self._message(row) for row in rows]
        if not include_hidden:
            result = [message for message in result if not message.hidden]
        return result

    def last_message(self, session_id: int, role: str | None = None) -> Message | None:
        if role:
            row = self.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role = ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id, role),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return self._message(row) if row else None

    def update_message(self, message_id: int, content: str) -> None:
        self.conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?", (content, message_id)
        )
        self.conn.commit()

    def delete_message(self, message_id: int) -> None:
        self.conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        self.conn.commit()

    def delete_last_exchange(self, session_id: int) -> int:
        """Удаляет последний ответ и предшествующий ему вопрос.

        Если ответа нет (генерацию прервали), удаляет висящую реплику пользователя.
        """
        assistant = self.last_message(session_id, "assistant")
        if not assistant:
            user = self.last_message(session_id, "user")
            if not user:
                return 0
            self.conn.execute("DELETE FROM messages WHERE id = ?", (user.id,))
            self.conn.commit()
            return 1

        user = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND role = 'user' "
            "AND id < ? ORDER BY id DESC LIMIT 1",
            (session_id, assistant.id),
        ).fetchone()
        self.conn.execute("DELETE FROM messages WHERE id = ?", (assistant.id,))
        deleted = 1
        if user:
            self.conn.execute("DELETE FROM messages WHERE id = ?", (user["id"],))
            deleted += 1
        self.conn.commit()
        return deleted

    def count_messages(self, session_id: int, include_hidden: bool = True) -> int:
        if include_hidden:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row["n"])
        return len(self.messages(session_id, include_hidden=False))

    def count_turns(self, session_id: int) -> int:
        return len(
            self.messages(
                session_id, roles=("user",), include_hidden=False
            )
        )

    # ── память ────────────────────────────────────────────────────────────

    def latest_summary(self, session_id: int) -> Summary | None:
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE session_id = ? "
            "ORDER BY upto_message_id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return self._summary(row) if row else None

    def add_summary(self, session_id: int, upto_message_id: int, text: str) -> Summary:
        now = utcnow()
        cur = self.conn.execute(
            "INSERT INTO summaries(session_id, upto_message_id, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, upto_message_id, text, now),
        )
        self.conn.commit()
        return Summary(int(cur.lastrowid), session_id, upto_message_id, text, now)

    def count_since_summary(self, session_id: int) -> int:
        summary = self.latest_summary(session_id)
        after = summary.upto_message_id if summary else 0
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE session_id = ? AND id > ? AND role IN ('user', 'assistant')",
            (session_id, after),
        ).fetchone()
        return int(row["n"])

    # ── факты ─────────────────────────────────────────────────────────────

    def facts(self, active_only: bool = True) -> list[Fact]:
        sql = "SELECT * FROM facts"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY id"
        return [self._fact(r) for r in self.conn.execute(sql).fetchall()]

    def add_fact(self, text: str) -> int | None:
        text = " ".join(text.split()).strip(" -•\t")
        if len(text) < 3:
            return None
        normalized = text.casefold()
        for fact in self.facts(active_only=False):
            if fact.text.casefold() == normalized:
                if not fact.active:
                    self.conn.execute(
                        "UPDATE facts SET active = 1 WHERE id = ?", (fact.id,)
                    )
                    self.conn.commit()
                return fact.id
        cur = self.conn.execute(
            "INSERT INTO facts(text, created_at) VALUES (?, ?)", (text, utcnow())
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def deactivate_fact(self, fact_id: int) -> bool:
        cur = self.conn.execute(
            "UPDATE facts SET active = 0 WHERE id = ? AND active = 1", (fact_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ── кейсы ─────────────────────────────────────────────────────────────

    def add_case(self, situation: str, **fields: Any) -> Case:
        situation = " ".join(situation.split()).strip()
        if len(situation) < 3:
            raise ValueError("кейс слишком короткий")
        for case in self.cases(include_archived=True):
            if case.situation.casefold() == situation.casefold():
                if case.status == "archived":
                    self.conn.execute(
                        "UPDATE cases SET status = 'new', used_at = NULL, "
                        "updated_at = ? WHERE id = ?",
                        (utcnow(), case.id),
                    )
                    self.conn.commit()
                    refreshed = self.get_case(case.id)
                    return refreshed or case
                return case

        now = utcnow()
        values = {
            "situation": situation,
            "relationship": fields.get("relationship"),
            "her_goal": fields.get("her_goal"),
            "my_goal": fields.get("my_goal"),
            "tone": fields.get("tone"),
            "restrictions": fields.get("restrictions"),
            "opener": fields.get("opener") or "her",
            "tags": fields.get("tags"),
            "status": "new",
            "created_at": now,
            "updated_at": now,
        }
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        cur = self.conn.execute(
            f"INSERT INTO cases({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        self.conn.commit()
        case = self.get_case(int(cur.lastrowid))
        assert case is not None
        return case

    def cases(
        self,
        status: str | None = None,
        include_archived: bool = False,
    ) -> list[Case]:
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        elif not include_archived:
            conditions.append("status != 'archived'")
        sql = "SELECT * FROM cases"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY id"
        return [self._case(row) for row in self.conn.execute(sql, params).fetchall()]

    def get_case(self, case_id: int) -> Case | None:
        row = self.conn.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        return self._case(row) if row else None

    def update_case(self, case_id: int, **fields: Any) -> Case | None:
        updates = {
            key: value
            for key, value in fields.items()
            if key in CASE_FIELDS and value is not None
        }
        if not updates:
            return self.get_case(case_id)
        updates["updated_at"] = utcnow()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        self.conn.execute(
            f"UPDATE cases SET {assignments} WHERE id = ?",
            (*updates.values(), case_id),
        )
        self.conn.commit()
        return self.get_case(case_id)

    def set_case_status(self, case_id: int, status: str) -> bool:
        cur = self.conn.execute(
            "UPDATE cases SET status = ?, updated_at = ? WHERE id = ?",
            (status, utcnow(), case_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def archive_case(self, case_id: int) -> bool:
        cur = self.conn.execute(
            "UPDATE cases SET status = 'archived', updated_at = ? "
            "WHERE id = ? AND status != 'archived'",
            (utcnow(), case_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def purge_case(self, case_id: int) -> bool:
        """Жёсткое удаление вместе с прогонами. Использовать только после подтверждения."""
        cur = self.conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_case(self, case_id: int) -> bool:
        return self.purge_case(case_id)

    def case_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM cases GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    # ── прогоны ───────────────────────────────────────────────────────────

    def start_run(self, case: Case, session_id: int) -> Run:
        now = utcnow()
        snapshot = json.dumps(case.snapshot(), ensure_ascii=False)
        cur = self.conn.execute(
            "INSERT INTO runs("
            "case_id, session_id, scenario_snapshot, status, created_at"
            ") VALUES (?, ?, ?, 'active', ?)",
            (case.id, session_id, snapshot, now),
        )
        self.conn.execute(
            "UPDATE cases SET status = 'used', used_at = ?, updated_at = ? "
            "WHERE id = ?",
            (now, now, case.id),
        )
        self.conn.commit()
        run = self.get_run(int(cur.lastrowid))
        assert run is not None
        return run

    def get_run(self, run_id: int) -> Run | None:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return self._run(row) if row else None

    def runs(
        self,
        case_id: int | None = None,
        status: str | None = None,
    ) -> list[Run]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        conditions: list[str] = []
        if case_id is not None:
            conditions.append("case_id = ?")
            params.append(case_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY id"
        return [self._run(row) for row in self.conn.execute(sql, params).fetchall()]

    def active_run_for_session(self, session_id: int) -> Run | None:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE session_id = ? AND status = 'active' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return self._run(row) if row else None

    def latest_active_run(self) -> Run | None:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._run(row) if row else None

    def finish_run(self, run_id: int, rating: int, notes: str | None) -> bool:
        cur = self.conn.execute(
            "UPDATE runs SET status = 'finished', rating = ?, notes = ?, "
            "finished_at = ? WHERE id = ?",
            (rating, notes, utcnow(), run_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def abandon_run(self, run_id: int) -> bool:
        cur = self.conn.execute(
            "UPDATE runs SET status = 'abandoned', finished_at = ? WHERE id = ?",
            (utcnow(), run_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def run_stats(self, case_id: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, "
            "  SUM(CASE WHEN status = 'finished' THEN 1 ELSE 0 END) AS finished, "
            "  MAX(rating) AS best, "
            "  AVG(CASE WHEN rating IS NOT NULL THEN rating END) AS average "
            "FROM runs WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        return {
            "total": int(row["total"] or 0),
            "finished": int(row["finished"] or 0),
            "best": int(row["best"]) if row["best"] is not None else None,
            "average": float(row["average"]) if row["average"] is not None else None,
        }

    # ── служебное ─────────────────────────────────────────────────────────

    def _message(self, row: sqlite3.Row) -> Message:
        meta = json.loads(row["meta"]) if row["meta"] else None
        return Message(
            id=int(row["id"]),
            session_id=int(row["session_id"]),
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            meta=meta,
        )

    def _summary(self, row: sqlite3.Row) -> Summary:
        return Summary(
            id=int(row["id"]),
            session_id=int(row["session_id"]),
            upto_message_id=int(row["upto_message_id"]),
            text=row["text"],
            created_at=row["created_at"],
        )

    def _fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=int(row["id"]),
            text=row["text"],
            created_at=row["created_at"],
            active=int(row["active"]),
        )

    def _case(self, row: sqlite3.Row) -> Case:
        return Case(
            id=int(row["id"]),
            situation=row["situation"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"] or row["created_at"],
            relationship=row["relationship"],
            her_goal=row["her_goal"],
            my_goal=row["my_goal"],
            tone=row["tone"],
            restrictions=row["restrictions"],
            opener=row["opener"] or "her",
            tags=row["tags"],
            used_at=row["used_at"],
        )

    def _run(self, row: sqlite3.Row) -> Run:
        return Run(
            id=int(row["id"]),
            case_id=int(row["case_id"]),
            session_id=int(row["session_id"]),
            scenario_snapshot=row["scenario_snapshot"],
            status=row["status"],
            created_at=row["created_at"],
            rating=row["rating"],
            notes=row["notes"],
            finished_at=row["finished_at"],
        )
