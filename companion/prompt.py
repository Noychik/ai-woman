from __future__ import annotations

import json
import re
from typing import Any

from .store import Message


def estimate_tokens(text: str) -> int:
    # Грубо: для русского ~3 символа на токен. Нужен только для бюджета.
    return max(1, len(text) // 3)


def scenario_block(
    scenario: dict[str, Any], mode: str = "roleplay"
) -> str:
    lines = ["## Сцена"]
    situation = str(scenario.get("situation") or "").strip()
    if situation:
        lines.append(situation)

    fields = [
        ("relationship", "Отношения"),
        ("her_goal", "Её цель"),
        ("tone", "Тон"),
        ("restrictions", "Ограничения"),
    ]
    for key, label in fields:
        value = str(scenario.get(key) or "").strip()
        if value:
            lines.append(f"- {label}: {value}")

    if mode == "messaging":
        lines.append(
            "Ты уже в этой ситуации. Просто переписывайся, не пересказывай "
            "ситуацию и не описывай сцену."
        )
    else:
        lines.append(
            "Ты уже внутри этой сцены. Не пересказывай ситуацию, не объясняй "
            "правила и не говори о сцене в третьем лице — просто действуй."
        )
    return "\n".join(lines)


def build_system_prompt(
    persona: str,
    facts: list[str],
    summary: str | None,
    language: str = "ru",
    scenario: dict[str, Any] | None = None,
    adult: bool = True,
    mode: str = "roleplay",
) -> str:
    parts = [persona.strip()]

    if scenario:
        parts.append(scenario_block(scenario, mode))

    memory: list[str] = []
    if facts:
        memory.append(
            "### Что ты знаешь о нём\n"
            + "\n".join(f"- {fact}" for fact in facts[:80])
        )
    if summary:
        memory.append("### Краткое содержание прошлых разговоров\n" + summary.strip())
    if memory:
        parts.append("## Память\n" + "\n\n".join(memory))

    if language == "ru":
        length_rule = (
            "- Пиши коротко: 1–3 строки, как в чате.\n"
            if mode == "messaging"
            else "- Держи длину реплики естественной: обычно 1–3 предложения.\n"
        )
        rules = (
            "## Правила\n"
            "- Отвечай только от лица персонажа и только по-русски.\n"
            "- Не говори, что ты ИИ, модель или программа.\n"
            + length_rule
            + "- Не повторяй дословно свои прошлые реплики.\n"
            "- У тебя есть свои желания, мнения и границы: можешь не соглашаться и спорить.\n"
        )
        if adult:
            rules += (
                "- Ты взрослая женщина. Близость и секс — только по взаимному согласию.\n"
                "- Всё только между взрослыми; принуждение и несовершеннолетние исключены.\n"
                "- Ты можешь говорить о желаниях и близости откровенно, если это уместно в сцене.\n"
                "- Не читай мораль, не упоминай правила и ограничения модели, не выходи из персонажа.\n"
                "- Если он нарушает твои границы, скажи об этом прямо.\n"
            )
        if mode == "messaging":
            rules += (
                "- Ты переписываешься в мессенджере, а не играешь сцену.\n"
                "- Не используй *действия*, описания движений, обстановки и третье лицо.\n"
                "- Можно отправлять несколько коротких сообщений подряд.\n"
                "- Можно использовать ASCII-смайлики: :), :(, :[ , ;), :D, ^^, :\\, :3. Не злоупотребляй.\n"
                "- Не превращай сообщение в сплошной текст: короткие фразы лучше.\n"
            )
        else:
            rules += (
                "- Действия и мимику оформляй звёздочками: *улыбается*, *отводит взгляд*.\n"
                "- Действия пиши в третьем лице, в настоящем времени, коротко: одно–два на реплику.\n"
                "- Действия должны быть бытовыми и уместными, без странной физиологии.\n"
                "- Не превращай реплику в сплошное описание действий и не описывай его действия.\n"
            )
        rules += (
            "- Не задавай вопрос в каждой реплике: иногда просто реагируй.\n"
            "- Не выдумывай факты о нём, которых нет в памяти или в разговоре."
        )
        parts.append(rules)

    return "\n\n".join(parts)


def trim_history(
    history: list[Message],
    max_chars: int,
    keep_last: int = 6,
) -> list[Message]:
    if not history:
        return []

    def total(items: list[Message]) -> int:
        return sum(len(item.content) for item in items)

    trimmed = list(history)
    while len(trimmed) > keep_last and total(trimmed) > max_chars:
        trimmed.pop(0)

    # Если даже последние сообщения не влезли — режем с начала,
    # но никогда не выбрасываем последнюю реплику пользователя.
    while len(trimmed) > 1 and total(trimmed) > max_chars:
        trimmed.pop(0)

    # История должна начинаться с реплики пользователя, иначе некоторые
    # chat-шаблоны начинают капризничать.
    while trimmed and trimmed[0].role != "user":
        trimmed.pop(0)
    return trimmed


def build_messages(
    persona: str,
    facts: list[str],
    summary: str | None,
    history: list[Message],
    max_chars: int,
    language: str = "ru",
    scenario: dict[str, Any] | None = None,
    adult: bool = True,
    mode: str = "roleplay",
) -> list[dict[str, str]]:
    system = build_system_prompt(
        persona, facts, summary, language, scenario, adult, mode
    )
    budget = max(1000, max_chars - len(system) - 200)

    usable = [m for m in history if m.role in ("user", "assistant")]
    usable = trim_history(usable, budget)

    messages = [{"role": "system", "content": system}]
    messages.extend({"role": m.role, "content": m.content} for m in usable)
    return messages


def case_suggestion_messages(
    persona: str,
    seed: str | None = None,
    existing: list[str] | None = None,
    adult: bool = True,
) -> list[dict[str, str]]:
    adult_block = ""
    sexual = False
    if adult:
        adult_block = (
            "\n\nВзрослые сцены разрешены, если идея пользователя про близость "
            "или секс. Пиши такую сцену прямо, без эвфемизмов, без замены на "
            "разговор о чувствах и без запрета на секс. Только взрослые, "
            "по взаимному согласию."
        )
        seed_text = (seed or "").lower()
        sexual = any(
            cue in seed_text
            for cue in (
                "секс",
                "близост",
                "интим",
                "трах",
                "оргазм",
                "минет",
                "кунни",
                "во время",
            )
        )
        if sexual:
            adult_block += (
                "\nЕсли идея про секс, сцена должна начинаться уже во время "
                "близости, а не с прелюдии или разговора до. Диалог и действия "
                "происходят внутри процесса. Не заменяй секс разговором о "
                "желании и не переноси его в «потом»."
            )

    system = (
        "Ты — драматург. Придумываешь одну сцену для ролевого разговора "
        "с персонажем.\n\n"
        "Персонаж:\n"
        f"{persona}\n\n"
        "Придумай конкретную сцену по идее пользователя: бытовую, "
        "эмоциональную или эротическую. В ней должен быть конфликт, "
        "неловкость, желание или недопонимание. Не устраивай катастрофу, "
        "болезнь или мелодраму. Сцена должна быть узнаваемой и оставлять "
        "место для нескольких ходов."
        f"{adult_block}\n\n"
        "Верни строго JSON без markdown и без пояснений:\n"
        "{\n"
        '  "situation": "1–2 предложения: что произошло и где",\n'
        '  "relationship": "кто вы друг другу и как давно",\n'
        '  "her_goal": "чего она хочет в этой сцене",\n'
        '  "tone": "тон сцены",\n'
        '  "restrictions": "чего не должно быть",\n'
        '  "opener": "her"\n'
        "}\n\n"
        "Пиши по-русски. Все поля, кроме opener, — короткие строки. "
        "opener всегда her. Не противоречь фактам из карточки персонажа. "
        "restrictions — только реальные ограничения сцены; не запрещай "
        "чувства, разговоры и метафоры без причины. Не добавляй запреты, "
        "которые противоречат идее пользователя."
    )
    parts: list[str] = []
    if seed:
        parts.append(
            f"Идея пользователя: {seed}. Развей её буквально, не смягчай "
            "и не заменяй другой сценой."
        )
        if sexual:
            parts.append(
                "Начни situation с момента, когда вы уже занимаетесь сексом. "
                "Не описывай прелюдию и разговор до. В restrictions не должно "
                "быть запрета на секс."
            )
    else:
        parts.append("Придумай сцену с нуля.")
    if existing:
        recent = "; ".join(item[:120] for item in existing[-15:])
        parts.append(f"Не повторяй эти сцены: {recent}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def parse_case_card(raw: str) -> dict[str, str] | None:
    text = raw.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None

    situation = str(data.get("situation") or data.get("scene") or "").strip()
    if len(situation) < 3:
        return None

    card: dict[str, str] = {"situation": situation[:500]}
    for key in ("relationship", "her_goal", "tone", "restrictions"):
        value = data.get(key)
        if value:
            cleaned = " ".join(str(value).split()).strip()
            if cleaned:
                card[key] = cleaned[:300]

    opener = str(data.get("opener") or "her").strip().lower()
    card["opener"] = "me" if opener in ("me", "я", "он", "user") else "her"
    return card


def short_variant_instruction() -> str:
    return (
        "## Режим заготовок\n"
        "Не пиши полный ответ. Дай одну короткую заготовку её реплики, "
        "3–8 слов.\n"
        "Без действий, без звёздочек и без описаний — только слова.\n"
        "Верни только заготовку, без кавычек, нумерации и пояснений."
    )


def expand_instruction(beat: str, mode: str = "roleplay") -> str:
    if mode == "messaging":
        return (
            "## Разворот заготовки\n"
            f"Выбранная заготовка её ответа: «{beat}»\n"
            "Разверни её в 1–2 коротких сообщения в мессенджере.\n"
            "Без *действий*, без описаний движений и третьего лица.\n"
            "Можно добавить один ASCII-смайлик: :) или :[.\n"
            "Не добавляй новых фактов и не пересказывай его слова.\n"
            "Верни только текст сообщений."
        )
    return (
        "## Разворот заготовки\n"
        f"Выбранная заготовка её ответа: «{beat}»\n"
        "Разверни её в живую реплику персонажа: 2–3 предложения.\n"
        "Добавь конкретную реакцию, деталь или действие, но не меняй смысл.\n"
        "Можно добавить одно–два коротких действия в *звёздочках*.\n"
        "Действия — бытовые и уместные, без странной физиологии.\n"
        "Не добавляй новых фактов и не пересказывай его слова.\n"
        "Верни только реплику."
    )


def parse_short_variants(raw: str, limit: int) -> list[str]:
    text = raw.strip()
    if not text:
        return []

    def clean(value: str) -> str:
        value = value.strip()
        value = re.sub(r"^\d+[\.\)]\s*", "", value)
        value = re.sub(r"^[-•]\s*", "", value)
        value = re.sub(r"^\*\s+", "", value)
        # Заготовки — только слова. Действия добавит разворот.
        value = re.sub(r"\*[^*]{1,80}\*", "", value)
        value = " ".join(value.split())
        return value.strip(" «»\"'.,;:-")

    lines = [clean(line) for line in text.splitlines()]
    lines = [line for line in lines if 2 <= len(line) <= 160]

    if len(lines) < 2 and ";" in text:
        lines = [clean(part) for part in text.split(";")]
        lines = [line for line in lines if 2 <= len(line) <= 160]

    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(line)
    return result[:limit]


def summarization_messages(
    transcript: str,
    previous_summary: str | None = None,
) -> list[dict[str, str]]:
    system = (
        "Ты — архивариус личного разговора. Ты обновляешь долговременную "
        "память, а не пишешь сводку с нуля.\n\n"
        "Верни строго JSON без пояснений и без markdown:\n"
        '{"summary": "...", "facts": ["...", "..."]}\n\n'
        "Требования к summary:\n"
        "- 4–8 предложений, от третьего лица.\n"
        "- Сохрани всё важное из предыдущей сводки: имена, события, "
        "договорённости, обещания, конфликты и изменения в отношениях.\n"
        "- Добавь новое из последних сообщений и убери только то, что "
        "действительно устарело или было опровергнуто.\n"
        "- Не пересказывай каждую реплику, выделяй суть.\n"
        "- Пиши по-русски.\n\n"
        "Требования к facts:\n"
        "- Только новые устойчивые факты: имя, возраст, работа, вкусы, "
        "привычки, планы, границы, важные события.\n"
        "- Не повторяй факты, которые уже есть в предыдущей сводке.\n"
        "- Максимум 8 фактов, каждый — одно короткое предложение."
    )
    parts: list[str] = []
    if previous_summary:
        parts.append("Предыдущая сводка:\n" + previous_summary.strip())
        parts.append("Новые сообщения:\n" + transcript)
        parts.append(
            "Обнови сводку: сохрани всё важное из предыдущей и добавь новое. "
            "Не теряй старые сведения, если они не устарели."
        )
    else:
        parts.append(transcript)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_summary(raw: str) -> tuple[str, list[str]]:
    raw = raw.strip()
    if not raw:
        return "", []

    # Убираем ```json ... ``` и ищем первый JSON-объект.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            data: Any = json.loads(candidate)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            summary = str(data.get("summary") or "").strip()
            facts_raw = data.get("facts") or []
            if isinstance(facts_raw, str):
                facts_raw = [facts_raw]
            facts = [
                " ".join(str(fact).split()).strip(" -•\t")
                for fact in facts_raw
                if str(fact).strip()
            ]
            facts = [fact for fact in facts if 3 <= len(fact) <= 300][:8]
            if summary:
                return summary, facts

    # Модель не дала JSON — считаем весь ответ сводкой, фактов нет.
    return raw, []
