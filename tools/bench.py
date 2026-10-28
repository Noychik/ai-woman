#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from companion.config import load_config
from companion.llm import LLMError, make_llm
from companion.prompt import build_messages
from companion.store import Message

DEFAULT_MODELS = [
    "huihui_ai/qwen2.5-abliterate:7b",
    "gemma3:12b",
    "mistral-nemo:12b",
    "huihui_ai/qwen3-abliterated:8b-q4_K_M",
    "qwen3:4b-instruct-2507-q4_K_M",
]

REFUSAL_MARKERS = (
    "не могу",
    "не буду",
    "извини",
    "как ии",
    "как языковая модель",
    "не имею права",
    "не уместно",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data = json.loads(line)
        text = str(data.get("text") or "").strip()
        if not text:
            continue
        prompts.append(
            {
                "id": str(data.get("id") or f"p{len(prompts) + 1}"),
                "text": text,
            }
        )
    return prompts


def preview(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сравнить модели на фиксированном наборе промптов."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument(
        "--prompts", type=Path, default=ROOT / "bench" / "prompts_sfw.jsonl"
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Список моделей через запятую.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "bench")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", type=int, default=160)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--adult", action="store_true", default=None)
    mode.add_argument("--sfw", action="store_true", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    adult = cfg.content.adult
    if args.adult:
        adult = True
    if args.sfw:
        adult = False

    persona = cfg.path(cfg.paths.persona).read_text(encoding="utf-8")
    persona_hash = hashlib.sha256(persona.encode("utf-8")).hexdigest()[:12]
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        print("Нет промптов.")
        return 1

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        print("Нет моделей.")
        return 1

    probe = make_llm(cfg.llm)
    try:
        probe.health()
    except LLMError as exc:
        print(f"Бэкенд недоступен: {exc}")
        return 1
    installed = set(probe.list_models())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.out / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for model in models:
        if installed and model not in installed:
            print(f"[skip] {model}: не установлена (ollama pull {model})")
            summary.append({"model": model, "status": "missing"})
            continue

        # Qwen3 8B abliterated — thinking-модель, остальным /no_think не нужен.
        no_think = cfg.llm.no_think and "qwen3-abliterated:8b" in model
        model_cfg = replace(
            cfg.llm,
            model=model,
            no_think=no_think,
            keep_alive=-1,
        )
        if args.num_ctx:
            model_cfg = replace(model_cfg, num_ctx=args.num_ctx)
        llm = make_llm(model_cfg)
        model_dir = outdir / safe_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {model} ===")
        for prompt in prompts:
            history = [
                Message(
                    id=1,
                    session_id=1,
                    role="user",
                    content=prompt["text"],
                    created_at=datetime.now().isoformat(timespec="seconds"),
                )
            ]
            messages = build_messages(
                persona=persona,
                facts=[],
                summary=None,
                history=history,
                max_chars=cfg.chat.max_context_chars,
                language=cfg.chat.language,
                adult=adult,
            )
            started = time.perf_counter()
            text = ""
            error = None
            try:
                text = llm.complete(
                    messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            elapsed = time.perf_counter() - started
            stats = llm.last_stats.as_dict() if llm.last_stats else None
            refusal = any(marker in text.lower() for marker in REFUSAL_MARKERS)
            record = {
                "model": model,
                "prompt_id": prompt["id"],
                "prompt": prompt["text"],
                "output": text,
                "error": error,
                "wall_seconds": round(elapsed, 3),
                "stats": stats,
                "refusal": refusal,
                "adult": adult,
                "persona_hash": persona_hash,
                "temperature": (
                    args.temperature
                    if args.temperature is not None
                    else cfg.llm.temperature
                ),
                "max_tokens": (
                    args.max_tokens
                    if args.max_tokens is not None
                    else cfg.llm.max_tokens
                ),
                "num_ctx": model_cfg.num_ctx,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            path = model_dir / f"{safe_name(prompt['id'])}.json"
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tps = stats.get("tokens_per_second") if stats else None
            print(
                f"  {prompt['id']}: "
                f"{tps:.1f} ток/с" if tps else f"  {prompt['id']}: ошибка",
                end="",
            )
            print(
                f" · {elapsed:.1f}s"
                + (" · отказ" if refusal else "")
                + (f" · {error[:80]}" if error else "")
            )
            summary.append(
                {
                    "model": model,
                    "status": "ok" if not error else "error",
                    "prompt_id": prompt["id"],
                    "prompt": prompt["text"],
                    "tokens_per_second": tps,
                    "wall_seconds": round(elapsed, 3),
                    "prompt_seconds": (
                        stats.get("prompt_seconds") if stats else None
                    ),
                    "tokens": stats.get("tokens") if stats else None,
                    "refusal": refusal,
                    "error": error,
                    "preview": preview(text, args.preview),
                }
            )

        if cfg.llm.backend == "ollama":
            try:
                llm.unload_model()
            except LLMError:
                pass

    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "adult": adult,
                "persona_hash": persona_hash,
                "config": str(args.config),
                "prompts": str(args.prompts),
                "results": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = [
        "| модель | промпт | ток/с | стена | промпт, с | отказ |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in summary:
        if item.get("status") == "missing":
            rows.append(f"| `{item['model']}` | — | — | — | — | не установлена |")
            continue
        tps = item.get("tokens_per_second")
        rows.append(
            f"| `{item['model']}` | {item['prompt_id']} | "
            f"{tps:.1f} | {item['wall_seconds']:.1f} | "
            f"{item.get('prompt_seconds') or 0:.1f} | "
            f"{'да' if item.get('refusal') else 'нет'} |"
            if tps is not None
            else f"| `{item['model']}` | {item['prompt_id']} | — | "
            f"{item['wall_seconds']:.1f} | — | ошибка |"
        )
    (outdir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nРезультаты: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
