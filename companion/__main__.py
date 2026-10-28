from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="companion",
        description="Локальный персонаж: чат, память и ручной отбор ответов.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="путь к config.toml (по умолчанию ищется в текущей папке)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="заглушка вместо модели: проверить интерфейс без Ollama",
    )
    parser.add_argument("--model", help="переопределить модель на этот запуск")
    parser.add_argument("--session", type=int, help="открыть конкретную сессию")
    parser.add_argument(
        "--variants",
        type=int,
        help="переопределить число вариантов (0 — обычный чат)",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="не использовать полноэкранный интерфейс (старый линейный чат)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        cfg.llm.backend = "mock"
        cfg.llm.model = "mock"
    if args.model:
        cfg.llm.model = args.model
    if args.variants is not None:
        cfg.chat.variants = args.variants

    try:
        if sys.stdin.isatty() and not args.no_tui:
            from .tui import run_tui

            run_tui(cfg, session_id=args.session)
        else:
            from .cli import App

            App(cfg, session_id=args.session).run()
    except KeyboardInterrupt:
        print()
    except EOFError:
        print()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
