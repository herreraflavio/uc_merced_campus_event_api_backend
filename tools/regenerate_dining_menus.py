#!/usr/bin/env python3
"""Manually regenerate dining menus and update the content feed source."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EVENTS_PY_PATH = ROOT_DIR / "routes" / "events.py"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(ROOT_DIR / ".env")


def _load_events_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    spec = importlib.util.spec_from_file_location(
        "dining_menu_events_module",
        EVENTS_PY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVENTS_PY_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Pavilion and Yablokoff-Wallace Dining Center menus, "
            "write them into routes/polygons.json, and invalidate the "
            "/contentAPIURL in-process cache for this run."
        )
    )
    parser.add_argument(
        "--refresh-presence-cache",
        action="store_true",
        help="Also refresh the Presence event pages cache after menu regeneration.",
    )
    args = parser.parse_args(argv)

    _load_dotenv()
    events_module = _load_events_module()

    matched = events_module.generate_food_menus_and_update_polygons()
    print(
        "[manual] dining menus regenerated: "
        f"Pavilion={matched['774']}, "
        f"Yablokoff-Wallace Dining Center={matched['1130']}",
        flush=True,
    )

    if args.refresh_presence_cache:
        event_count = events_module.refresh_content_api_presence_cache()
        print(
            f"[manual] Presence pages cache refreshed: {event_count} event(s)",
            flush=True,
        )

    print("[manual] /contentAPIURL will read the updated polygons.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
