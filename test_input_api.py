#!/usr/bin/env python3
"""Utility script for probing Kramerius API responses via AltoProcessor."""

import argparse
from typing import List, Optional

from main_processor import AltoProcessor


def build_processor(api_bases: Optional[List[str]]) -> AltoProcessor:
    if not api_bases:
        return AltoProcessor()
    primary = api_bases[0]
    fallback = api_bases[1:] if len(api_bases) > 1 else None
    return AltoProcessor(api_base_url=primary, fallback_api_bases=fallback)


def run_probe(uuid: str, api_bases: Optional[List[str]]) -> None:
    processor = build_processor(api_bases)
    print(f"Testing full fetch for UUID: {uuid}")

    context = processor.get_book_context(uuid)
    if not context:
        print("❌ Nepodařilo se načíst kontext dokumentu.")
        return

    book_uuid = context.get("book_uuid")
    page_uuid = context.get("page_uuid")
    pages = context.get("pages") or []
    mods = context.get("mods") or []

    print(f"Book UUID: {book_uuid}")
    print(f"Page UUID: {page_uuid}")
    print(f"Počet nalezených stran: {len(pages)}")
    print(f"Počet MODS položek: {len(mods)}")

    top_children = processor.get_children(book_uuid or "")
    print(f"Počet children uzlů (1. úroveň): {len(top_children)}")

    alto_xml = processor.get_alto_data(page_uuid or "")
    if not alto_xml:
        print("❌ ALTO data se nepodařilo načíst.")
        return

    print(f"Délka ALTO XML: {len(alto_xml)} znaků")

    formatted = processor.get_formatted_text(alto_xml, page_uuid or "", 800, 1200)
    print(f"Délka formátovaného výstupu: {len(formatted)} znaků")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Kramerius API proti více základnám.")
    parser.add_argument(
        "uuid",
        nargs="?",
        default="89c55de0-79cc-11e4-964c-5ef3fc9bb22f",
        help="UUID stránky nebo knihy, se kterou se má testovat (výchozí MZK příklad).",
    )
    parser.add_argument(
        "--bases",
        nargs="*",
        help="Volitelný seznam API základen v prioritním pořadí (např. https://host/api/v5.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe(args.uuid, args.bases)
