"""Normalize a typed Item stream: strip running headers/footers."""
from __future__ import annotations
from collections import Counter

from src.models.document import Item

_THRESHOLD = 0.6  # fraction of pages a line must appear on to be flagged


def normalize_items(items: list[Item]) -> list[Item]:
    """Remove running headers and footers from a page-ordered item stream.

    Algorithm (R8):
    1. Group TextItems by page.
    2. Take the first and last text line of each page.
    3. If the same line appears on ≥ 60% of pages at the same position
       (first vs last), mark it as a running header/footer.
    4. Drop matched lines from all items before returning.
    """
    if not items:
        return items

    pages: dict[int, list[Item]] = {}
    for item in items:
        page = item.page_range[0]
        pages.setdefault(page, []).append(item)

    total_pages = len(pages)
    if total_pages == 0:
        return items

    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()

    for page_items in pages.values():
        text_items = [it for it in page_items if it.kind == "text"]
        if not text_items:
            continue
        first_line = text_items[0].text.split("\n")[0].strip()
        last_line = text_items[-1].text.split("\n")[-1].strip()
        if first_line:
            first_lines[first_line] += 1
        if last_line:
            last_lines[last_line] += 1

    banned: set[str] = set()
    for line, count in first_lines.items():
        if count / total_pages >= _THRESHOLD:
            banned.add(line)
    for line, count in last_lines.items():
        if count / total_pages >= _THRESHOLD:
            banned.add(line)

    if not banned:
        return items

    cleaned: list[Item] = []
    for item in items:
        if item.kind != "text":
            cleaned.append(item)
            continue

        lines = item.text.split("\n")
        filtered = [ln for ln in lines if ln.strip() not in banned]
        new_text = "\n".join(filtered).strip()
        if new_text:
            cleaned.append(item.model_copy(update={"text": new_text}))

    return cleaned
