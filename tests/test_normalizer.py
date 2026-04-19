"""Header/footer dedup threshold behavior."""
import pytest
from src.models.document import Item
from src.pipeline.normalizer import normalize_items


def _text(text: str, page: int) -> Item:
    return Item(kind="text", text=text, page_range=(page, page), token_count=len(text.split()))


def test_running_header_removed_above_threshold():
    header = "My Book Title"
    items = [_text(f"{header}\nContent on page {i}.", i) for i in range(1, 6)]
    result = normalize_items(items)
    for item in result:
        assert header not in item.text


def test_occasional_line_not_removed():
    items = [_text(f"Content on page {i}.", i) for i in range(1, 10)]
    # Only 2 out of 9 pages have "Appendix" as first line — below 60%
    items[0] = _text("Appendix\nContent on page 1.", 1)
    items[3] = _text("Appendix\nContent on page 4.", 4)
    result = normalize_items(items)
    found = any("Appendix" in item.text for item in result)
    assert found, "Occasional repeated line should not be stripped"


def test_atomic_items_pass_through_unchanged():
    table = Item(kind="table", text="| col |\n| val |", page_range=(1, 1), token_count=5)
    result = normalize_items([table])
    assert result == [table]


def test_empty_items_handled():
    assert normalize_items([]) == []
