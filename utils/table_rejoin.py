#!/usr/bin/env python3
"""Merge Docling TableItems that appear to be fragments of the same multi-page table.

Heuristic (conservative, designed to avoid false merges):
  Two consecutive TableItems A, B are merged when ALL of:
    - A.last_page + 1 == B.first_page           (adjacent pages)
    - A.num_cols == B.num_cols                   (same column count)
    - B's first row matches A's header row      OR  B has no header row
  The merged table's rows = A.rows + (B.rows minus duplicated header if matched)

Returns a new list of logical tables as dicts:
  {
    "num_rows": int,
    "num_cols": int,
    "pages": [int, ...],
    "rows": [[cell, ...], ...],     # text-only, for downstream joining
    "source_table_indices": [int, ...],
  }
"""

from __future__ import annotations

from typing import Any, Dict, List


def _first_page(prov: list) -> int:
    return min((p.page_no for p in prov), default=0)


def _last_page(prov: list) -> int:
    return max((p.page_no for p in prov), default=0)


def _rows_as_text(table_data) -> List[List[str]]:
    """Convert Docling TableData into a list-of-rows of plain text cells."""
    if table_data is None or not hasattr(table_data, "table_cells"):
        return []
    # Build a dense grid from cell start/end row/col information.
    grid: Dict[tuple, str] = {}
    max_row = -1
    max_col = -1
    for cell in table_data.table_cells:
        r = getattr(cell, "start_row_offset_idx", getattr(cell, "row_index", 0))
        c = getattr(cell, "start_col_offset_idx", getattr(cell, "col_index", 0))
        text = getattr(cell, "text", "") or ""
        grid[(r, c)] = text.strip()
        max_row = max(max_row, r)
        max_col = max(max_col, c)
    rows = []
    for r in range(max_row + 1):
        rows.append([grid.get((r, c), "") for c in range(max_col + 1)])
    return rows


def rejoin_tables(doc_tables: List[Any]) -> List[Dict[str, Any]]:
    """Merge adjacent same-shape tables across page boundaries."""
    logical: List[Dict[str, Any]] = []
    for i, t in enumerate(doc_tables):
        prov = getattr(t, "prov", []) or []
        data = getattr(t, "data", None)
        rows = _rows_as_text(data)
        num_cols = data.num_cols if data else (len(rows[0]) if rows else 0)
        first_p = _first_page(prov)
        last_p = _last_page(prov)
        entry = {
            "num_rows": len(rows),
            "num_cols": num_cols,
            "pages": sorted({p.page_no for p in prov}),
            "rows": rows,
            "source_table_indices": [i],
        }
        if logical:
            prev = logical[-1]
            prev_last_page = max(prev["pages"]) if prev["pages"] else -1
            # Merge condition
            if (
                first_p == prev_last_page + 1
                and prev["num_cols"] == num_cols
                and num_cols > 0
            ):
                # Detect header duplication on the second fragment
                start_row = 0
                if prev["rows"] and rows and rows[0] == prev["rows"][0]:
                    start_row = 1
                prev["rows"].extend(rows[start_row:])
                prev["num_rows"] = len(prev["rows"])
                prev["pages"] = sorted(set(prev["pages"]) | {p.page_no for p in prov})
                prev["source_table_indices"].append(i)
                continue
        logical.append(entry)
    return logical
