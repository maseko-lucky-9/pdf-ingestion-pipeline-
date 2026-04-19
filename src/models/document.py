from __future__ import annotations
import hashlib
from typing import Literal
from pydantic import BaseModel, Field, computed_field


class Item(BaseModel):
    kind: Literal["text", "table", "formula", "code"]
    text: str
    page_range: tuple[int, int]
    token_count: int


class Chunk(BaseModel):
    docid: str
    source_pdf: str
    chunk_index: int
    chunk_type: Literal["text", "table", "formula", "code"]
    content: str
    page_range: tuple[int, int]
    token_count: int
    collection: str
    domain: str = ""
    book: str = ""

    @computed_field
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]


class Collection(BaseModel):
    name: str
    db_path: str
    domain: str = ""
