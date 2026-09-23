from dataclasses import dataclass, field
from typing import Literal

Side = Literal["BEFORE", "AFTER"]


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    section: str | None
    locator: str
    text: str


@dataclass
class Document:
    id: str
    side: Side
    filename: str
    format: str
    chunks: list[SourceChunk] = field(default_factory=list)
