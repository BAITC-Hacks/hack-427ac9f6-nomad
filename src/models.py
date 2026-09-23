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


@dataclass
class OrgUnit:
    id: str
    name: str
    parent: str | None  # Source parent name, not an inferred relationship.
    source_refs: list[str]

    aliases: list[str] = field(default_factory=list)

    @property
    def canonical_name(self) -> str:
        return self.name


@dataclass
class Function:
    id: str
    unit_id: str
    text: str
    source_refs: list[str]


@dataclass
class ExtractionResult:
    document_id: str
    units: list[OrgUnit] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    discarded_references: int = 0
    discarded_items: int = 0
