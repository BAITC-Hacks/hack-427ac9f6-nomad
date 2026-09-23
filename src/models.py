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

# H3 schemas are shared by Structured Outputs and the evidence gate.
from pydantic import BaseModel, ConfigDict, Field


class ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnitChange(ComparisonModel):
    before_unit: str | None
    after_unit: str | None
    status: Literal["PRESERVED", "CREATED", "DELETED", "TRANSFORMED"]
    reason: str
    confidence: float = Field(ge=0, le=1)
    before_evidence_ids: list[str]
    after_evidence_ids: list[str]


class ResponsibilityChange(ComparisonModel):
    before_unit: str
    after_unit: str | None
    before_text: str
    after_text: str | None
    status: Literal["SAME", "MODIFIED", "MOVED", "POTENTIALLY_LOST"]
    reason: str
    confidence: float = Field(ge=0, le=1)
    before_evidence_ids: list[str]
    after_evidence_ids: list[str]


class UnitEvidence(ComparisonModel):
    unit_id: str
    evidence_ids: list[str]


class Finding(ComparisonModel):
    type: Literal["FUNCTION_LOSS", "DUPLICATION", "CONFLICT_OF_INTEREST"]
    title: str
    description: str
    affected_unit_ids: list[str]
    reason: str
    confidence: float = Field(ge=0, le=1)
    before_evidence_ids: list[str]
    after_evidence_ids: list[str]
    unit_evidence: list[UnitEvidence]
    recommendation: str | None


@dataclass
class ComparisonResult:
    unit_changes: list[UnitChange] = field(default_factory=list)
    responsibility_changes: list[ResponsibilityChange] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    discarded_items: int = 0
    before_chunks_used: int = 0
    after_chunks_used: int = 0
