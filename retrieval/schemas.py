"""CLI、长驻进程和未来 MCP 共同使用的请求与门控 JSON 契约。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=262144)
    candidate_k: int = Field(default=200, ge=1, le=200)
    top_k: int = Field(default=5, ge=1, le=30)
    rerank: bool = True
    quality_gate: bool = True
    context_budget_tokens: int = Field(default=2048, ge=128, le=32768)
    doc_ids: list[str] | None = Field(default=None, max_length=100)
    term_scopes: list[str] | None = Field(default=None, max_length=20,
        description="可选的缩写适用领域，如 reinforcement_learning；不能单独消除同领域多义")

    @model_validator(mode="after")
    def validate_options(self):
        self.query = self.query.strip()
        if not self.query or self.candidate_k < self.top_k:
            raise ValueError("问题不能为空，candidate_k 必须不小于 top_k")
        if self.doc_ids is not None and (not self.doc_ids or any(not x.strip() for x in self.doc_ids)):
            raise ValueError("doc_ids 如提供必须包含非空文档 ID")
        if self.term_scopes is not None and (not self.term_scopes or any(not x.strip() or len(x)>100 for x in self.term_scopes)):
            raise ValueError("term_scopes 应为非空领域名称列表，每项不超过 100 字符")
        return self


class ChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    doc_id: str = Field(min_length=1, max_length=200)
    chunk_id: str = Field(min_length=1, max_length=200)


class GateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GatePolicy(GateModel):
    version: str = Field(default="1", min_length=1)
    batch_size: int = Field(default=16, ge=1, le=30)
    max_candidates: int = Field(default=30, ge=1, le=200)
    max_rounds: int = Field(default=4, ge=1, le=20)
    max_api_attempts: int = Field(default=6, ge=1, le=40)
    retries_per_round: int = Field(default=1, ge=0, le=3)
    total_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    retry_backoff_seconds: float = Field(default=0.5, ge=0, le=10)
    max_request_bytes: int = Field(default=100000, ge=1024)
    max_total_request_bytes: int = Field(default=300000, ge=1024)
    max_total_output_tokens: int = Field(default=24576, ge=1)
    allow_partial_answers: bool = True


class GateSubquestion(GateModel):
    subquestion_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)


class SupportSpan(GateModel):
    # 类型严格校验；范围在本地逐字对齐时校验，以便修正可唯一定位的错误偏移。
    start: int
    end: int
    quote: str = Field(min_length=1, max_length=4000)


class GateAssessment(GateModel):
    evidence_id: str = Field(min_length=1)
    relation: Literal["direct_support", "partial_support", "background", "contradicts", "irrelevant", "uncertain"]
    applicability: Literal["supported", "missing", "not_applicable", "uncertain"]
    time_scope: Literal["supported", "missing", "not_applicable", "uncertain"]
    conditions: Literal["supported", "missing", "not_applicable", "uncertain"]
    support_spans: list[SupportSpan] = Field(max_length=20)
    reason: str = Field(min_length=1, max_length=2000)


class GateCoverage(GateModel):
    subquestion_id: str
    # 每个内层列表是联合充分的证据组；外层列表列出可替代的证据组。
    support_sets: list[list[str]] = Field(max_length=30)
    partial_evidence_ids: list[str]
    missing: list[str] = Field(max_length=30)


class GateConflict(GateModel):
    subquestion_id: str
    evidence_ids: list[str] = Field(min_length=2)
    explanation: str = Field(min_length=1, max_length=2000)


class GatePayload(GateModel):
    subquestions: list[GateSubquestion] = Field(min_length=1, max_length=20)
    assessments: list[GateAssessment]
    coverage: list[GateCoverage]
    conflicts: list[GateConflict]
