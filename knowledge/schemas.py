"""MCP 和 JSONL 共用的严格接口契约。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReadRequest(Request):
    doc_id: str = Field(min_length=1, max_length=200)
    section: str | None = Field(default=None, max_length=500)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=5, ge=1, le=30)
    max_chars: int = Field(default=24000, ge=1024, le=100000)

    @model_validator(mode="after")
    def page_range(self):
        if self.page_start and self.page_end and self.page_start > self.page_end:
            raise ValueError("page_start 不能大于 page_end")
        return self


class PaperIngestRequest(Request):
    path: str = Field(min_length=1, max_length=4096)
    title: str | None = Field(default=None, min_length=1, max_length=300)


class DocumentIngestRequest(Request):
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    content: str | None = Field(default=None, min_length=1, max_length=1000000)
    title: str = Field(min_length=1, max_length=300)
    source_url: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def one_source(self):
        if (self.path is None) == (self.content is None):
            raise ValueError("path 与 content 必须且只能提供一个")
        if self.content is not None and not self.content.strip():
            raise ValueError("正文不能为空")
        return self


class Message(Request):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(min_length=1, max_length=200000)
    timestamp: str | None = Field(default=None, max_length=100)


class ConversationRequest(Request):
    title: str = Field(min_length=1, max_length=300)
    messages: list[Message] = Field(min_length=1, max_length=500)
    conversation_id: str | None = Field(default=None, max_length=200)
    source_url: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def total_size(self):
        if sum(len(m.content) for m in self.messages) > 1000000:
            raise ValueError("对话正文总长度不能超过 1000000 字符")
        return self


class JobRequest(Request):
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
