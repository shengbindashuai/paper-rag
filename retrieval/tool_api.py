"""MCP 与 JSONL 共用的工具定义、严格校验和调用分发。"""
from .schemas import SearchRequest, ChunkRequest
from knowledge.schemas import (ReadRequest, PaperIngestRequest, DocumentIngestRequest,
                               ConversationRequest, JobRequest)


def definitions():
    return [
        {"name": "paper_search", "description": "检索本地论文，返回原文证据及来源；默认启用 DeepSeek 质量门控，可用 quality_gate=false 关闭",
         "inputSchema": SearchRequest.model_json_schema()},
        {"name": "paper_get_chunk", "description": "按文档和片段 ID 读取可追溯原文",
         "inputSchema": ChunkRequest.model_json_schema()},
        {"name": "paper_retrieval_status", "description": "查看索引修订和常驻模型状态",
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "paper_read", "description": "分页读取文档已索引原文片段，可按章节或 PDF 页码过滤；保留来源，非完整 PDF 排版；不授予门控回答许可",
         "inputSchema": ReadRequest.model_json_schema()},
        {"name": "paper_ingest", "description": "提交本地 PDF 增量入库，返回 job_id；应查询任务直到 completed 才报告成功",
         "inputSchema": PaperIngestRequest.model_json_schema()},
        {"name": "document_ingest", "description": "提交 Markdown/TXT/DOCX 文件或正文入库，保留文档来源；返回 job_id",
         "inputSchema": DocumentIngestRequest.model_json_schema()},
        {"name": "conversation_save", "description": "将调用方提供的对话消息按角色与顺序归档并入库；不自动获取聊天历史，返回 job_id",
         "inputSchema": ConversationRequest.model_json_schema()},
        {"name": "ingest_job_status", "description": "查询持久化入库任务；completed 表示文档可检索且索引快照已发布",
         "inputSchema": JobRequest.model_json_schema()},
    ]


class RetrievalToolAPI:
    def __init__(self, service, jobs=None):
        self.service = service
        self.jobs = jobs

    def call(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("arguments 必须为对象")
        if name == "paper_search":
            return self.service.search(SearchRequest.model_validate(arguments))
        if name == "paper_get_chunk":
            request = ChunkRequest.model_validate(arguments)
            return self.service.get_chunk(request.doc_id, request.chunk_id)
        if name == "paper_retrieval_status":
            if arguments:
                raise ValueError("status 不接受参数")
            result = self.service.status()
            if self.jobs is not None:
                import json
                path = self.jobs.index_dir / 'automation_status.json'
                if path.is_file():
                    result['automation'] = json.loads(path.read_text(encoding='utf-8'))
                pointer = self.jobs.root / 'RAG/terminology/auto/verification_status.json'
                if pointer.is_file():
                    result['terminology_verification'] = json.loads(pointer.read_text(encoding='utf-8'))
                ledger_path = self.jobs.root / 'RAG/library/api_accounting.sqlite3'
                if ledger_path.is_file():
                    from rag_core.api_accounting import APILedger
                    result['api_usage'] = APILedger(ledger_path).summary()
            return result
        if name == "paper_read":
            return self.service.read_document(ReadRequest.model_validate(arguments))
        if name in {"paper_ingest", "document_ingest", "conversation_save", "ingest_job_status"}:
            if self.jobs is None:
                raise RuntimeError("当前入口未启用入库队列，请使用 MCP 服务或启用写入的 JSONL serve")
            if name == "ingest_job_status":
                return self.jobs.status(JobRequest.model_validate(arguments).job_id)
            kind, schema = {"paper_ingest": ("paper", PaperIngestRequest),
                            "document_ingest": ("document", DocumentIngestRequest),
                            "conversation_save": ("conversation", ConversationRequest)}[name]
            return self.jobs.submit(kind, schema.model_validate(arguments))
        raise ValueError(f"未知检索工具：{name}")
