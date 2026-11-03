"""在线检索配置；路径相对工作区解析，API 密钥仅从环境变量读取。"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from rag_core.config import resource_path

INSTRUCTION = "Given a scientific question, retrieve passages that answer the question."


@dataclass
class RetrievalConfig:
    root: Path
    index_dir: str = "RAG/library"
    index_kind: str = "library"
    embedding_dir: str = "RAG/models/Qwen3-Embedding-0.6B"
    reranker_dir: str = "RAG/models/Qwen3-Reranker-0.6B"
    models_dir: str = "RAG/models"
    cache_dir: str = "RAG/model_cache"
    instruction: str = INSTRUCTION
    npu_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    gpu_max_tokens: int = 32768
    reranker_max_tokens: int = 32768
    reranker_batch_size: int = 2
    max_chunks_per_document: int = 8
    keyword_enabled: bool = True
    rrf_k: int = 60
    rerank_candidate_k: int = 40
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    term_enrichment_enabled: bool = True
    terminology_pointer: str = "RAG/terminology/auto/current.json"
    term_scope_cues_file: str = "config/term_scope_cues.json"
    term_weight: float = 0.15
    term_max_keywords: int = 12
    term_max_expansions: int = 8
    gate_policy_file: str = "config/quality_gate_policy.json"
    gate_rubric_file: str = "config/quality_gate_rubric.md"
    deepseek: dict = field(default_factory=lambda: {
        "base_url": "https://api.deepseek.com", "model": "deepseek-flash",
        "api_key_env": "DEEPSEEK_API_KEY", "timeout_seconds": 60, "max_tokens": 4096})

    def __post_init__(self):
        import math
        if type(self.term_enrichment_enabled) is not bool:
            raise ValueError("term_enrichment_enabled 必须是布尔值")
        if type(self.term_weight) not in (int,float) or not math.isfinite(self.term_weight) or not 0 <= self.term_weight <= .5:
            raise ValueError("term_weight 必须在 0～0.5 之间")
        if type(self.term_max_keywords) is not int or not 1 <= self.term_max_keywords <= 50 or type(self.term_max_expansions) is not int or not 1 <= self.term_max_expansions <= 20:
            raise ValueError("词项/扩展数量超出预算")
        if type(self.max_chunks_per_document) is not int or not 1 <= self.max_chunks_per_document <= 200:
            raise ValueError("max_chunks_per_document 必须是 1～200 的整数")
        if type(self.rrf_k) is not int or not 1 <= self.rrf_k <= 1000:
            raise ValueError("rrf_k 必须是 1～1000 的整数")
        if type(self.rerank_candidate_k) is not int or not 1 <= self.rerank_candidate_k <= 400:
            raise ValueError("rerank_candidate_k 必须是 1～400 的整数")
        if type(self.keyword_enabled) is not bool or type(self.mmr_enabled) is not bool:
            raise ValueError("keyword_enabled/mmr_enabled 必须为布尔值")
        import math
        if type(self.mmr_lambda) not in (int, float) or not math.isfinite(self.mmr_lambda) or not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda 必须是 0～1 的有限数字")

    @classmethod
    def load(cls, path: Path, root: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["root"] = root.resolve()
        if "npu_lengths" in data:
            data["npu_lengths"] = tuple(data["npu_lengths"])
        config = cls(**data)
        if config.npu_lengths != (128, 256, 512, 1024):
            raise ValueError("当前策略固定为 128/256/512/1024 四档")
        if not 1024 < config.gpu_max_tokens <= 32768 or not 1 <= config.reranker_max_tokens <= 32768:
            raise ValueError("embedding/reranker 上限超出模型范围")
        if not 1 <= config.reranker_batch_size <= 8:
            raise ValueError("重排 batch 应为 1～8")
        return config

    def path(self, name):
        return resource_path(self.root, getattr(self, name))
