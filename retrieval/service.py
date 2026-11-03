"""查询编码→召回→重排→可选门控→上下文；不生成回答，长驻实例复用模型。"""
from threading import RLock
import time
import uuid

from .embedding_router import QueryEmbeddingRouter
from .index import SnapshotIndex, ExperimentIndex
from .quality_gate import DeepSeekQualityGate
from .gate_runner import QualityGateRunner
from .reranker import QwenReranker
from .schemas import GatePolicy, SearchRequest
from .hybrid import mmr_reorder, identity
from .term_enrichment import TermEnrichment


def limit_document_chunks(hits, maximum):
    """按召回顺序保留每个 doc_id 的前 maximum 块，不用低排名块突破限额补齐。"""
    counts, selected = {}, []
    for hit in hits:
        doc_id = hit.get("linked_doc_id") or hit["doc_id"]
        if counts.get(doc_id, 0) < maximum:
            selected.append(hit)
            counts[doc_id] = counts.get(doc_id, 0) + 1
    return selected, {"max_chunks_per_document": maximum, "selected_count": len(selected),
                      "dropped_count": len(hits) - len(selected),
                      "input_document_count": len({h.get("linked_doc_id") or h["doc_id"] for h in hits}),
                      "selected_document_count": len(counts), "per_document_counts": counts}


def select_context(hits, tokenizer, budget):
    """保留完整块；计入引用标签，预算不含未来回答模型的 system/query。"""
    context, used = [], 0
    for hit in hits:
        piece = f"[{hit['evidence_id']}] {hit.get('title', '')}\n{hit.get('section', '')}\n{hit['text']}"
        combined = "\n\n".join([c["context_text"] for c in context] + [piece])
        cost = len(tokenizer(combined)["input_ids"])
        if cost > budget:
            continue
        context.append({**hit, "context_text": piece})
        used = cost
    return context, used


class RetrievalService:
    def __init__(self, config, *, router=None, index=None, reranker=None, gate=None):
        self.config, self.lock = config, RLock()
        # 先验证本地文件，避免配置错误后遗留已打开的索引或设备资源。
        self.gate_policy = GatePolicy.model_validate_json(config.path("gate_policy_file").read_text(encoding="utf-8"))
        rubric = config.path("gate_rubric_file").read_text(encoding="utf-8")
        self.gate = gate or DeepSeekQualityGate(config.deepseek, rubric=rubric,
            cache_path=config.root / 'RAG/library/api_accounting.sqlite3',
            tokenizer_file=config.path('embedding_dir') / 'tokenizer.json')
        self.gate_runner = QualityGateRunner(self.gate, self.gate_policy)
        if index is None:
            if config.index_kind == "library":
                index = SnapshotIndex(config.path("index_dir"), root=config.root)
            elif config.index_kind == "experiment":
                index = ExperimentIndex(config.path("index_dir"))
            else:
                raise ValueError("index_kind 必须是 library 或显式测试用的 experiment")
        self.index, self.router = index, router or QueryEmbeddingRouter(config)
        expected = self.index.signature.get("model_sha256")
        if not expected or expected != self.router.model_sha256:
            self.index.close()
            self.router.close()
            raise ValueError("查询 embedding 权重与建库权重不一致")
        self.reranker = reranker or QwenReranker(config.path("reranker_dir"), config.instruction,
                                               config.reranker_batch_size, config.reranker_max_tokens)
        self.router.before_gpu = self.reranker.close
        self.terms = TermEnrichment(config)

    def start(self):
        """启动时加载预热 128 档，其余档位保持未加载。"""
        self.router.start()

    def search(self, request):
        request = request if isinstance(request, SearchRequest) else SearchRequest.model_validate(request)
        started = time.perf_counter()
        with self.lock:
            if request.quality_gate:
                self.gate.require_configured()
            with self.terms.snapshot() as (term_db, resources):
                recall_query, expansion = self.terms.expand(term_db, request.query, request.term_scopes, request.doc_ids)
            expansion["resources"] = resources
            if recall_query != request.query and self.router.token_length(recall_query) > self.config.gpu_max_tokens:
                recall_query = request.query
                expansion["status"] = "expansion_exceeds_input_budget"
            else:
                expansion["status"] = "expanded" if recall_query != request.query else "unchanged"
            expansion["recall_query"] = recall_query
            vector, routing = self.router.encode(recall_query)
            search_started = time.perf_counter()
            if self.config.keyword_enabled and hasattr(self.index, "hybrid_search"):
                candidates, hybrid_meta = self.index.hybrid_search(vector, recall_query, request.candidate_k,
                                                                  request.doc_ids, self.config.rrf_k)
            else:
                candidates = self.index.search(vector, request.candidate_k, request.doc_ids)
                hybrid_meta = {"dense_candidate_count": len(candidates),
                               "keyword": {"status": "disabled" if not self.config.keyword_enabled else "unsupported_index", "count": 0}}
            recall_ms = (time.perf_counter() - search_started) * 1000
            for rank, hit in enumerate(candidates, 1):
                hit["evidence_id"] = f"{hit['doc_id']}:{hit['chunk_id']}"
                if "rrf_rank" not in hit:
                    hit["dense_rank"] = rank
            limited, diversity_meta = limit_document_chunks(candidates, self.config.max_chunks_per_document)
            rerank_candidates = limited[:self.config.rerank_candidate_k]
            diversity_meta["rerank_pool_limit"] = self.config.rerank_candidate_k
            diversity_meta["pool_trimmed_count"] = len(limited) - len(rerank_candidates)
            if request.rerank:
                ordered, rerank_meta = self.reranker.rerank(request.query, rerank_candidates)
            else:
                ordered, rerank_meta = rerank_candidates, {"status": "disabled", "count": len(rerank_candidates)}
            # Qwen 已用原问题打分；词频仅对原问题相关词项进行低权重的辅助排序。
            ordered = [{**hit, "neural_rerank_rank": rank} for rank, hit in enumerate(ordered, 1)]
            # 重新打开短读事务，不在 NPU/GPU 推理或 API 门控期间持有术语数据库锁。
            with self.terms.snapshot() as (term_db, weighting_resources):
                ordered, term_weighting = self.terms.weight(term_db, request.query, ordered)
            term_weighting["resources"] = weighting_resources
            ordered = [{**hit, "rerank_rank": rank} for rank, hit in enumerate(ordered, 1)]
            mmr_input = {identity(hit): hit for hit in ordered}
            before_mmr = [identity(hit) for hit in ordered]
            if self.config.mmr_enabled and ordered:
                # 单片段的排列是恒等映射，不需要额外读取向量。
                vectors = self.index.vectors_for(ordered) if len(ordered) > 1 else [[1.0]]
                ordered = mmr_reorder(ordered, vectors, self.config.mmr_lambda)
            after_mmr = [identity(hit) for hit in ordered]
            payload_unchanged = all(all(hit.get(key) == value for key, value in mmr_input[identity(hit)].items())
                                    for hit in ordered)
            if not payload_unchanged:
                raise RuntimeError("MMR 修改了原始片段字段")
            mmr_meta = {"enabled": self.config.mmr_enabled, "lambda": self.config.mmr_lambda,
                        "relevance_normalization": "min_max", "input_count": len(before_mmr),
                        "output_count": len(after_mmr), "same_candidates": sorted(before_mmr) == sorted(after_mmr),
                        "payload_unchanged": payload_unchanged,
                        "input_ids": [list(key) for key in before_mmr], "output_ids": [list(key) for key in after_mmr]}
            gate_meta = {"status": "not_requested", "validated": False, "schema_validated": False,
                         "decision": "not_evaluated", "answer_allowed": False, "answer_scope": []}
            if request.quality_gate:
                context, count, gate_meta = self.gate_runner.run(
                    request.query, ordered, self.router.tokenizer, request.context_budget_tokens,
                    request.top_k, select_context)
                results = [{k: v for k, v in h.items() if k != "context_text"} for h in context]
            else:
                results = ordered[:request.top_k]
                context, count = select_context(results, self.router.tokenizer, request.context_budget_tokens)
            for rank, hit in enumerate(results, 1):
                hit["rank"] = rank
                hit["score"] = hit.get("term_rerank_score", hit.get("rerank_score", hit["dense_score"]))
                hit["quality_reviewed"] = request.quality_gate and gate_meta["schema_validated"]
                hit["quality_validated"] = gate_meta["validated"]
            result_by_id = {h["evidence_id"]: h for h in results}
            context = [{**h, **result_by_id[h["evidence_id"]]} for h in context]
            if results:
                empty_reason = None
            elif not candidates:
                empty_reason = "no_candidates"
            elif gate_meta["status"] == "failed":
                empty_reason = "quality_review_failed"
            elif gate_meta["status"] == "budget_exhausted":
                empty_reason = "quality_review_budget_exhausted"
            else:
                empty_reason = "no_evidence_fits_context" if gate_meta.get("eligible_count") else "no_evidence_passed_gate"
            return {"schema_version": 2, "request_id": uuid.uuid4().hex, "query": request.query,
                    "query_expansion": expansion,
                    "results": results, "context": context, "context_tokens": count,
                    "context_budget_tokens": request.context_budget_tokens, "quality_gate": gate_meta,
                    "answer_generated": False,
                    "empty_reason": empty_reason,
                    "encoding": routing, "retrieval": {"candidate_count": len(candidates), "recall_ms": recall_ms,
                    "rerank_candidate_count": len(rerank_candidates), "diversity": diversity_meta,
                    **hybrid_meta, "mmr": mmr_meta,
                    "term_weighting": term_weighting,
                    "reranker": rerank_meta, "index": self.index.status()},
                    "total_ms": (time.perf_counter() - started) * 1000}

    def get_chunk(self, doc_id, chunk_id):
        with self.lock:
            return self.index.get_chunk(doc_id, chunk_id)

    def read_document(self, request):
        with self.lock:
            if not hasattr(self.index, "read_document"):
                raise ValueError("paper_read 只支持正式知识库索引")
            return self.index.read_document(request)

    def status(self):
        return {"embedding": self.router.status(), "index": self.index.status(),
                "retrieval_policy": {"default_candidate_k": SearchRequest.model_fields["candidate_k"].default,
                                     "max_chunks_per_document": self.config.max_chunks_per_document,
                                     "keyword_enabled": self.config.keyword_enabled, "rrf_k": self.config.rrf_k,
                                     "rerank_candidate_k": self.config.rerank_candidate_k,
                                     "mmr_enabled": self.config.mmr_enabled, "mmr_lambda": self.config.mmr_lambda},
                "term_policy": {"enabled": self.config.term_enrichment_enabled,
                                "weight": self.config.term_weight, "max_keywords": self.config.term_max_keywords,
                                "max_expansions": self.config.term_max_expansions,
                                "active_terms_only": True, "rerank_and_gate_use_original_query": True},
                "reranker_loaded": self.reranker.model is not None,
                "quality_gate": {"provider": "deepseek",
                                 "default_enabled": SearchRequest.model_fields["quality_gate"].default,
                                 "policy_version": self.gate_policy.version,
                                 "rubric_sha256": self.gate.rubric_sha256,
                                 "limits": self.gate_policy.model_dump()}}

    def close(self):
        with self.lock:
            self.router.close()
            self.reranker.close()
            self.index.close()
