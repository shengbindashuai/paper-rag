"""验证召回、文档限额、重排和门控之间的实际数据边界。"""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from retrieval.config import RetrievalConfig
from retrieval.quality_gate import DeepSeekQualityGate
from retrieval.schemas import GatePolicy, SearchRequest
from retrieval.service import RetrievalService, limit_document_chunks
from retrieval.tool_api import definitions
from tests.test_quality_gate import make_audit, api_response


ROOT = Path(__file__).resolve().parents[1]


class CandidateDiversityTests(unittest.TestCase):
    def make_service(self, handler):
        hits = []
        for doc, count in [("dominant", 120), *[(f"D{i}", 8) for i in range(10)]]:
            for i in range(count):
                hits.append({"doc_id": doc, "chunk_id": f"{doc}_{i}", "text": f"Evidence {doc} {i}",
                             "dense_score": 1 - len(hits) / 1000, "retrieval_eligible": True})
        class Router:
            model_sha256 = "test"
            tokenizer = staticmethod(lambda text: {"input_ids": list(text)})
            def encode(self, query): return np.ones(1024), {}
            def close(self): pass
        class Index:
            signature = {"model_sha256": "test"}
            def search(self, vector, count, doc_ids):
                self.requested_count = count
                selected = [h for h in hits if not doc_ids or h["doc_id"] in doc_ids]
                return copy.deepcopy(selected[:count])
            def status(self): return {}
            def vectors_for(self, candidates): return np.eye(len(candidates), 1024, dtype=np.float32)
            def close(self): pass
        class Reranker:
            model = None
            def rerank(self, query, candidates):
                self.received = copy.deepcopy(candidates)
                return [{**h, "rerank_score": 1.0} for h in reversed(candidates)], {"count": len(candidates)}
            def close(self): pass
        options = RetrievalConfig(ROOT).deepseek
        gate = DeepSeekQualityGate(options, transport=httpx.MockTransport(handler))
        service = RetrievalService(RetrievalConfig(ROOT), router=Router(), index=Index(), reranker=Reranker(), gate=gate)
        self.addCleanup(service.close)
        return service

    def test_200_recalled_per_document_8_then_gate_receives_top16(self):
        received = []
        def handler(request):
            evidence = json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]
            received.append(evidence)
            return api_response(make_audit(evidence))
        service = self.make_service(handler)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            result = service.search({"query": "mechanism"})
        self.assertEqual(service.index.requested_count, 200)
        self.assertEqual(result["retrieval"]["candidate_count"], 200)
        self.assertEqual(len(service.reranker.received), 40)
        self.assertTrue(all(n == 8 for n in result["retrieval"]["diversity"]["per_document_counts"].values()))
        self.assertEqual(result["retrieval"]["diversity"]["dropped_count"], 112)
        self.assertEqual(service.reranker.received[8]["dense_rank"], 121)
        self.assertEqual(len(received), 1)
        self.assertEqual(len(received[0]), 16)
        expected = list(reversed(service.reranker.received))[:16]
        self.assertEqual([h["evidence_id"] for h in received[0]], [h["evidence_id"] for h in expected])
        self.assertTrue(result["quality_gate"]["answer_allowed"])

    def test_insufficient_gate_expands_from_16_to_30_and_stops(self):
        counts = []
        def handler(request):
            evidence = json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]
            counts.append(len(evidence))
            audit = make_audit(evidence, groups=[])
            for item in audit["assessments"]:
                item.update(relation="background", support_spans=[])
            return api_response(audit)
        service = self.make_service(handler)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            result = service.search({"query": "mechanism"})
        self.assertEqual(counts, [16, 30])
        self.assertEqual(result["quality_gate"]["reviewed_count"], 30)
        self.assertFalse(result["quality_gate"]["answer_allowed"])

    def test_gate_follows_changed_mmr_order_and_preserves_the_40_candidate_set(self):
        received = []
        def handler(request):
            evidence = json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]
            received.extend((h["evidence_id"] for h in evidence))
            return api_response(make_audit(evidence))
        service = self.make_service(handler)
        vectors = np.eye(40, 1024, dtype=np.float32)
        vectors[1] = vectors[0]
        with patch.object(service.index, "vectors_for", return_value=vectors), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            result = service.search({"query": "mechanism"})
        metadata = result["retrieval"]["mmr"]
        self.assertEqual((metadata["input_count"], metadata["output_count"]), (40, 40))
        self.assertTrue(metadata["same_candidates"])
        self.assertNotEqual(metadata["input_ids"], metadata["output_ids"])
        self.assertEqual(received, [f"{doc}:{chunk}" for doc, chunk in metadata["output_ids"][:16]])

    def test_targeted_query_and_disabled_rerank_keep_document_cap(self):
        service = self.make_service(lambda request: self.fail("门控关闭时不应调用 API"))
        with patch.object(service.reranker, "rerank", side_effect=AssertionError("重排已关闭")):
            result = service.search({"query": "q", "doc_ids": ["dominant"], "top_k": 10,
                                     "quality_gate": False, "rerank": False})
        self.assertEqual(len(result["results"]), 8)
        self.assertEqual(result["retrieval"]["candidate_count"], 120)
        self.assertEqual(result["retrieval"]["rerank_candidate_count"], 8)

    def test_limiter_preserves_order_and_empty_pool(self):
        hits = [{"doc_id": doc, "dense_rank": i} for i, doc in enumerate(["A", "B", "A", "A", "B"], 1)]
        selected, meta = limit_document_chunks(hits, 2)
        self.assertEqual([h["dense_rank"] for h in selected], [1, 2, 3, 5])
        self.assertEqual(meta["per_document_counts"], {"A": 2, "B": 2})
        self.assertEqual(limit_document_chunks([], 8)[0], [])

    def test_public_defaults_and_cap_configuration(self):
        self.assertEqual(SearchRequest(query="q").candidate_k, 200)
        self.assertEqual(definitions()[0]["inputSchema"]["properties"]["candidate_k"]["default"], 200)
        policy = GatePolicy.model_validate_json((ROOT / "config/quality_gate_policy.json").read_text(encoding="utf-8"))
        self.assertEqual((policy.batch_size, policy.max_candidates), (16, 30))
        config = RetrievalConfig.load(ROOT / "config/retrieval.json", ROOT)
        self.assertEqual(config.max_chunks_per_document, 8)
        for invalid in [0, -1, True, 8.5, "8", 201]:
            with self.assertRaises(ValueError):
                RetrievalConfig(ROOT, max_chunks_per_document=invalid)


if __name__ == "__main__":
    unittest.main()
