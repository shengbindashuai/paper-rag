"""混合召回、增量关键词索引和完整候选 MMR 排列的契约检查。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rag_core.storage import LibraryStore
from retrieval.config import RetrievalConfig
from retrieval.hybrid import reciprocal_rank_fusion, mmr_reorder, identity
from retrieval.index import SnapshotIndex
from retrieval.keyword_index import keyword_tokens
from retrieval.service import limit_document_chunks


ROOT = Path(__file__).resolve().parents[1]


def hit(doc, score=0.9):
    return {"doc_id": doc, "chunk_id": doc + "_1", "title": doc, "text": "original " + doc,
            "source_file": "PDF/" + doc + ".pdf", "source_spans": [{"page": 1}],
            "dense_score": score, "rerank_score": score, "dense_rank": 1}


class MMRTests(unittest.TestCase):
    def test_novel_candidate_moves_up_without_changing_any_input_field(self):
        hits = [hit("A", 1), hit("B", 0.99), hit("C", 0.98), hit("D", 0.1)]
        original = copy.deepcopy(hits)
        vectors = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=float)
        reordered = mmr_reorder(hits, vectors, 0.7)
        self.assertEqual([h["doc_id"] for h in reordered[:2]], ["A", "C"])
        self.assertEqual({identity(h) for h in hits}, {identity(h) for h in reordered})
        self.assertEqual(hits, original)
        lookup = {identity(h): h for h in original}
        for rank, result in enumerate(reordered, 1):
            self.assertEqual(result["mmr_rank"], rank)
            self.assertEqual({k: v for k, v in result.items() if k not in {"mmr_rank", "mmr_score"}}, lookup[identity(result)])

    def test_full_40_in_full_40_out_even_for_identical_vectors_and_text(self):
        hits = [{**hit(str(i), 0.99), "text": "same body"} for i in range(40)]
        reordered = mmr_reorder(hits, np.ones((40, 3)), 0.7)
        self.assertEqual(len(reordered), 40)
        self.assertEqual([identity(h) for h in reordered], [identity(h) for h in hits])
        self.assertEqual(mmr_reorder([], np.empty((0, 3))), [])

    def test_invalid_vectors_and_duplicate_identity_fail_instead_of_dropping(self):
        hits = [hit("A"), hit("B")]
        for vectors in [np.ones((1, 3)), np.zeros((2, 3)), np.full((2, 3), np.nan)]:
            with self.assertRaises(ValueError):
                mmr_reorder(hits, vectors)
        with self.assertRaises(ValueError):
            mmr_reorder([hits[0], hits[0]], np.ones((2, 3)))
        for value in [-1, 2, float("nan")]:
            with self.assertRaises(ValueError):
                mmr_reorder(hits, np.ones((2, 3)), value)

    def test_weight_one_preserves_relevance_order(self):
        hits = [hit("A", 1), hit("B", 0.9), hit("C", 0.8)]
        self.assertEqual([identity(h) for h in mmr_reorder(hits, np.ones((3, 2)), 1)], [identity(h) for h in hits])

    def test_pdf_and_linked_notes_share_cap_without_changing_ids(self):
        inputs = [{**hit("P1"), "chunk_id": "p1"}, {**hit("N1"), "linked_doc_id": "P1"}, hit("P2")]
        selected, meta = limit_document_chunks(inputs, 1)
        self.assertEqual([h["doc_id"] for h in selected], ["P1", "P2"])
        self.assertEqual(meta["per_document_counts"], {"P1": 1, "P2": 1})


class RRFTests(unittest.TestCase):
    def test_fuses_by_identity_keeps_scores_and_rewards_both_channels(self):
        a, b, c = hit("A", 0.6), hit("B", 0.5), hit("C", 0.2)
        keyword = [{**b, "keyword_score": -7}, {**c, "keyword_score": -3}]
        original = copy.deepcopy([a, b, *keyword])
        result = reciprocal_rank_fusion([a, b], keyword, 60)
        self.assertEqual([h["doc_id"] for h in result], ["B", "A", "C"])
        self.assertAlmostEqual(result[0]["rrf_score"], 1/62 + 1/61)
        self.assertEqual(result[0]["dense_score"], 0.5)
        self.assertEqual(result[0]["keyword_score"], -7)
        self.assertIsNone(result[2]["dense_rank"])
        self.assertEqual(original, [a, b, *keyword])

    def test_repeated_channel_entry_is_not_double_counted(self):
        a = hit("A")
        result = reciprocal_rank_fusion([a, a], [])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["rrf_score"], 1/61)


class KeywordTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "tmp" / "hybrid_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.store = LibraryStore(self.root / "library", {"test": 1})
        self.index = None

    def tearDown(self):
        if self.index:
            self.index.close()
        self.store.close()
        self.temp.cleanup()

    def put(self, doc, text, axis=0, title=None):
        item = {"doc_id": doc, "source_file": f"PDF/{doc}.pdf", "sha256": "hash" + doc, "title": title or doc}
        chunk = {"doc_id": doc, "chunk_id": doc + "_1", "text": text, "retrieval_eligible": True,
                 "source_file": item["source_file"], "source_sha256": item["sha256"], "section": "Algorithm"}
        vectors = np.zeros((1, 1024), dtype=np.float32)
        vectors[0, axis] = 1
        self.store.commit_document(item, [chunk], vectors)
        self.store.refresh_snapshot()

    def start(self):
        self.index = SnapshotIndex(self.store.directory)
        vector = np.zeros(1024, dtype=np.float32)
        vector[0] = 1
        return vector

    def test_keyword_only_hit_joins_dense_pool_and_original_vectors_are_used(self):
        self.put("DENSE", "unrelated dense passage", axis=0)
        self.put("KEYWORD", "PPO reuses sampled trajectories for multiple epochs", axis=1)
        query = self.start()
        combined, meta = self.index.hybrid_search(query, "PPO", 1)
        self.assertEqual({h["doc_id"] for h in combined}, {"DENSE", "KEYWORD"})
        self.assertEqual(meta["keyword"]["count"], 1)
        vectors = self.index.vectors_for(combined)
        by_id = {h["doc_id"]: v for h, v in zip(combined, vectors)}
        self.assertEqual(by_id["DENSE"][0], 1)
        self.assertEqual(by_id["KEYWORD"][1], 1)

    def test_incremental_replacement_deletion_and_live_revision(self):
        self.put("A", "PPO original evidence")
        query = self.start()
        self.assertEqual(self.index.hybrid_search(query, "PPO", 10)[1]["keyword"]["count"], 1)
        self.put("A", "replacement with SAC")
        self.assertEqual(self.index.hybrid_search(query, "PPO", 10)[1]["keyword"]["count"], 0)
        self.assertEqual(self.index.hybrid_search(query, "SAC", 10)[1]["keyword"]["count"], 1)
        self.put("B", "SAC new evidence")
        self.assertEqual(self.index.hybrid_search(query, "SAC", 10)[1]["keyword"]["count"], 2)
        self.store.record({"doc_id": "A", "source_file": "PDF/A.pdf", "sha256": "hashA", "title": "A"}, "missing")
        self.store.refresh_snapshot()
        self.assertEqual(self.index.hybrid_search(query, "SAC", 10)[1]["keyword"]["count"], 1)

    def test_doc_filter_alias_and_chinese_bigram_query(self):
        self.put("A", "算法通过重复利用采样数据更新策略")
        self.put("B", "同样通过重复利用数据进行优化")
        self.store.record({"doc_id": "alias", "source_file": "x", "sha256": "h", "title": "x"}, "duplicate", alias_of="A")
        query = self.start()
        results, meta = self.index.hybrid_search(query, "如何重复利用采样数据", 10, ["alias"])
        self.assertEqual({h["doc_id"] for h in results}, {"A"})
        self.assertEqual(meta["keyword"]["count"], 1)
        self.assertIn("重复", keyword_tokens("如何重复利用采样数据"))

    def test_fts_syntax_is_data_and_no_search_terms_are_explicit(self):
        self.put("A", "PPO evidence")
        query = self.start()
        self.assertEqual(self.index.hybrid_search(query, 'PPO" OR * - NEAR(', 10)[1]["keyword"]["count"], 1)
        self.assertEqual(self.index.hybrid_search(query, "!!!", 10)[1]["keyword"]["status"], "no_search_terms")

    def test_mmr_rejects_stale_candidate_vectors(self):
        self.put("A", "PPO evidence")
        query = self.start()
        hits = self.index.search(query, 10)
        self.put("A", "updated PPO evidence")
        with self.assertRaises(RuntimeError):
            self.index.vectors_for(hits)

    def test_config_rejects_invalid_hybrid_parameters(self):
        for kwargs in [{"mmr_lambda": float("nan")}, {"rrf_k": 0}, {"mmr_enabled": "yes"}, {"rerank_candidate_k": 0}]:
            with self.assertRaises(ValueError):
                RetrievalConfig(ROOT, **kwargs)


if __name__ == "__main__":
    unittest.main()
