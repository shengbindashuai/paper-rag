"""验证容易造成证据损坏的边界：断词、否定/数字、长句、来源区间与长度。

运行：python -m unittest discover -s tests -t . -v
无需下载模型，不加载权重，不修改论文文件。
"""
import unittest
from pathlib import Path

from ingestion.preprocessing.chunking import chunk_document, load_tokenizer
from ingestion.preprocessing.cleaning import clean_lines
from rag_core.config import ProcessingConfig, Workspace


def lines(*texts):
    return [{"text": text, "line_id": f"test:l{i}"} for i, text in enumerate(texts)]


class CleaningTests(unittest.TestCase):
    def test_dehyphenation_requires_document_evidence(self):
        text, _, changes = clean_lines(lines("we al-", "ternate updates"), {"alternate"}, False)
        self.assertEqual(text, "we alternate updates")
        self.assertTrue(any(c["action"] == "dehyphenate_with_document_evidence" for c in changes))

    def test_real_hyphen_negative_number_and_negation_survive(self):
        text, flags, _ = clean_lines(lines("fine-", "tuning does not change −0.2 or 1e-4."), set(), False)
        self.assertEqual(text, "fine-tuning does not change −0.2 or 1e-4.")
        self.assertIn("line_end_hyphen_preserved", flags)

    def test_ing_word_can_use_document_lemma_as_evidence(self):
        text, _, changes = clean_lines(lines("includ-", "ing examples"), {"include"}, False)
        self.assertEqual(text, "including examples")
        self.assertEqual(changes[0]["action"], "dehyphenate_with_document_morphology")

    def test_unknown_math_glyph_is_visible_not_silently_deleted(self):
        text, flags, _ = clean_lines(lines("loss = \x02x\x03"), set(), True)
        self.assertIn("[PDF_GLYPH_0002]", text)
        self.assertIn("unmapped_pdf_glyph", flags)


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (Workspace(Path(__file__).resolve().parents[1]).tokenizer / "tokenizer.json").is_file():
            raise unittest.SkipTest("可选分块集成测试需要本地 embedding tokenizer")
        cls.tokenizer = load_tokenizer(Workspace(Path(__file__).resolve().parents[1]).tokenizer)

    def make_document(self, text, eligible=True):
        return {
            "doc_id": "PTEST", "title": "Test paper", "source_file": "PDF/test.pdf", "source_sha256": "test",
            "segments": [{"segment_id": "s1", "kind": "paragraph", "parent_id": "section1",
                          "section_path": ["Methods"], "text": text, "quality_flags": [],
                          "retrieval_eligible": eligible,
                          "sources": [{"page": 1, "block_id": "b1", "bbox": [0, 0, 100, 100], "line_ids": ["l1"]}]}],
        }

    def test_long_sentence_keeps_chinese_characters_and_every_source_character(self):
        text = "这是一条包含否定词不得删除和数值−0.2的很长证据" * 70
        document = self.make_document(text)
        config = ProcessingConfig(target_tokens=60, max_tokens=90, overlap_tokens=10)
        chunks = chunk_document(document, self.tokenizer, config)
        self.assertGreater(len(chunks), 1)
        coverage = set()
        for chunk in chunks:
            self.assertLessEqual(chunk["token_count"], 90)
            self.assertNotIn("\ufffd", chunk["text"])
            for span in chunk["source_spans"]:
                a, b = span["segment_char_start"], span["segment_char_end"]
                self.assertEqual(text[a:b], span["text"])
                coverage.update(range(a, b))
        self.assertEqual(coverage, set(range(len(text))))
        # 回归：计数必须与推理时批量 tokenizer 的输入一致，包括自动结束标记。
        batch = self.tokenizer([c["embedding_text"] for c in chunks], padding=True, return_tensors="pt")
        actual_lengths = batch["attention_mask"].sum(dim=1).tolist()
        self.assertEqual(actual_lengths, [c["token_count"] for c in chunks])
        self.assertLessEqual(max(actual_lengths), config.max_tokens)

    def test_overlap_does_not_create_tail_only_duplicate(self):
        document = self.make_document("First evidence sentence. Second evidence sentence. Third evidence sentence. " * 12)
        config = ProcessingConfig(target_tokens=50, max_tokens=80, overlap_tokens=15)
        chunks = chunk_document(document, self.tokenizer, config)
        seen = set()
        for chunk in chunks:
            current = {(s["segment_id"], s["segment_char_start"], s["segment_char_end"]) for s in chunk["source_spans"]}
            self.assertTrue(current - seen, "每个片段必须包含新内容")
            seen.update(current)

    def test_non_eligible_evidence_is_retained_but_not_promoted(self):
        chunks = chunk_document(self.make_document("Needs source review.", False), self.tokenizer, ProcessingConfig())
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0]["retrieval_eligible"])

    def test_page_attribution_follows_piece_not_whole_cross_page_paragraph(self):
        page1 = "First page explains the method. " * 12
        page2 = "Second page describes the experiments. " * 12
        document = self.make_document(page1 + page2)
        document["segments"][0]["sources"] = [
            {"page": 1, "block_id": "b1", "bbox": [0, 0, 100, 100],
             "segment_char_start": 0, "segment_char_end": len(page1)},
            {"page": 2, "block_id": "b2", "bbox": [0, 0, 100, 100],
             "segment_char_start": len(page1), "segment_char_end": len(page1 + page2)},
        ]
        chunks = chunk_document(document, self.tokenizer, ProcessingConfig(50, 80, 10))
        self.assertEqual(chunks[0]["pages"], [1])
        self.assertEqual(chunks[-1]["pages"], [2])


if __name__ == "__main__":
    unittest.main()
