"""持久化事务、增量替换、快照一致性及双栏顺序的关键检查。"""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ingestion.preprocessing.quality import order_columns
from rag_core.storage import LibraryStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[1] / "tmp" / "library_tests"
        target.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.store = LibraryStore(Path(self.temp.name), {"test": 1})
        self.item = {"doc_id": "P0001", "source_file": "PDF/a.pdf", "sha256": "sha", "title": "A"}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def vector(self, axis=0):
        result = np.zeros((1, 1024), dtype=np.float32)
        result[0, axis] = 1
        return result

    def chunk(self, text="evidence"):
        return [{"doc_id": "P0001", "chunk_id": "P0001_0001", "retrieval_eligible": True, "text": text}]

    def test_invalid_replacement_preserves_committed_document(self):
        self.store.commit_document(self.item, self.chunk(), self.vector())
        with self.assertRaises(ValueError):
            self.store.commit_document(self.item, self.chunk("bad"), np.zeros((1, 1024), dtype=np.float32))
        self.assertEqual(self.store.status()["chunks"], 1)
        data = self.store.db.execute("SELECT data FROM chunks").fetchone()[0]
        self.assertIn("evidence", data)

    def test_incremental_update_requires_new_snapshot(self):
        self.store.commit_document(self.item, self.chunk(), self.vector())
        first = self.store.refresh_snapshot()
        self.assertEqual(self.store.search(self.vector()[0])[0]["text"], "evidence")
        self.store.commit_document(self.item, self.chunk("replacement"), self.vector(1))
        with self.assertRaises(RuntimeError):
            self.store.search(self.vector()[0])
        second = self.store.refresh_snapshot()
        self.assertGreater(second["revision"], first["revision"])
        hits = self.store.search(self.vector(1)[0])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["text"], "replacement")
        self.assertAlmostEqual(hits[0]["score"], 1)

    def test_signature_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            other = LibraryStore(Path(self.temp.name), {"test": 2})
            other.close()


class ColumnTests(unittest.TestCase):
    def test_reading_excerpt_preserves_line_sources_and_excludes_header_metadata(self):
        from ingestion.preprocessing.notes import extract_note
        target = Path(__file__).resolve().parents[1] / "tmp" / "library_tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            path = Path(directory) / "note.md"
            path.write_text("# Paper title\n\ntier: T3\n\n## Method\n\nOriginal evidence\ncontinues here.\n", encoding="utf-8")
            raw, cleaned = extract_note({"path": str(path), "doc_id": "Ntest", "source_file": "阅读摘录/note.md",
                                         "sha256": "test", "title": "note"})
            self.assertEqual(raw["source_type"], "reading_excerpt")
            self.assertEqual(cleaned["segments"][0]["text"], "Original evidence continues here.")
            anchor = cleaned["segments"][0]["sources"][0]
            self.assertEqual((anchor["start_line"], anchor["end_line"]), (7, 8))
            self.assertNotIn("tier:", " ".join(s["text"] for s in cleaned["segments"]))

    def test_interleaved_columns_reordered_without_touching_original(self):
        def block(name, box):
            return {"block_id": name, "bbox": box, "text": "narrative words " * 15,
                    "lines": [{"text": "narrative words " * 15, "bbox": box}]}
        raw = {"pages": [{"page": 1, "width": 600, "blocks": [
            block("L1", [40, 100, 280, 200]), block("R1", [320, 100, 560, 200]),
            block("L2", [40, 220, 280, 300]), block("R2", [320, 220, 560, 300])]}]}
        ordered, report = order_columns(raw)
        self.assertEqual([b["block_id"] for b in ordered["pages"][0]["blocks"]], ["L1", "L2", "R1", "R2"])
        self.assertEqual(raw["pages"][0]["blocks"][1]["block_id"], "R1")
        self.assertEqual(report["two_column_reordered_pages"], [1])


if __name__ == "__main__":
    unittest.main()
