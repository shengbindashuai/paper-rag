"""只读消费建库快照；一次搜索使用一个 SQLite 读事务，避免混读不同修订。"""
import json
from pathlib import Path
import sqlite3

import numpy as np


def normalized_query(vector):
    vector = np.array(vector, dtype=np.float32, copy=True).reshape(-1)
    if vector.shape != (1024,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("查询向量无效")
    return vector / np.linalg.norm(vector)


class SnapshotIndex:
    def __init__(self, directory: Path, root: Path | None = None):
        self.directory = directory.resolve()
        self.root = root
        database = self.directory / "library.sqlite3"
        if not database.is_file():
            raise FileNotFoundError(f"建库索引尚不存在：{database}")
        self.db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        signature = self.db.execute("SELECT value FROM settings WHERE key='signature'").fetchone()
        self.signature = json.loads(signature[0]) if signature else {}

    def status(self):
        path = self.directory / "current.json"
        return {"kind": "library", "directory": str(self.directory),
                "snapshot": json.loads(path.read_text(encoding="utf-8")) if path.exists() else None,
                "model_sha256": self.signature.get("model_sha256")}

    def _decorate(self, row):
        chunk = json.loads(row["data"])
        document = self.db.execute("SELECT title FROM documents WHERE doc_id=?", (row["doc_id"],)).fetchone()
        aliases = [dict(r) for r in self.db.execute("SELECT doc_id,source_file FROM documents WHERE alias_of=?", (row["doc_id"],))]
        result = {"title": document["title"] if document else "", "aliases": aliases, **chunk}
        if self.root:
            from knowledge.sources import source_metadata
            meta = source_metadata(self.root, chunk["doc_id"], chunk.get("source_sha256"))
            if meta:
                result.update(source_type=meta["source_type"], source_metadata={
                    k: meta[k] for k in ["source_url", "conversation_id", "registered_at", "original_path"] if meta.get(k)})
        return result

    def read_document(self, request):
        """按原文片段顺序分页；返回完整片段，不截断引用。"""
        row = self.db.execute("SELECT * FROM documents WHERE doc_id=?", (request.doc_id,)).fetchone()
        if row is None:
            raise KeyError(f"文档不存在：{request.doc_id}")
        canonical = row["alias_of"] or request.doc_id
        self.db.execute("BEGIN")
        try:
            revision = int(self.db.execute("SELECT value FROM settings WHERE key='revision'").fetchone()[0])
            matched, results, used, more = 0, [], 0, False
            for row in self.db.execute("SELECT data,doc_id FROM chunks WHERE doc_id=? ORDER BY id", (canonical,)):
                chunk = json.loads(row["data"])
                if request.section and request.section.casefold() not in chunk.get("section", "").casefold():
                    continue
                if request.page_start or request.page_end:
                    if not any((request.page_start is None or p >= request.page_start)
                               and (request.page_end is None or p <= request.page_end) for p in chunk.get("pages", [])):
                        continue
                matched += 1
                if matched <= request.offset:
                    continue
                size = len(chunk["text"])
                if len(results) == request.limit or used + size > request.max_chars:
                    if not results:
                        raise ValueError(f"首个片段需要 {size} 字符预算，请增大 max_chars 或用 paper_get_chunk")
                    more = True
                    break
                results.append(self._decorate(row))
                used += size
            return {"doc_id": request.doc_id, "canonical_doc_id": canonical, "chunks": results,
                    "offset": request.offset, "next_offset": request.offset + len(results) if more else None,
                    "has_more": more, "snapshot_revision": revision, "chars": used,
                    "scope": "indexed_chunks", "answer_authorized": False}
        finally:
            self.db.rollback()

    def get_chunk(self, doc_id, chunk_id):
        row = self.db.execute("SELECT data,doc_id FROM chunks WHERE doc_id=? AND chunk_id=?", (doc_id, chunk_id)).fetchone()
        if row is None:
            raise KeyError(f"片段不存在：{doc_id}/{chunk_id}")
        return self._decorate(row)

    def hybrid_search(self, vector, query, top_k, doc_ids=None, rrf_k=60):
        from .hybrid import reciprocal_rank_fusion
        from .keyword_index import KeywordIndex
        dense = self.search(vector, top_k, doc_ids)
        self.db.execute("BEGIN")
        try:
            revision = int(self.db.execute("SELECT value FROM settings WHERE key='revision'").fetchone()[0])
            if any(h["snapshot_revision"] != revision for h in dense):
                raise RuntimeError("向量与关键词召回跨越索引修订，请重试")
            canonical = set(doc_ids) if doc_ids else None
            if canonical:
                for doc_id in doc_ids:
                    row = self.db.execute("SELECT alias_of FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
                    if row and row[0]:
                        canonical.add(row[0])
            rows, metadata = KeywordIndex(self.directory).search(self.db, revision, query, top_k, canonical)
            keyword, normalized = [], normalized_query(vector)
            for rank, (row_id, score) in enumerate(rows, 1):
                row = self.db.execute("SELECT data,doc_id,vector FROM chunks WHERE id=?", (row_id,)).fetchone()
                if row is None:
                    raise RuntimeError("关键词索引包含不存在的片段")
                hit = self._decorate(row)
                hit.update(dense_score=float(np.frombuffer(row["vector"], dtype=np.float32) @ normalized),
                           keyword_score=float(score), keyword_rank=rank, snapshot_revision=revision)
                keyword.append(hit)
            dense = [{**h, "dense_rank": rank} for rank, h in enumerate(dense, 1)]
            combined = reciprocal_rank_fusion(dense, keyword, rrf_k)
            return combined, {"dense_candidate_count": len(dense), "keyword": metadata,
                              "rrf": {"k": rrf_k, "dense_count": len(dense), "keyword_count": len(keyword),
                                      "fused_count": len(combined)}}
        finally:
            self.db.rollback()

    def vectors_for(self, hits):
        """仅供本地 MMR 使用；按输入 ID 顺序取原始向量，不附加到工具响应。"""
        if not hits:
            return np.empty((0, 1024), dtype=np.float32)
        self.db.execute("BEGIN")
        try:
            revision = int(self.db.execute("SELECT value FROM settings WHERE key='revision'").fetchone()[0])
            vectors = []
            for hit in hits:
                if hit.get("snapshot_revision", revision) != revision:
                    raise RuntimeError("MMR 候选与当前向量修订不一致，请重试")
                row = self.db.execute("SELECT vector FROM chunks WHERE doc_id=? AND chunk_id=?", (hit["doc_id"], hit["chunk_id"])).fetchone()
                if row is None:
                    raise KeyError("MMR 候选向量不存在")
                vectors.append(np.frombuffer(row[0], dtype=np.float32).copy())
            return np.stack(vectors)
        finally:
            self.db.rollback()

    def search(self, vector, top_k, doc_ids=None):
        self.db.execute("BEGIN")
        try:
            path = self.directory / "current.json"
            if not path.exists():
                raise RuntimeError("索引尚未发布快照")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            revision = int(self.db.execute("SELECT value FROM settings WHERE key='revision'").fetchone()[0])
            if manifest["revision"] != revision:
                raise RuntimeError("建库快照过期或仍在发布，请在建库完成后重试")
            directory = (self.directory / manifest["directory"]).resolve()
            if not directory.is_relative_to(self.directory):
                raise ValueError("快照目录越界")
            matrix = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
            ids = np.load(directory / "ids.npy", mmap_mode="r", allow_pickle=False)
            if matrix.shape != (manifest["count"], 1024) or ids.shape != (manifest["count"],):
                raise ValueError("快照形状不符合 manifest")
            allowed = None
            if doc_ids:
                canonical = set(doc_ids)
                for doc_id in doc_ids:
                    row = self.db.execute("SELECT alias_of FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
                    if row and row[0]:
                        canonical.add(row[0])
                marks = ",".join("?" for _ in canonical)
                allowed = {r[0] for r in self.db.execute(f"SELECT id FROM chunks WHERE doc_id IN ({marks})", tuple(canonical))}
            query, candidates = normalized_query(vector), []
            for start in range(0, len(matrix), 16384):
                block_ids = np.asarray(ids[start:start + 16384])
                scores = matrix[start:start + 16384] @ query
                mask = np.ones(len(scores), dtype=bool) if allowed is None else np.array([int(i) in allowed for i in block_ids])
                selected = np.flatnonzero(mask)
                # 确定性同分排序：按数据库行 ID 打破并列。
                selected = selected[np.lexsort((block_ids[selected], -scores[selected]))[:top_k]]
                candidates.extend((float(scores[i]), int(block_ids[i])) for i in selected)
            selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
            hits = []
            for score, row_id in selected:
                row = self.db.execute("SELECT data,doc_id FROM chunks WHERE id=?", (row_id,)).fetchone()
                if row is None:
                    raise RuntimeError("索引快照与数据库记录不一致")
                hits.append({**self._decorate(row), "dense_score": score, "snapshot_revision": revision})
            return hits
        finally:
            self.db.rollback()

    def close(self):
        self.db.close()


class ExperimentIndex:
    """显式指定才使用的四篇论文测试适配器；不会冒充已经建好的全库。"""
    def __init__(self, directory):
        self.directory = directory
        self.chunks = [json.loads(x) for x in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.matrix = np.load(directory / "document_embeddings.npy", mmap_mode="r", allow_pickle=False)
        if self.matrix.shape != (len(self.chunks), 1024):
            raise ValueError("实验索引行数或维度不符")
        source = json.loads((directory.parent / "results.json").read_text(encoding="utf-8"))
        self.signature = {"model_sha256": source["model_provenance"]["sha256"]}

    def status(self):
        return {"kind": "experiment", "count": len(self.chunks), "directory": str(self.directory)}

    def search(self, vector, top_k, doc_ids=None):
        scores = self.matrix @ normalized_query(vector)
        order = np.argsort(-scores, kind="stable")
        chosen = [i for i in order if not doc_ids or self.chunks[i]["doc_id"] in doc_ids][:top_k]
        return [{**self.chunks[i], "title": self.chunks[i]["embedding_text"].splitlines()[0].removeprefix("Title: "),
                 "aliases": [], "dense_score": float(scores[i])} for i in chosen]

    def get_chunk(self, doc_id, chunk_id):
        for chunk in self.chunks:
            if chunk["doc_id"] == doc_id and chunk["chunk_id"] == chunk_id:
                return chunk
        raise KeyError(f"片段不存在：{doc_id}/{chunk_id}")

    def vectors_for(self, hits):
        positions = {(h["doc_id"], h["chunk_id"]): i for i, h in enumerate(self.chunks)}
        return np.asarray([self.matrix[positions[(h["doc_id"], h["chunk_id"])]] for h in hits])

    def close(self):
        self.matrix = None
