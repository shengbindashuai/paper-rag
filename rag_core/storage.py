"""SQLite 保存来源和向量；NumPy 内存映射快照用于精确余弦检索。

每篇文档在一个事务中替换，失败不会留下半篇向量。快照写完后才原子发布。
不用把全部原文/向量一次加载进内存；新增文档只编码自身，快照刷新只做 I/O。
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

import numpy as np
from .io import atomic_json, utc_now


@contextmanager
def writer_lock(directory: Path):
    """操作系统文件锁：禁止两个建库/入库进程同时写；进程退出自动释放。"""
    if os.name == "nt":
        import msvcrt
    else:
        import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "writer.lock").open("a+b") as stream:
        try:
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("已有建库/入库进程持有写锁；请先查看 status") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class LibraryStore:
    def __init__(self, directory: Path, signature: dict | None = None):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "library.sqlite3", timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA cache_size=-65536;
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents(
                doc_id TEXT PRIMARY KEY, source_file TEXT,source_sha256 TEXT,title TEXT,
                status TEXT NOT NULL,chunk_count INTEGER DEFAULT 0,error TEXT,alias_of TEXT,updated_at TEXT);
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,doc_id TEXT NOT NULL,chunk_id TEXT NOT NULL,
                data TEXT NOT NULL,vector BLOB NOT NULL,UNIQUE(doc_id,chunk_id));
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(doc_id);
        """)
        if not self.setting("revision"):
            self.set_setting("revision", "0")
        if signature is not None:
            serialized = json.dumps(signature, sort_keys=True, ensure_ascii=False)
            previous = self.setting("signature")
            if previous and previous != serialized:
                self.db.close()
                raise ValueError("建库参数、处理代码或模型发生变化；请指定新的 --index-dir，避免混合不同版本的向量")
            self.set_setting("signature", serialized)

    def close(self):
        self.db.close()

    def setting(self, key):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_setting(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, value))

    def document(self, doc_id):
        row = self.db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def record(self, item, status, error=None, alias_of=None):
        """终止状态也入账，明确区分无法处理、无合格正文与已入库。"""
        with self.db:
            old = self.db.execute("SELECT count(*) FROM chunks WHERE doc_id=?", (item["doc_id"],)).fetchone()[0]
            self.db.execute("DELETE FROM chunks WHERE doc_id=?", (item["doc_id"],))
            self.db.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,0,?,?,?)",
                            (item["doc_id"], item["source_file"], item["sha256"], item["title"], status, error, alias_of, utc_now()))
            if old:
                self._bump_revision()

    def _bump_revision(self):
        self.db.execute("UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")

    def commit_document(self, item, chunks, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), 1024) or not len(chunks):
            raise ValueError("向量数量或维度不符")
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("向量数值/归一化检查失败")
        if any(c["doc_id"] != item["doc_id"] or not c["retrieval_eligible"] for c in chunks):
            raise ValueError("不允许混入其他文档或未通过筛选的片段")
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE doc_id=?", (item["doc_id"],))
            self.db.executemany("INSERT INTO chunks(doc_id,chunk_id,data,vector) VALUES (?,?,?,?)", [
                (item["doc_id"], c["chunk_id"], json.dumps(c, ensure_ascii=False, separators=(",", ":")), v.tobytes())
                for c, v in zip(chunks, vectors)])
            self.db.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,NULL,NULL,?)",
                            (item["doc_id"], item["source_file"], item["sha256"], item["title"], "indexed", len(chunks), utc_now()))
            self._bump_revision()

    def status(self):
        counts = {row[0]: row[1] for row in self.db.execute("SELECT status,count(*) FROM documents GROUP BY status")}
        return {"documents": counts, "chunks": self.db.execute("SELECT count(*) FROM chunks").fetchone()[0],
                "revision": int(self.setting("revision")), "snapshot": self.snapshot_manifest()}

    def snapshot_manifest(self):
        path = self.directory / "current.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def refresh_snapshot(self):
        count = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if not count:
            raise ValueError("没有可发布的向量")
        revision = int(self.setting("revision"))
        current = self.snapshot_manifest()
        if current and current["revision"] == revision:
            return current
        snapshots = self.directory / "snapshots"
        destination = snapshots / f"r{revision}_{uuid.uuid4().hex[:8]}"
        destination.mkdir(parents=True)
        vectors = np.lib.format.open_memmap(destination / "vectors.npy", mode="w+", dtype=np.float32, shape=(count, 1024))
        ids = np.lib.format.open_memmap(destination / "ids.npy", mode="w+", dtype=np.int64, shape=(count,))
        cursor, offset = self.db.execute("SELECT id,vector FROM chunks ORDER BY id"), 0
        while batch := cursor.fetchmany(2048):
            values = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in batch])
            if values.shape[1] != 1024 or not np.isfinite(values).all():
                raise RuntimeError("持久化向量损坏")
            vectors[offset:offset + len(batch)] = values
            ids[offset:offset + len(batch)] = [row["id"] for row in batch]
            offset += len(batch)
        vectors.flush()
        ids.flush()
        del vectors, ids
        manifest = {"revision": revision, "count": count, "dimension": 1024, "dtype": "float32",
                    "directory": destination.relative_to(self.directory).as_posix(), "published_at": utc_now()}
        atomic_json(destination / "manifest.json", manifest)
        atomic_json(self.directory / "current.json", manifest)
        # 只清理本程序生成、位于 snapshots 下且有 manifest 的旧快照，保留最近两份。
        valid = [p for p in snapshots.iterdir() if p.is_dir() and (p / "manifest.json").is_file()]
        for old in sorted(valid, key=lambda p: p.stat().st_mtime, reverse=True)[2:]:
            resolved = old.resolve()
            if resolved.parent == snapshots.resolve() and resolved.is_relative_to(self.directory.resolve()):
                try:
                    shutil.rmtree(resolved)
                except OSError:
                    pass  # Windows 上正在被读者 mmap 的旧快照稍后再清理。
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")
        return manifest

    def search(self, vector, top_k=5):
        manifest = self.snapshot_manifest()
        if not manifest or manifest["revision"] != int(self.setting("revision")):
            raise RuntimeError("索引尚未发布或快照过期，请运行 refresh")
        directory = (self.directory / manifest["directory"]).resolve()
        if not directory.is_relative_to(self.directory.resolve()):
            raise ValueError("快照目录越界")
        matrix = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        ids = np.load(directory / "ids.npy", mmap_mode="r", allow_pickle=False)
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape != (1024,) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("查询向量无效")
        query /= np.linalg.norm(query)
        candidates = []
        k = min(max(int(top_k), 1), len(matrix))
        # 分块读取避免额外复制整个库；操作系统会用空闲内存缓存常用向量页。
        for start in range(0, len(matrix), 16384):
            scores = matrix[start:start + 16384] @ query
            chosen = np.argpartition(-scores, min(k, len(scores)) - 1)[:k]
            candidates.extend((float(scores[i]), int(ids[start + i])) for i in chosen)
        selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:k]
        results = []
        for score, row_id in selected:
            row = self.db.execute("SELECT data,doc_id FROM chunks WHERE id=?", (row_id,)).fetchone()
            if row is None:
                raise RuntimeError("建库过程中索引已改变，请在发布完成后重试")
            chunk = json.loads(row["data"])
            document = self.document(row["doc_id"])
            aliases = [dict(a) for a in self.db.execute("SELECT doc_id,source_file FROM documents WHERE alias_of=?", (row["doc_id"],))]
            results.append({"score": score, "title": document["title"], "aliases": aliases, **chunk})
        return results
