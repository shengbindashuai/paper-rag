"""SQLite FTS5/BM25 辅助索引；增量跟随正式库修订，不改变原文或向量。"""
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from rag_core.storage import writer_lock


STOPWORDS = set("how what why when where does do is are the a an of to in for with and or on through".split())
STOPWORDS.update({"如何", "怎样", "什么", "为何", "进行", "通过", "使用", "一个", "哪些", "是否"})


def keyword_tokens(text):
    """英文词/缩写与中文二元词；只处理搜索副本，原文及来源不变。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    tokens = []
    for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text):
        if "\u3400" <= token[0] <= "\u9fff":
            parts = [token] if len(token) <= 2 else [token[i:i+2] for i in range(len(token) - 1)]
        else:
            parts = [token]
        tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


class KeywordIndex:
    def __init__(self, directory: Path):
        self.directory = directory / "keyword_index"
        self.path = self.directory / "index.sqlite3"

    def ensure(self, source, revision):
        """source 必须处于读事务；辅助库和修订号在同一事务中发布。"""
        with writer_lock(self.directory):
            db = sqlite3.connect(self.path, timeout=30)
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)")
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(doc_id UNINDEXED, chunk_id UNINDEXED, title, section, body, tokenize='unicode61')")
                old = dict(db.execute("SELECT key,value FROM state"))
                source_path = next(row[2] for row in source.execute("PRAGMA database_list") if row[1] == "main")
                source_identity = str(Path(source_path).stat().st_ino) if source_path else "memory"
                if (old.get("revision") == str(revision) and old.get("source_identity") == source_identity
                        and old.get("format") == "keyword-bigram-v1"):
                    return
                with db:
                    source_ids = {row[0] for row in source.execute("SELECT id FROM chunks")}
                    indexed_ids = {row[0] for row in db.execute("SELECT rowid FROM passages")}
                    removed = indexed_ids - source_ids
                    db.executemany("DELETE FROM passages WHERE rowid=?", [(i,) for i in removed])
                    last_id = int(old.get("max_id", "0"))
                    # AUTOINCREMENT 为新增/替换片段分配新 ID；重建库或身份不一致时重建副本。
                    maximum = max(source_ids, default=0)
                    rebuild = (maximum < last_id or old.get("format") != "keyword-bigram-v1"
                               or old.get("source_identity") != source_identity
                               or any(i <= last_id for i in source_ids - indexed_ids))
                    if rebuild:
                        db.execute("DELETE FROM passages")
                        last_id = 0
                    rows = source.execute("SELECT c.id,c.doc_id,c.chunk_id,c.data,d.title FROM chunks c LEFT JOIN documents d ON d.doc_id=c.doc_id WHERE c.id>? ORDER BY c.id", (last_id,))
                    for batch in iter(lambda: rows.fetchmany(512), []):
                        values = []
                        for row in batch:
                            chunk = json.loads(row[3])
                            values.append((row[0], row[1], row[2], " ".join(keyword_tokens(row[4] or "")),
                                           " ".join(keyword_tokens(chunk.get("section", ""))),
                                           " ".join(keyword_tokens(chunk["text"]))))
                        db.executemany("INSERT INTO passages(rowid,doc_id,chunk_id,title,section,body) VALUES (?,?,?,?,?,?)", values)
                    db.executemany("INSERT OR REPLACE INTO state VALUES (?,?)", [
                        ("revision", str(revision)), ("max_id", str(maximum)), ("format", "keyword-bigram-v1"),
                        ("source_identity", source_identity)])
            finally:
                db.close()

    def search(self, source, revision, query, top_k, doc_ids=None):
        tokens = list(dict.fromkeys(keyword_tokens(query)))[:32]
        if not tokens:
            return [], {"status": "no_search_terms", "count": 0, "terms": []}
        self.ensure(source, revision)
        expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            recorded = db.execute("SELECT value FROM state WHERE key='revision'").fetchone()
            if recorded is None or int(recorded[0]) != revision:
                raise RuntimeError("关键词索引修订变化，请重试")
            sql = "SELECT rowid,bm25(passages,0,0,3,1.5,1) AS score FROM passages WHERE passages MATCH ?"
            args = [expression]
            if doc_ids:
                sql += " AND doc_id IN (" + ",".join("?" for _ in doc_ids) + ")"
                args.extend(sorted(doc_ids))
            sql += " ORDER BY score,rowid LIMIT ?"
            rows = list(db.execute(sql, (*args, top_k)))
            return rows, {"status": "completed", "count": len(rows), "terms": tokens,
                          "snapshot_revision": revision, "score_type": "fts5_bm25_lower_is_better"}
        finally:
            db.close()
