"""建库后核对来源、向量快照和失败清单；输出一份 JSON 和 HTML。"""
from collections import Counter
from html import escape
import json
from pathlib import Path
import sqlite3

import numpy as np

from rag_core.io import atomic_json, file_sha256, utc_now


def audit_library(directory: Path, verify_sources=True):
    directory = directory.resolve()
    inventory_record = json.loads((directory / "inventory.json").read_text(encoding="utf-8"))
    inventory = inventory_record["documents"]
    database = sqlite3.connect(directory / "library.sqlite3")
    database.row_factory = sqlite3.Row
    documents = [dict(row) for row in database.execute("SELECT * FROM documents ORDER BY doc_id")]
    manifest = json.loads((directory / "current.json").read_text(encoding="utf-8"))
    revision = int(database.execute("SELECT value FROM settings WHERE key='revision'").fetchone()[0])
    if revision != manifest["revision"]:
        raise RuntimeError("建库尚未完成发布，请稍后运行 audit")
    errors = []
    if database.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        errors.append("SQLite quick_check failed")
    matrix = np.load(directory / manifest["directory"] / "vectors.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(directory / manifest["directory"] / "ids.npy", mmap_mode="r", allow_pickle=False)
    count = database.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if matrix.shape != (count, 1024) or len(ids) != count:
        errors.append("snapshot shape mismatch")
    max_norm_error = 0.
    for start in range(0, len(matrix), 4096):
        block = matrix[start:start + 4096]
        if not np.isfinite(block).all():
            errors.append("non-finite vector")
        max_norm_error = max(max_norm_error, float(np.max(np.abs(np.linalg.norm(block, axis=1) - 1))))
    if max_norm_error > 1e-4:
        errors.append("normalization error")
    if verify_sources:
        for item in inventory:
            if file_sha256(Path(item["path"])) != item["sha256"]:
                errors.append("source changed: " + item["source_file"])
    actual = {d["doc_id"] for d in documents}
    errors.extend("unaccounted source: " + item["doc_id"] for item in inventory if item["doc_id"] not in actual)
    sources = Counter("reading_excerpt" if item.get("source_type") == "reading_excerpt" else "pdf" for item in inventory)
    inventory_record.update(source_count=len(inventory), pdf_count=sources["pdf"], reading_excerpt_count=sources["reading_excerpt"])
    atomic_json(directory / "inventory.json", inventory_record)
    build_path = directory / "last_build.json"
    if build_path.exists():
        build = json.loads(build_path.read_text(encoding="utf-8"))
        if "inventory_sources" not in build:
            build["inventory_sources"] = build["inventory_pdfs"]
            build["inventory_pdfs"] = sources["pdf"]
            build["inventory_reading_excerpts"] = sources["reading_excerpt"]
            build["selected_sources"] = build.pop("selected_pdfs", len(inventory))
            atomic_json(build_path, build)
    statuses = Counter(d["status"] for d in documents)
    exceptions = [d for d in documents if d["status"] in {"failed", "quarantined", "missing"}]
    result = {"at": utc_now(), "passed": not errors, "errors": errors, "sources": dict(sources),
              "documents": dict(statuses), "chunks": count, "snapshot": manifest,
              "max_norm_error": max_norm_error, "sources_hash_verified": verify_sources,
              "exceptions": exceptions}
    atomic_json(directory / "audit.json", result)
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(d.get(k, '')))}</td>" for k in
                                   ["doc_id", "title", "status", "source_file", "error"]) + "</tr>" for d in exceptions)
    html = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>论文库建库核对</title>"
            "<style>body{font:16px/1.6 system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>"
            f"<h1>论文库建库核对</h1><p>来源：{escape(str(dict(sources)))}；状态：{escape(str(dict(statuses)))}；"
            f"向量块数：{count}；校验通过：{not errors}。</p>"
            "<p>PDF 以页码和坐标定位；阅读摘录使用文件与行号，不表示完整论文。公式、表格和复杂版面仅保留审计内容，"
            "未通过文本质量筛选的内容不进入向量库。此处的完整性校验不等价于语义质量评估。</p>"
            "<h2>失败和待复核来源</h2><table><tr><th>编号</th><th>标题</th><th>状态</th><th>文件</th><th>原因</th></tr>"
            + rows + "</table></html>")
    (directory / "report.html").write_text(html, encoding="utf-8")
    database.close()
    return result
