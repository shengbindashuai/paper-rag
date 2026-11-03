"""扫描、哈希去重和新增文档注册。原有 PDF 与 metadata 不覆盖。"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil

from rag_core.storage import atomic_json, utc_now
from ingestion.preprocessing.pdf_extract import file_sha256


def scan_library(root: Path):
    rag = root / "RAG"
    metadata, unavailable = {}, []
    for path in sorted((rag / "metadata").glob("P*.json")):
        if ".processing" in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("file"):
            unavailable.append(path.stem)
            continue
        target = (rag / data["file"]).resolve()
        if not target.is_relative_to((rag / "PDF").resolve()):
            raise ValueError(f"metadata 指向 PDF 目录之外：{path}")
        if target.is_file():
            metadata[target] = (path.stem, data)
        else:
            unavailable.append(path.stem)
    paths = sorted((rag / "PDF").rglob("*.pdf"))
    notes = rag / "阅读摘录"
    if notes.is_dir():
        paths += sorted(p for p in notes.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"})
    def inspect(path):
        path = path.resolve()
        sha = file_sha256(path)
        if path.suffix.lower() != ".pdf":
            import hashlib
            relative = path.relative_to(rag).as_posix()
            from knowledge.sources import source_metadata, note_id
            managed = source_metadata(root, note_id(relative), sha)
            linked = re.match(r"(P\d+)", path.name)
            return {"doc_id": "N" + hashlib.sha256(relative.encode()).hexdigest()[:16], "path": str(path),
                    "source_file": relative, "sha256": sha, "title": managed.get("title", path.stem), "bytes": path.stat().st_size,
                    "pages": 0, "metadata_hash_mismatch": False, "source_type": "reading_excerpt",
                    "linked_doc_id": linked[1] if linked else None}
        doc_id, meta = metadata.get(path, ("L" + sha[:16], {}))
        return {"doc_id": doc_id, "path": str(path), "source_file": path.relative_to(rag).as_posix(),
                "sha256": sha, "title": meta.get("title") or path.stem, "bytes": path.stat().st_size,
                "pages": meta.get("pages", 0) or 0,
                "metadata_hash_mismatch": bool(meta.get("sha256") and meta["sha256"] != sha)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        items = list(pool.map(inspect, paths))
    # 同名编号的不同文件不能覆盖彼此；没有 metadata 的文件已用内容哈希命名。
    seen = {}
    for item in items:
        if item["doc_id"] in seen:
            item["doc_id"] += "_" + item["sha256"][:8]
        seen[item["doc_id"]] = True
    return sorted(items, key=lambda x: x["doc_id"]), unavailable


def register_pdf(root: Path, pdf: Path, title: str | None = None):
    """Python 入库接口的注册部分；外部文件只读取，复制到论文库后处理。"""
    pdf = pdf.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("需要存在的 PDF 文件")
    rag = root / "RAG"
    sha = file_sha256(pdf)
    maximum = 0
    for path in (rag / "metadata").glob("P*.json"):
        match = re.fullmatch(r"P(\d+)", path.stem)
        if not match:
            continue
        maximum = max(maximum, int(match[1]))
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("sha256") == sha and data.get("file"):
            existing = (rag / data["file"]).resolve()
            if existing.is_file() and file_sha256(existing) == sha:
                return path.stem
    doc_id = f"P{maximum + 1:04}"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title or pdf.stem)[:110].strip(" .") or "paper"
    destination = rag / "PDF" / f"{doc_id}__{name}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".pdf.tmp")
    shutil.copyfile(pdf, temporary)
    if file_sha256(temporary) != sha:
        raise RuntimeError("复制后校验失败")
    temporary.replace(destination)
    atomic_json(rag / "metadata" / f"{doc_id}.json", {
        "archive_id": doc_id, "title": title or pdf.stem, "file": destination.relative_to(rag).as_posix(),
        "sha256": sha, "bytes": destination.stat().st_size, "status": "pdf_saved", "registered_at": utc_now(),
        "registered_from": str(pdf)})
    return doc_id
