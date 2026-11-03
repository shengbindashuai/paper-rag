"""将新文档转换为现有摘录格式；原件和角色信息单独归档。"""
import hashlib
import json
from pathlib import Path

from rag_core.io import atomic_json, utc_now


def note_id(relative):
    return "N" + hashlib.sha256(relative.encode()).hexdigest()[:16]


def source_metadata(root, doc_id, sha):
    path = root / "RAG" / "knowledge_sources" / f"{doc_id}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("sha256") == sha else {}


def register_text(root, payload):
    kind = payload["source_type"]
    if kind == "conversation":
        body = "\n\n".join(
            f"### {i}. {m['role']}" + (f" — {m['timestamp']}" if m.get('timestamp') else "")
            + "\n\n" + m["content"] for i, m in enumerate(payload["messages"], 1))
    else:
        body = payload["content"]
    # 首个二级标题使现有摘录解析器完整保留正文前言。
    text = f"# {payload['title'].replace(chr(10), ' ').replace(chr(13), ' ')}\n\n## 正文\n\n{body}\n"
    stable = {k: v for k, v in payload.items() if k != "result_path"}
    identity = hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    relative = f"阅读摘录/知识库写入/{identity}.md"
    destination = root / "RAG" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(text.encode()).hexdigest()
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
        raise ValueError("已归档正文被修改，拒绝覆盖")
    if not destination.exists():
        temporary = destination.with_suffix(".md.tmp")
        temporary.write_text(text, encoding="utf-8", newline="")
        temporary.replace(destination)
    doc_id = note_id(relative)
    metadata = {"doc_id": doc_id, "title": payload["title"], "source_type": kind,
                "sha256": sha, "source_file": relative, "registered_at": utc_now(),
                "source_url": payload.get("source_url"), "original_path": payload.get("original_path"),
                "conversation_id": payload.get("conversation_id")}
    if kind == "conversation":
        metadata["messages"] = payload["messages"]
    if payload.get("original_path"):
        original = Path(payload["original_path"])
        archive = destination.with_suffix(original.suffix.lower() + ".original")
        if not archive.exists():
            import shutil
            shutil.copyfile(original, archive)
        metadata["original_archive"] = archive.relative_to(root / "RAG").as_posix()
    atomic_json(root / "RAG" / "knowledge_sources" / f"{doc_id}.json", metadata)
    return doc_id


def extract_document(path):
    if path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8-sig")
    elif path.suffix.lower() == ".docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        document = Document(path)
        pieces = []
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                pieces.append(Paragraph(child, document).text)
            elif child.tag.endswith("}tbl"):
                pieces.extend(" | ".join(cell.text for cell in row.cells) for row in Table(child, document).rows)
        text = "\n\n".join(pieces)
    else:
        raise ValueError("文档支持 .md/.txt/.docx；PDF 请使用 paper_ingest")
    if not text.strip() or len(text) > 1000000:
        raise ValueError("文档必须包含正文，且不能超过 1000000 字符")
    return text
