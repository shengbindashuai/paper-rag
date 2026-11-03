"""阅读摘录适配：保留 Markdown 行号，不伪造 PDF 页码或坐标。"""
import re


def extract_note(item):
    from pathlib import Path
    original = Path(item["path"]).read_text(encoding="utf-8-sig")
    lines = original.splitlines()
    title = next((line.lstrip('# ').strip() for line in lines if line.startswith('# ')), item["title"])
    parent, section = item["doc_id"] + ":root", ["阅读摘录"]
    sections = [{"parent_id": parent, "path": section, "start_line": 1}]
    segments, pending, start = [], [], 0
    started = False
    def emit(end):
        if not pending:
            return
        text = " ".join(x.strip() for x in pending).strip()
        sid = f"{item['doc_id']}:s{len(segments)+1:04}"
        segments.append({"segment_id": sid, "kind": "paragraph", "parent_id": parent,
                         "section_path": list(section), "text": text, "retrieval_eligible": True,
                         "quality_flags": ["reading_excerpt_not_full_paper"], "changes": [],
                         "sources": [{"page": 0, "block_id": sid, "bbox": [], "line_ids": [],
                                      "start_line": start, "end_line": end,
                                      "segment_char_start": 0, "segment_char_end": len(text)}]})
        pending.clear()
    for number, line in enumerate(lines, 1):
        if re.match(r"^#{1,6}\s+\S", line):
            emit(number - 1)
            if line.startswith('# '):
                continue
            started = True
            section = ["阅读摘录", line.lstrip('# ').strip()]
            parent = f"{item['doc_id']}:section:{len(sections)}"
            sections.append({"parent_id": parent, "path": list(section), "start_line": number})
        elif not line.strip():
            emit(number - 1)
        elif started:
            if not pending:
                start = number
            pending.append(line)
    emit(len(lines))
    # 无 Markdown 二级标题的纯文本也可入库，整篇作为一个有行号的段落。
    if not segments and original.strip():
        pending.extend(lines)
        start = 1
        emit(len(lines))
    cleaned = {"doc_id": item["doc_id"], "title": title, "source_file": item["source_file"],
               "source_sha256": item["sha256"], "sections": sections, "segments": segments,
               "removed_blocks": [], "layout_warnings": [], "schema_version": 1}
    return {"source_file": item["source_file"], "source_sha256": item["sha256"], "text": original,
            "line_count": len(lines), "source_type": "reading_excerpt"}, cleaned
