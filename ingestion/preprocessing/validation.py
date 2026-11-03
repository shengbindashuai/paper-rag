"""检查可程序验证的性质；不把这些检查冒充论文语义或数学公式的正确性证明。"""
from collections import Counter, defaultdict

from .chunking import token_count


def validate_document(raw: dict, cleaned: dict, chunks: list[dict], tokenizer, config) -> dict:
    errors, warnings = [], []
    raw_blocks = {b["block_id"]: (p, b) for p in raw["pages"] for b in p["blocks"]}
    segments = {s["segment_id"]: s for s in cleaned["segments"]}
    kept = [a["block_id"] for s in cleaned["segments"] for a in s["sources"]]
    removed = [b["block_id"] for b in cleaned["removed_blocks"]]
    if set(kept) & set(removed) or set(kept + removed) != set(raw_blocks):
        errors.append("原始块未完整归入保留/移除日志，或两类发生重叠")
    if len(kept) != len(set(kept)):
        errors.append("同一个原始块被重复放入多个清洗段")
    if not chunks or len({c["chunk_id"] for c in chunks}) != len(chunks):
        errors.append("片段为空或 ID 重复")
    valid_parents = {s["parent_id"] for s in cleaned["sections"]}
    covered = defaultdict(list)
    for chunk in chunks:
        if chunk["parent_id"] not in valid_parents:
            errors.append(f"父章节不存在：{chunk['chunk_id']}")
        actual = token_count(tokenizer, chunk["embedding_text"])
        if actual != chunk["token_count"] or actual > config.max_tokens:
            errors.append(f"token 计数或长度上限异常：{chunk['chunk_id']}")
        for span in chunk["source_spans"]:
            segment = segments[span["segment_id"]]
            start, end = span["segment_char_start"], span["segment_char_end"]
            if segment["text"][start:end] != span["text"]:
                errors.append(f"片段不能回溯到清洗段的字符区间：{chunk['chunk_id']}")
            if not span["pdf_blocks"]:
                errors.append(f"片段缺少 PDF 定位：{chunk['chunk_id']}")
            covered[segment["segment_id"]].append((start, end))
            for anchor in span["pdf_blocks"]:
                if anchor["block_id"] not in raw_blocks or not 1 <= anchor["page"] <= raw["page_count"]:
                    errors.append(f"无效的 PDF 定位：{chunk['chunk_id']}")
                else:
                    source_page, source_block = raw_blocks[anchor["block_id"]]
                    if anchor["page"] != source_page["page"] or anchor["bbox"] != source_block["bbox"]:
                        errors.append(f"PDF 页码或坐标不匹配：{chunk['chunk_id']}")
        anchored_pages = sorted({a["page"] for s in chunk["source_spans"] for a in s["pdf_blocks"]})
        if chunk["pages"] != anchored_pages:
            errors.append(f"片段页码与实际定位不匹配：{chunk['chunk_id']}")
        if chunk["retrieval_eligible"] and {"unmapped_pdf_glyph", "layout_requires_visual_review", "inline_math_layout_requires_review"}.intersection(chunk["quality_flags"]):
            errors.append(f"高风险版面内容被错误标为直接入库：{chunk['chunk_id']}")
    for segment in cleaned["segments"]:
        if segment["kind"] in {"heading", "front_matter"} or not segment["text"]:
            continue
        cursor = 0
        for start, end in sorted(covered[segment["segment_id"]]):
            if start > cursor and segment["text"][cursor:start].strip():
                errors.append(f"切块遗漏了正文字符：{segment['segment_id']}")
                break
            cursor = max(cursor, end)
        if segment["text"][cursor:].strip():
            errors.append(f"切块遗漏了段落末尾：{segment['segment_id']}")
    review = [s for s in cleaned["segments"] if {"layout_requires_visual_review", "unmapped_pdf_glyph", "inline_math_layout_requires_review"}.intersection(s["quality_flags"])]
    if review:
        warnings.append("公式、表格、图中刻度或伪代码仅保留文本与位置；标记内容暂不直接用于检索")
    if cleaned["layout_warnings"]:
        warnings.append("检测到可能的多栏页面，需人工检查阅读顺序")
    sizes = sorted(c["token_count"] for c in chunks)
    return {
        "status": "failed" if errors else "processed_with_review_flags" if warnings else "processed",
        "structural_checks_passed": not errors, "errors": errors, "warnings": warnings,
        "doc_id": raw["doc_id"], "source_sha256": raw["source_sha256"],
        "page_count": raw["page_count"], "raw_block_count": len(raw_blocks),
        "removed_block_counts": dict(Counter(b["reason"] for b in cleaned["removed_blocks"])),
        "segment_counts": dict(Counter(s["kind"] for s in cleaned["segments"])),
        "section_count": len(cleaned["sections"]), "chunk_count": len(chunks),
        "retrieval_eligible_chunks": sum(c["retrieval_eligible"] for c in chunks),
        "token_counts": {"min": min(sizes) if sizes else 0, "median": sizes[len(sizes) // 2] if sizes else 0,
                         "max": max(sizes) if sizes else 0},
        "cleaning_actions": dict(Counter(change["action"] for s in cleaned["segments"] for change in s["changes"])),
        "review_segments": [{"segment_id": s["segment_id"], "kind": s["kind"],
                             "pages": sorted({a["page"] for a in s["sources"]}),
                             "quality_flags": s["quality_flags"]} for s in review],
        "config": config.to_dict(),
    }
