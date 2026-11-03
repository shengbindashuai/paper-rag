"""CPU 子进程：独立解析 PDF 并写压缩审计文件，不加载模型权重。

返回小型结果，不把整篇 PDF 的字形/坐标通过进程队列反复复制。
report.json 最后发布，是这份预处理缓存已经完整写完的标志。
"""
import gzip
import json
from pathlib import Path
import time
import traceback

from ingestion.preprocessing.chunking import chunk_document, load_tokenizer, token_count
from ingestion.preprocessing.cleaning import clean_document
from rag_core.config import ProcessingConfig
from ingestion.preprocessing.quality import order_columns, screen_segments
from rag_core.storage import atomic_json
from ingestion.preprocessing.pdf_extract import extract_pdf
from ingestion.preprocessing.validation import validate_document


_TOKENIZER = None


def initialize_worker(model_dir):
    global _TOKENIZER
    _TOKENIZER = load_tokenizer(Path(model_dir))


def write_gzip(path, data, jsonl=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
        if jsonl:
            for item in data:
                stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def load_chunks(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def prepare_document(item, directory, config_dict, signature):
    started = time.perf_counter()
    destination = Path(directory) / "processed" / item["doc_id"]
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    chunk_path = destination / "chunks.jsonl.gz"
    try:
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if (previous.get("signature") == signature and previous.get("source_sha256") == item["sha256"]
                    and previous.get("original_title") == item["title"]
                    and all((destination / n).is_file() for n in ["raw.json.gz", "cleaned.json.gz", "chunks.jsonl.gz"])):
                return {"item": item, "status": previous["library_status"], "chunks_path": str(chunk_path),
                        "chunk_count": previous["indexed_chunk_count"], "report_path": str(report_path), "cached": True}
        config = ProcessingConfig(**config_dict)
        note = item.get("source_type") == "reading_excerpt"
        if note:
            from .preprocessing.notes import extract_note
            raw, cleaned = extract_note(item)
            layout = {"two_column_reordered_pages": [], "mixed_column_block_ids": []}
        else:
            raw = extract_pdf(Path(item["path"]), item["doc_id"], item["source_file"])
        if raw["source_sha256"] != item["sha256"]:
            raise ValueError("PDF 在扫描后发生变化；下次重跑时重新扫描")
        if not note:
            ordered, layout = order_columns(raw)
            cleaned = clean_document(ordered, item["title"])
        # 保留完整标题；只压缩用于 embedding 的超长标题，避免挤占全部正文预算。
        cleaned["original_title"] = item["title"]
        while token_count(_TOKENIZER, cleaned["title"]) > 96:
            cleaned["title"] = cleaned["title"][:int(len(cleaned["title"]) * .85)].rstrip()
        exclusions = screen_segments(cleaned, _TOKENIZER, layout)
        all_chunks = chunk_document(cleaned, _TOKENIZER, config)
        if note:
            mapping = {s["segment_id"]: s["text"] for s in cleaned["segments"]}
            for chunk in all_chunks:
                chunk.update(pages=[], source_type="reading_excerpt", linked_doc_id=item.get("linked_doc_id"))
                for span in chunk["source_spans"]:
                    if mapping[span["segment_id"]][span["segment_char_start"]:span["segment_char_end"]] != span["text"]:
                        raise ValueError("摘录字符区间无法回溯")
                    anchors = span.pop("pdf_blocks")
                    span["text_sources"] = [{k: a[k] for k in ["start_line", "end_line"]} for a in anchors]
            checks = {"structural_checks_passed": True, "warnings": ["阅读摘录不是完整论文，使用文本行号溯源"], "errors": []}
        elif all_chunks:
            checks = validate_document(raw, cleaned, all_chunks, _TOKENIZER, config)
            if not checks["structural_checks_passed"]:
                raise ValueError("; ".join(checks["errors"][:5]))
        else:
            checks = {"structural_checks_passed": True, "warnings": ["未识别出可分块正文"], "errors": []}
        chunks = [c for c in all_chunks if c["retrieval_eligible"]]
        status = "prepared" if chunks else "quarantined"
        report = {**checks, "signature": signature, "source_sha256": item["sha256"],
                  "original_title": item["title"], "embedding_title": cleaned["title"],
                  "layout": layout, "exclusion_counts": exclusions, "library_status": status,
                  "total_chunk_count": len(all_chunks), "indexed_chunk_count": len(chunks),
                  "processing_seconds": time.perf_counter() - started,
                  "page_count": raw.get("page_count", 0), "config": config_dict}
        write_gzip(destination / "raw.json.gz", raw)
        write_gzip(destination / "cleaned.json.gz", cleaned)
        # 完整分块也保留，以便审查被排除的内容；入库时再次按 eligibility 过滤。
        write_gzip(chunk_path, all_chunks, jsonl=True)
        atomic_json(report_path, report)
        return {"item": item, "status": status, "chunks_path": str(chunk_path), "chunk_count": len(chunks),
                "report_path": str(report_path), "processing_seconds": report["processing_seconds"], "cached": False}
    except Exception as exc:
        error = {"item": item, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc(), "processing_seconds": time.perf_counter() - started}
        atomic_json(destination / "error.json", error)
        return error
