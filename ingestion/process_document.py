"""单文档入口。示例：激活 Python 环境 后运行：

python rag_library.py process --doc-id P0011 --preview-pages 1 3 10

输出目录：RAG/extracted、RAG/cleaned、RAG/chunks；处理报告存放在 metadata。
原始 metadata 和 PDF 不修改。本步骤不会生成向量，也不会处理其他论文。
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from ingestion.preprocessing.chunking import chunk_document, load_tokenizer
from ingestion.preprocessing.cleaning import clean_document
from rag_core.config import ProcessingConfig, Workspace, default_root
from ingestion.preprocessing.pdf_extract import extract_pdf, render_previews
from rag_core.io import file_sha256, atomic_json as write_json
from ingestion.preprocessing.validation import validate_document


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True, help="例如 P0011；必须有对应 metadata JSON")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--model-dir", type=Path, help="模型/Tokenizer 的本地目录；默认查找 RAG/models")
    defaults = ProcessingConfig()
    parser.add_argument("--target-tokens", type=int, default=defaults.target_tokens)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--overlap-tokens", type=int, default=defaults.overlap_tokens)
    parser.add_argument("--preview-pages", nargs="*", type=int, default=[])
    args = parser.parse_args(argv)
    if not re.fullmatch(r"P\d{4,}", args.doc_id):
        parser.error("doc-id 应为 P 加至少四位数字")
    workspace = Workspace(args.root.resolve(), args.model_dir)
    config = ProcessingConfig(args.target_tokens, args.max_tokens, args.overlap_tokens)
    config.validate()
    meta_path = workspace.rag / "metadata" / f"{args.doc_id}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pdf_path = (workspace.rag / meta["file"]).resolve()
    if not pdf_path.is_relative_to((workspace.rag / "PDF").resolve()) or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("metadata.file 必须指向 RAG/PDF 内的 PDF 文件")
    before = file_sha256(pdf_path)
    if meta.get("sha256") and before != meta["sha256"]:
        raise ValueError("PDF 与原 metadata 中的 SHA256 不一致，先核对文件版本")
    title = meta.get("title") or pdf_path.stem
    print(f"1/4 提取 {args.doc_id}: {title}", flush=True)
    raw = extract_pdf(pdf_path, args.doc_id, pdf_path.relative_to(workspace.rag).as_posix())
    print(f"2/4 清洗 {raw['page_count']} 页，保留原始块与坐标", flush=True)
    cleaned = clean_document(raw, title)
    print("3/4 使用本地 Qwen Tokenizer 切块（不加载模型权重）", flush=True)
    tokenizer = load_tokenizer(workspace.tokenizer)
    chunks = chunk_document(cleaned, tokenizer, config)
    print("4/4 校验字符覆盖、来源引用、章节关联及 token 上限", flush=True)
    report = validate_document(raw, cleaned, chunks, tokenizer, config)
    report["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
    # 同时记录脚本哈希与依赖版本，便于之后审查“同样的配置为何结果变化”。
    report["code_sha256"] = {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("preprocessing/*.py"))}
    report["code_sha256"]["process_document.py"] = file_sha256(Path(__file__))
    import importlib.metadata
    report["dependencies"] = {name: importlib.metadata.version(name) for name in ["PyMuPDF", "transformers", "tokenizers"]}
    report["source_unchanged"] = file_sha256(pdf_path) == before
    if not report["source_unchanged"]:
        raise RuntimeError("原 PDF 在处理过程中发生变化")
    paths = workspace.outputs(args.doc_id)
    report["output_files"] = {key: str(path.relative_to(workspace.root)) for key, path in paths.items()}
    report["previews"] = render_previews(pdf_path, workspace.root / "tmp" / "pdfs" / args.doc_id, args.preview_pages)
    # 失败时仍保存原始提取、清洗和报告，方便定位；不发布无效的 chunks 文件。
    write_json(paths["extracted"], raw)
    write_json(paths["cleaned"], cleaned)
    if report["structural_checks_passed"]:
        paths["chunks"].parent.mkdir(parents=True, exist_ok=True)
        temp = paths["chunks"].with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        temp.replace(paths["chunks"])
    write_json(paths["report"], report)
    print(json.dumps({key: report[key] for key in ["status", "page_count", "chunk_count", "retrieval_eligible_chunks", "token_counts", "errors", "warnings"]}, ensure_ascii=False, indent=2))
    print(f"报告：{paths['report']}")
    return 0 if report["structural_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
