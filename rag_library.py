"""本地论文向量库命令入口。

全库/续建：python rag_library.py build
新增 PDF：python rag_library.py ingest --pdf "D:/papers/new.pdf"
已有编号：python rag_library.py build --doc-id P0011
查看状态：python rag_library.py status
查询示例：python rag_library.py search --query "PPO 怎样重复利用数据？"
刷新快照：python rag_library.py refresh

PDF 原文保留；审计和索引放在 RAG/library。GPU 建库，NPU 在线编码查询。
"""
import argparse
import json
from pathlib import Path
import sys
from retrieval.schemas import SearchRequest
from rag_core.config import default_root


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--index-dir", type=Path, help="独立索引目录；默认 RAG/library")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["build", "ingest"]:
        command = commands.add_parser(name)
        command.add_argument("--workers", type=int, help="CPU 进程数；自动按核心数和可用内存选择")
        command.add_argument("--max-batch-size", type=int, default=128)
        command.add_argument("--retry-failed", action="store_true")
        if name == "build":
            command.add_argument("--doc-id", nargs="+")
            command.add_argument("--limit", type=int, help="仅用于小规模试跑")
        else:
            command.add_argument("--pdf", nargs="+", type=Path, required=True)
            command.add_argument("--title", help="单份新增文档的可选标题")
    commands.add_parser("status")
    commands.add_parser("refresh")
    commands.add_parser("audit", help="核对已发布快照、来源哈希和待复核文件")
    single = commands.add_parser("process", help="单文档预处理学习入口，不写向量库")
    single.add_argument("--doc-id", required=True)
    single.add_argument("--preview-pages", nargs="*", type=int, default=[])
    query = commands.add_parser("search")
    query.add_argument("--query", required=True)
    query.add_argument("--term-scope", dest="term_scopes", action="append")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--candidate-k", type=int, default=SearchRequest.model_fields["candidate_k"].default)
    query.add_argument("--no-rerank", action="store_true")
    query.add_argument("--quality-gate", action=argparse.BooleanOptionalAction,
                       default=SearchRequest.model_fields["quality_gate"].default,
                       help="DeepSeek 质量门控默认开启；--no-quality-gate 显式关闭")
    args = parser.parse_args()
    root = args.root.resolve()
    directory = (args.index_dir or root / "RAG" / "library").resolve()
    if args.command == "audit":
        from ingestion.audit import audit_library
        result = audit_library(directory)
        print(json.dumps({k: v for k, v in result.items() if k != "exceptions"}, ensure_ascii=False, indent=2))
        return
    if args.command == "process":
        from ingestion.process_document import main as process_main
        return process_main(["--root", str(root), "--doc-id", args.doc_id,
                             "--preview-pages", *map(str, args.preview_pages)])
    if args.command in {"build", "ingest"}:
        from ingestion.pipeline import build_library, ingest_documents
        options = {"index_dir": directory, "workers": args.workers, "max_batch_size": args.max_batch_size,
                   "retry_failed": args.retry_failed}
        if args.command == "build":
            build_library(root, doc_ids=args.doc_id, limit=args.limit, **options)
        else:
            if args.title and len(args.pdf) != 1:
                parser.error("--title 只支持单份 PDF")
            ingest_documents(root, args.pdf, title=args.title, **options)
        return
    from rag_core.storage import LibraryStore, writer_lock
    if not (directory / "library.sqlite3").is_file():
        parser.error("索引不存在，请先运行 build")
    store = LibraryStore(directory)
    try:
        if args.command == "status":
            print(json.dumps(store.status(), ensure_ascii=False, indent=2))
        elif args.command == "refresh":
            with writer_lock(directory):
                print(json.dumps(store.refresh_snapshot(), ensure_ascii=False, indent=2))
        else:
            from retrieval.query import PaperRetriever
            retriever = PaperRetriever(root, directory)
            try:
                result = retriever.search(args.query, args.top_k, candidate_k=args.candidate_k,
                                          rerank=not args.no_rerank, quality_gate=args.quality_gate, term_scopes=args.term_scopes)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            finally:
                retriever.close()
    finally:
        store.close()


if __name__ == "__main__":
    main()
