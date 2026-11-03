"""search 单次调用；serve 为复用模型的 JSONL 长驻进程（尚不是 MCP server）。

python -m retrieval.cli search --query "PPO 怎样利用采样数据？"
python -m retrieval.cli serve
python -m retrieval.cli tools

serve 每行输入 {"id":1,"name":"paper_search","arguments":{"query":"..."}}。
"""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from retrieval.config import RetrievalConfig
from rag_core.config import default_root, config_file
from retrieval.schemas import SearchRequest
from retrieval.tool_api import definitions, RetrievalToolAPI


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--experiment-index", type=Path, help="显式使用四篇论文实验向量，仅供试跑")
    parser.add_argument("--allow-read-root", type=Path, action="append", default=[])
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--term-scope", dest="term_scopes", action="append", help="可重复的术语适用领域，如 reinforcement_learning")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--candidate-k", type=int, default=SearchRequest.model_fields["candidate_k"].default)
    search.add_argument("--no-rerank", action="store_true")
    search.add_argument("--quality-gate", action=argparse.BooleanOptionalAction,
                        default=SearchRequest.model_fields["quality_gate"].default,
                        help="DeepSeek 质量门控默认开启；--no-quality-gate 显式关闭")
    search.add_argument("--context-budget-tokens", type=int, default=2048)
    commands.add_parser("serve")
    commands.add_parser("tools")
    args = parser.parse_args()
    if args.command == "tools":
        print(json.dumps(definitions(), ensure_ascii=False, indent=2))
        return
    config = RetrievalConfig.load(args.config or config_file(args.root, "retrieval.json"), args.root)
    if args.index_dir:
        config.index_dir = str(args.index_dir.resolve())
    if args.experiment_index:
        config.index_kind, config.index_dir = "experiment", str(args.experiment_index.resolve())
    from retrieval.service import RetrievalService
    service = RetrievalService(config)
    jobs = None
    try:
        service.start()
        if args.command == "search":
            request = SearchRequest(query=args.query, top_k=args.top_k, candidate_k=args.candidate_k,
                                    rerank=not args.no_rerank, quality_gate=args.quality_gate,
                                    context_budget_tokens=args.context_budget_tokens, term_scopes=args.term_scopes)
            print(json.dumps(service.search(request), ensure_ascii=False, indent=2))
        else:
            from knowledge.jobs import IngestJobs
            if config.index_kind == "library":
                jobs = IngestJobs(service, allowed_roots=[args.root.resolve(), *args.allow_read_root])
            api = RetrievalToolAPI(service, jobs)
            print("Retrieval JSONL ready; 128 tier resident.", file=sys.stderr, flush=True)
            for line in sys.stdin:
                identifier = None
                try:
                    message = json.loads(line)
                    identifier = message.get("id")
                    response = {"id": identifier, "result": api.call(message["name"], message.get("arguments", {}))}
                except Exception as exc:
                    response = {"id": identifier, "error": {"type": type(exc).__name__, "message": str(exc)}}
                print(json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        if jobs:
            jobs.close()
        service.close()


if __name__ == "__main__":
    main()
