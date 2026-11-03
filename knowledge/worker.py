"""单次入库子进程，隔离 GPU 权重和 Windows spawn 入口。"""
import argparse
import json
from pathlib import Path

from rag_core.io import atomic_json


def execute(root, directory, payload):
    from ingestion.inventory import register_pdf
    from ingestion.pipeline import build_library
    from rag_core.storage import LibraryStore, writer_lock
    from knowledge.sources import register_text
    with writer_lock(directory):
        if payload["source_type"] == "paper":
            doc_id = register_pdf(root, Path(payload["path"]), title=payload.get("title"))
        else:
            doc_id = register_text(root, payload)
        # 持久化注册结果，即使后续失败也能回溯到原件。
        result = {"doc_id": doc_id, "indexed": False}
        atomic_json(Path(payload["result_path"]), result)
        run = build_library(root, index_dir=directory, doc_ids=[doc_id], retry_failed=True, _locked=True)
        store = LibraryStore(directory)
        try:
            document = store.document(doc_id)
            canonical = store.document(document["alias_of"]) if document and document["alias_of"] else document
            snapshot = store.snapshot_manifest()
            result.update(document=document, snapshot=snapshot)
            if not canonical or canonical["status"] != "indexed" or not snapshot:
                raise RuntimeError("文档未成功编码或未发布，查看 result.document 和入库日志")
            if snapshot["revision"] != run["final_status"]["revision"]:
                raise RuntimeError("快照尚未发布到当前修订")
            result["indexed"] = True
        finally:
            atomic_json(Path(payload["result_path"]), result)
            store.close()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    execute(args.root, args.index_dir, json.loads(args.payload.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
