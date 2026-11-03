"""Download pinned official weights into a reusable local workspace."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.config import default_root
from rag_core.io import atomic_json, file_sha256


MODELS = {
    "embedding": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "reranker": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
}
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.json", "merges.txt", "generation_config.json", "LICENSE"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--local-files-only", action="store_true", help="Only use the Hugging Face cache")
    parser.add_argument("--dry-run", action="store_true", help="Show pinned revisions and paths without downloading")
    args = parser.parse_args()
    selected = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    root = args.root.resolve()
    for repo, revision in selected.values():
        directory = root / "RAG/models" / repo.split("/")[-1]
        print(json.dumps({"model": repo, "revision": revision, "destination": str(directory)}, ensure_ascii=False))
        if args.dry_run:
            continue
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, revision=revision, local_dir=directory,
                          allow_patterns=FILES, local_files_only=args.local_files_only)
        required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"}
        if not all((directory / name).is_file() for name in required):
            raise RuntimeError(f"Incomplete model download: {directory}")
        hashes = {name: file_sha256(directory / name) for name in FILES if (directory / name).is_file()}
        previous = directory / "provenance.json"
        if previous.is_file():
            recorded = json.loads(previous.read_text(encoding="utf-8"))
            if recorded.get("revision") == revision:
                recorded_hashes = recorded.get("files_sha256") or ({"model.safetensors": recorded["sha256"]} if recorded.get("sha256") else {})
                for name, expected in recorded_hashes.items():
                    if name in hashes and hashes[name] != expected:
                        raise RuntimeError(f"Model integrity check failed: {name}")
        atomic_json(previous, {"model": repo, "revision": revision, "files_sha256": hashes})
        print(f"Ready: {directory}")


if __name__ == "__main__":
    main()
