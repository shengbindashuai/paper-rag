"""Check dependencies, CUDA, model files and the index without calling external APIs."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.config import default_root, config_file
from rag_core.io import file_sha256


def inspect(root, *, require_models=False, require_index=False, require_cuda=False):
    checks = []

    def add(name, ok, detail, required=True):
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    add("python", (3, 11) <= sys.version_info < (3, 12), sys.version.split()[0])
    packages = {"numpy": "numpy", "torch": "torch", "transformers": "transformers", "tokenizers": "tokenizers",
                "safetensors": "safetensors", "huggingface-hub": "huggingface_hub", "PyMuPDF": "pymupdf",
                "psutil": "psutil", "httpx": "httpx", "pydantic": "pydantic", "requests": "requests",
                "mcp": "mcp", "python-docx": "docx", "starlette": "starlette", "uvicorn": "uvicorn",
                "anyio": "anyio"}
    packages.update({"pywin32": "win32file"} if os.name == "nt" else {"watchdog": "watchdog"})
    for package, module in packages.items():
        installed = importlib.util.find_spec(module) is not None
        version = importlib.metadata.version(package) if installed else "missing"
        add(package, installed, version)
    try:
        import torch
        cuda = torch.cuda.is_available()
        add("cuda", cuda, torch.cuda.get_device_name(0) if cuda else "CUDA unavailable", require_cuda)
    except (ImportError, OSError) as exc:
        add("cuda", False, type(exc).__name__, require_cuda)
    with sqlite3.connect(":memory:") as db:
        try:
            db.execute("CREATE VIRTUAL TABLE check_fts USING fts5(text)")
            add("sqlite_fts5", True, sqlite3.sqlite_version)
        except sqlite3.OperationalError:
            add("sqlite_fts5", False, "SQLite must include FTS5")
    for name in ("retrieval.json", "quality_gate_policy.json", "knowledge_automation.json", "terminology_verification.json"):
        path = config_file(root, name)
        try:
            json.loads(path.read_text(encoding="utf-8"))
            add(name, True, str(path))
        except (OSError, ValueError) as exc:
            add(name, False, str(exc))
    for name in ("Qwen3-Embedding-0.6B", "Qwen3-Reranker-0.6B"):
        directory = root / "RAG/models" / name
        required = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
        add(name, all((directory / f).is_file() for f in required), str(directory), require_models)
        provenance = directory / "provenance.json"
        if require_models and provenance.is_file():
            record = json.loads(provenance.read_text(encoding="utf-8"))
            hashes = record.get("files_sha256") or ({"model.safetensors": record["sha256"]} if record.get("sha256") else {})
            valid = bool(hashes) and all((directory / f).is_file() and file_sha256(directory / f) == sha
                                        for f, sha in hashes.items())
            add(name + "_hashes", valid, record.get("revision", "unknown"))
    index = root / "RAG/library/current.json"
    add("index", index.is_file(), str(index), require_index)
    add("deepseek_api_key", bool(os.environ.get("DEEPSEEK_API_KEY")), "configured" if os.environ.get("DEEPSEEK_API_KEY") else "not configured", False)
    return {"root": str(root), "ok": all(c["ok"] for c in checks if c["required"]), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-index", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = inspect(args.root.resolve(), require_models=args.require_models,
                     require_index=args.require_index, require_cuda=args.require_cuda)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
