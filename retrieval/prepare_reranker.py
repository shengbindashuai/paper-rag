"""下载官方 Qwen3-Reranker-0.6B 到工作区，固定 revision 并校验权重哈希。

运行：python -m retrieval.prepare_reranker
只下载模型文件，不安装依赖、不执行仓库代码、不改 embedding 权重。
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import argparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from rag_core.config import default_root
from rag_core.io import file_sha256
from rag_core.storage import atomic_json
from rag_core.api_accounting import APILedger, cache_key


def cached_metadata(session, url, path, ledger, *, refresh=False):
    cached = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    key = cache_key(url)
    if cached and not refresh and time.time() - cached['at'] < 86400:
        ledger.record('model_metadata', 'not_a_token_api', key, {'unit': 'bytes', 'response_bytes': 0}, hit=True)
        return cached['metadata']
    headers = {'If-None-Match': cached['etag']} if cached and cached.get('etag') else {}
    response = session.get(url, headers=headers, timeout=45)
    if response.status_code == 304 and cached:
        metadata = cached['metadata']
    else:
        response.raise_for_status()
        raw = response.json()
        metadata = {'sha': raw['sha'], 'siblings': raw['siblings']}
    atomic_json(path, {'at': time.time(), 'etag': response.headers.get('ETag', headers.get('If-None-Match', '')),
                       'metadata': metadata})
    ledger.record('model_metadata', 'not_a_token_api', key,
                  {'unit': 'bytes', 'response_bytes': len(response.content), 'http_status': response.status_code},
                  hit=response.status_code == 304)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='重新校验远端模型元数据（支持 ETag）')
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    directory = args.root.resolve() / "RAG" / "models" / "Qwen3-Reranker-0.6B"
    directory.mkdir(parents=True, exist_ok=True)
    repo = "Qwen/Qwen3-Reranker-0.6B"
    session = requests.Session()
    ledger = APILedger(directory.parents[1] / 'library/api_accounting.sqlite3')
    metadata = cached_metadata(session, f"https://huggingface.co/api/models/{repo}?blobs=true",
                               directory / 'repository_metadata.json', ledger, refresh=args.refresh)
    revision = metadata["sha"]
    names = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json"}
    files = [f for f in metadata["siblings"] if f["rfilename"] in names]
    if not {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"}.issubset({f["rfilename"] for f in files}):
        raise RuntimeError("官方仓库文件结构与预期不符")
    verified = {}
    for info in files:
        name, size = info["rfilename"], info["size"]
        destination = directory / name
        expected = info.get("lfs", {}).get("sha256")
        if destination.exists() and destination.stat().st_size == size and (not expected or file_sha256(destination) == expected):
            print(f"Verified existing {name}", flush=True)
            ledger.record('model_file', 'not_a_token_api', cache_key([repo, revision, name]),
                          {'unit': 'bytes', 'response_bytes': 0, 'file_bytes': size}, hit=True)
        else:
            partial = directory / (name + ".partial")
            for attempt in range(30):
                received_bytes = 0
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset == size:
                        if expected and file_sha256(partial) != expected:
                            raise RuntimeError(f"下载 SHA256 不一致：{name}")
                        partial.replace(destination)
                        break
                    with session.get(f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                                     params={"download": "true", "offset": str(offset)},
                                     headers={"Range": f"bytes={offset}-"} if offset else {},
                                     timeout=(20, 40), stream=True) as stream:
                        stream.raise_for_status()
                        mode = "ab" if offset and stream.status_code == 206 else "wb"
                        with partial.open(mode) as output:
                            for data in stream.iter_content(2**20):
                                received_bytes += len(data)
                                output.write(data)
                    if partial.stat().st_size != size:
                        raise RuntimeError(f"下载大小不一致：{name}")
                    if expected and file_sha256(partial) != expected:
                        raise RuntimeError(f"下载 SHA256 不一致：{name}")
                    partial.replace(destination)
                    print(f"Downloaded and verified {name} ({size / 2**20:.1f} MiB)", flush=True)
                    ledger.record('model_file', 'not_a_token_api', cache_key([repo, revision, name]),
                                  {'unit': 'bytes', 'response_bytes': received_bytes, 'file_bytes': size})
                    break
                except (requests.RequestException, OSError) as exc:
                    ledger.record('model_file', 'not_a_token_api', cache_key([repo, revision, name]),
                                  {'unit': 'bytes', 'response_bytes': received_bytes}, error=type(exc).__name__)
                    print(f"Retry {attempt + 1}: {name} ({type(exc).__name__})", flush=True)
                    if attempt == 29:
                        raise RuntimeError(f"下载失败：{name}，可重试断点续传") from exc
                    time.sleep(1)
        # 非 LFS 小文件也比对官方 Git blob ID。
        if not expected and info.get("blobId"):
            data = destination.read_bytes()
            if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != info["blobId"]:
                raise RuntimeError(f"Git blob 校验失败：{name}")
        verified[name] = file_sha256(destination)
    atomic_json(directory / "provenance.json", {"model": repo, "revision": revision, "files_sha256": verified})
    print(f"Ready: {directory}", flush=True)


if __name__ == "__main__":
    main()
