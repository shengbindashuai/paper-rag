"""全库流水线及 Python 接口：有界 CPU 预取 + 单 GPU 大批量编码 + 文档事务。

Windows 上使用 spawn 多进程，调用方必须在 if __name__ == '__main__' 下启动。
CPU 工作者只加载 tokenizer；GPU 权重只在主进程加载一次。所有 GPU 失败显式报错。
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import nullcontext
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil

from rag_core.config import ProcessingConfig, Workspace, config_file, resource_path
from ingestion.inventory import register_pdf, scan_library
from ingestion.worker import initialize_worker, load_chunks, prepare_document
from rag_core.storage import LibraryStore, atomic_json, utc_now, writer_lock
from ingestion.preprocessing.pdf_extract import file_sha256


class ResourceMonitor:
    def __init__(self, directory):
        self.directory, self.last, self.samples = directory, {}, []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pynvml, handle = None, None
        psutil.cpu_percent()
        with (self.directory / "resources.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
            while not self.stop_event.wait(2):
                memory = psutil.virtual_memory()
                process = psutil.Process()
                processes = [process] + process.children(recursive=True)
                rss = 0
                for child in processes:
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                sample = {"at": utc_now(), "cpu_percent": psutil.cpu_percent(), "ram_used_gib": memory.used / 2**30,
                          "ram_available_gib": memory.available / 2**30, "pipeline_rss_gib": rss / 2**30,
                          "process_count": len(processes)}
                if handle is not None:
                    try:
                        sample.update(gpu_percent=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                                      gpu_memory_mib=pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20)
                    except Exception:
                        pass
                self.last = sample
                self.samples.append(sample)
                stream.write(json.dumps(sample) + "\n")
        if pynvml is not None:
            pynvml.nvmlShutdown()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=5)


def processing_signature(workspace, config):
    base = Path(__file__).resolve().parents[1]
    files = ["ingestion/worker.py", "ingestion/preprocessing/quality.py", "ingestion/preprocessing/chunking.py",
             "ingestion/preprocessing/cleaning.py", "ingestion/preprocessing/pdf_extract.py",
             "ingestion/preprocessing/validation.py", "ingestion/preprocessing/notes.py"]
    details = {"schema": "paper-library-v1", "model": "Qwen/Qwen3-Embedding-0.6B", "dimension": 1024,
               "model_sha256": file_sha256(workspace.tokenizer / "model.safetensors"), "config": config.to_dict(),
               "code_sha256": {name: file_sha256(base / name) for name in files},
               "encoder_recipe_version": "qwen-last-token-fp16-v1",
               "document_input": "Title + Section + body; automatic EOS; left padding; last-token; FP32 L2; FP16 CUDA"}
    details["digest"] = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    return details


def calibrate_batch(embedder, maximum=128):
    """用接近 512 token 的输入测吞吐和峰值显存，选择实际较快的批量。"""
    import torch
    text = "context " * 500
    embedder.encode([text] * 4, batch_size=4, max_tokens=512)
    choices = sorted({n for n in [16, 32, 64, 128, maximum] if 1 <= n <= maximum})
    measurements = []
    free, _ = torch.cuda.mem_get_info()
    budget = torch.cuda.memory_allocated() + max(0, free - 768 * 2**20)
    for batch in choices:
        if len(measurements) >= 2:
            # 先根据已测批量估算下一档峰值，防止 WDDM 把显存挤到共享 RAM 后严重降速。
            previous, latest = measurements[-2:]
            slope = max(0, (latest["peak_allocated_mib"] - previous["peak_allocated_mib"]) /
                        (latest["batch_size"] - previous["batch_size"]))
            predicted = latest["peak_allocated_mib"] + slope * (batch - latest["batch_size"])
            if predicted * 2**20 > budget:
                break
        try:
            _, stats = embedder.encode([text] * (batch * 2), batch_size=batch, max_tokens=512)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        if stats["peak_allocated_mib"] * 2**20 > budget:
            torch.cuda.empty_cache()
            break
        measurements.append({"batch_size": batch, "texts_per_second": batch * 2 / stats["encoding_seconds"], **stats})
        print(f"GPU calibration: batch={batch}, {measurements[-1]['texts_per_second']:.1f} chunks/s, peak={stats['peak_allocated_mib']:.0f} MiB", flush=True)
    if not measurements:
        return 4, []
    return max(measurements, key=lambda row: row["texts_per_second"])["batch_size"], measurements


def build_library(root: Path, *, index_dir: Path | None = None, workers: int | None = None,
                  max_batch_size=128, doc_ids=None, limit=None, retry_failed=False, _locked=False):
    """可重入的增量 Python 接口。没有变化的 indexed 文档不重新解析或编码。"""
    root = root.resolve()
    directory = (index_dir or root / "RAG" / "library").resolve()
    with (nullcontext() if _locked else writer_lock(directory)):
        result = _build(root, directory, workers, max_batch_size, doc_ids, limit, retry_failed)
    if not _locked:
        result['terminology'] = post_build_verification(root, directory)
    return result


def post_build_verification(root, directory):
    """独立建库使用与 MCP 新文件相同的完整三表核验流程。"""
    path = config_file(root, "knowledge_automation.json")
    if not path.is_file():
        return {'status': 'disabled'}
    options = json.loads(path.read_text(encoding='utf-8'))
    if not options.get('enabled'):
        return {'status': 'disabled'}
    target = root / options['terminology_directory'] if directory == root / 'RAG/library' else directory / 'terminology/auto'
    try:
        from terminology.maintenance import synchronize
        from terminology.semantic_review import run
        initial = resource_path(root, options['initial_decisions']) if options.get('initial_decisions') else None
        result = synchronize(root, directory, target, initial)
        result['semantic_verification'] = run(root, target)
        return result
    except Exception as exc:
        # 文献索引已发布，不能把外部接口失败误报成文献不存在。
        result = {'status': 'failed', 'error_type': type(exc).__name__,
                  'message': '索引已发布，三表验证尚未完成；查看本地验证状态并断点重试'}
        atomic_json(directory / 'terminology_followup.json', result)
        return result


def _build(root, directory, workers, max_batch_size, doc_ids, limit, retry_failed):
    from rag_core.encoders.gpu import QwenEmbedder
    import torch
    workspace, config = Workspace(root), ProcessingConfig()
    config.validate()
    signature = processing_signature(workspace, config)
    store = LibraryStore(directory, signature)
    started = time.perf_counter()
    print("Scanning PDFs and checking SHA256 ...", flush=True)
    inventory, unavailable = scan_library(root)
    atomic_json(directory / "inventory.json", {"at": utc_now(), "source_count": len(inventory),
                                               "pdf_count": sum(i.get("source_type") != "reading_excerpt" for i in inventory),
                                               "reading_excerpt_count": sum(i.get("source_type") == "reading_excerpt" for i in inventory),
                                               "metadata_without_pdf": unavailable, "documents": inventory})
    selected = [item for item in inventory if doc_ids is None or item["doc_id"] in set(doc_ids)]
    if doc_ids and set(doc_ids) - {item["doc_id"] for item in selected}:
        raise ValueError("指定的文档编号没有对应 PDF")
    if limit is not None:
        selected = selected[:limit]
    canonical = {row["source_sha256"]: row["doc_id"] for row in store.db.execute("SELECT * FROM documents WHERE status='indexed'")
                 if (workspace.rag / row["source_file"]).is_file()}
    jobs, reused, duplicate_count = [], 0, 0
    for item in selected:
        old = store.document(item["doc_id"])
        if item["metadata_hash_mismatch"]:
            store.record(item, "failed", "PDF SHA256 与原 metadata 不符，请人工核对版本")
            continue
        if item["sha256"] in canonical and canonical[item["sha256"]] != item["doc_id"]:
            store.record(item, "duplicate", alias_of=canonical[item["sha256"]])
            duplicate_count += 1
            continue
        canonical[item["sha256"]] = item["doc_id"]
        if old and old["source_sha256"] == item["sha256"] and old["title"] == item["title"]:
            if old["status"] == "indexed" or (old["status"] in {"failed", "quarantined"} and not retry_failed):
                reused += 1
                continue
        jobs.append(item)
    # 全库扫描才同步“已移除”的文件；单文档增量入库不会影响其他文档。
    if doc_ids is None and limit is None:
        actual = {item["doc_id"] for item in inventory}
        for old in list(store.db.execute("SELECT * FROM documents")):
            if old["doc_id"] not in actual:
                store.record({"doc_id": old["doc_id"], "source_file": old["source_file"],
                              "sha256": old["source_sha256"], "title": old["title"]}, "missing", "源 PDF 当前不存在")
    available = psutil.virtual_memory().available / 2**30
    workers = workers or max(1, min(8, (os.cpu_count() or 4) - 3, int((available - 5) / 1.25)))
    if workers < 1 or max_batch_size < 1:
        raise ValueError("workers/batch size 必须为正整数")
    print(f"Inventory={len(inventory)}, pending={len(jobs)}, reused={reused}, duplicate={duplicate_count}; CPU workers={workers}", flush=True)
    run = {"started_at": utc_now(), "inventory_sources": len(inventory), "selected_sources": len(selected),
           "inventory_pdfs": sum(i.get("source_type") != "reading_excerpt" for i in inventory),
           "inventory_reading_excerpts": sum(i.get("source_type") == "reading_excerpt" for i in inventory),
           "metadata_without_pdf": len(unavailable), "pending_at_start": len(jobs), "reused": reused,
           "cpu_workers": workers, "signature": signature, "encoded_documents": 0, "encoded_chunks": 0,
           "cpu_completed": 0, "failed_this_run": 0, "quarantined_this_run": 0, "encoding_seconds": 0.,
           "actual_input_tokens": 0, "padded_input_tokens": 0, "gpu_batches": []}
    if jobs:
        # spawn 子进程继承这些设置，避免每个进程再开启 20 个内部计算线程。
        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"]:
            os.environ[key] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(2)
        embedder = QwenEmbedder(workspace.tokenizer, "Given a scientific question, retrieve passages that answer the question.")
        batch_size, calibration = calibrate_batch(embedder, max_batch_size)
        torch.cuda.empty_cache()  # 释放校准时更大批量遗留的空闲显存块。
        run["batch_calibration"], run["selected_gpu_batch_size"] = calibration, batch_size
        buffered, buffered_count, next_job = [], 0, 0
        last_print = time.monotonic()
        with ResourceMonitor(directory) as monitor, ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker, initargs=(str(workspace.tokenizer),)) as pool, ThreadPoolExecutor(max_workers=1) as gpu_pool:
            futures = {}
            gpu_future = None

            def submit_jobs():
                nonlocal next_job
                # 保留至少 3 GiB 空闲内存；任务/结果数量有界，磁盘文件承担长队列。
                while (next_job < len(jobs) and len(futures) < workers * 3 and buffered_count < 4096
                       and (psutil.virtual_memory().available > 3 * 2**30 or not futures)):
                    item = jobs[next_job]
                    future = pool.submit(prepare_document, item, str(directory), config.to_dict(), signature["digest"])
                    futures[future] = item
                    next_job += 1

            def encode_buffer(documents, initial_batch_size):
                # 此线程只计算 GPU 向量；SQLite 始终由主线程提交。
                chunks = [c for _, block in documents for c in block]
                order = np.argsort([c["token_count"] for c in chunks], kind="stable")
                dense_batch = initial_batch_size
                pieces, reports, start = [], [], 0
                while start < len(order):
                    end = start + 1
                    # 短文本可用更大 batch，在相同 padded-token 预算下提高吞吐。
                    while (end < len(order) and end - start < max_batch_size and
                           (end - start + 1) * chunks[order[end]]["token_count"] <= dense_batch * config.max_tokens):
                        end += 1
                    try:
                        encoded, stats = embedder.encode([chunks[i]["embedding_text"] for i in order[start:end]],
                                                          batch_size=end - start, max_tokens=config.max_tokens)
                    except torch.cuda.OutOfMemoryError:
                        if dense_batch == 1:
                            raise
                        dense_batch = max(1, dense_batch // 2)
                        torch.cuda.empty_cache()
                        continue
                    pieces.append(encoded)
                    reports.append(stats)
                    start = end
                encoded = np.concatenate(pieces)
                vectors = np.empty_like(encoded)
                vectors[order] = encoded
                stats = {key: sum(r[key] for r in reports) for key in ["encoding_seconds", "actual_input_tokens", "padded_input_tokens"]}
                stats.update(peak_allocated_mib=max(r["peak_allocated_mib"] for r in reports),
                             peak_reserved_mib=max(r["peak_reserved_mib"] for r in reports),
                             microbatch_count=len(reports), min_batch_size=min(r["batch_size"] for r in reports),
                             max_batch_size=max(r["batch_size"] for r in reports), dense_batch_size=dense_batch)
                return documents, vectors, stats, dense_batch

            def commit_vectors(completed):
                nonlocal batch_size
                documents, vectors, stats, batch_size = completed
                offset = 0
                for item, block in documents:
                    store.commit_document(item, block, vectors[offset:offset + len(block)])
                    offset += len(block)
                    run["encoded_documents"] += 1
                run["encoded_chunks"] += len(vectors)
                for key in ["encoding_seconds", "actual_input_tokens", "padded_input_tokens"]:
                    run[key] += stats[key]
                run["gpu_batches"].append({"documents": len(documents), "chunks": len(vectors), **stats})

            submit_jobs()
            while futures or next_job < len(jobs) or buffered or gpu_future is not None:
                if not futures:
                    submit_jobs()
                if futures:
                    done, _ = wait(futures, timeout=.1, return_when=FIRST_COMPLETED)
                else:
                    done = []
                    time.sleep(.05)
                for future in done:
                    item = futures.pop(future)
                    result = future.result()
                    run["cpu_completed"] += 1
                    if result["status"] == "prepared":
                        chunks = [c for c in load_chunks(result["chunks_path"]) if c["retrieval_eligible"]]
                        if len(chunks) != result["chunk_count"]:
                            raise RuntimeError("CPU 缓存片段数量不一致")
                        buffered.append((item, chunks))
                        buffered_count += len(chunks)
                    else:
                        status = result["status"]
                        store.record(item, status, result.get("error", "没有通过质量筛选的正文；保留审计文件待复核"))
                        run["failed_this_run" if status == "failed" else "quarantined_this_run"] += 1
                        print(f"{status}: {item['doc_id']}: {result.get('error', 'no eligible text')}", flush=True)
                submit_jobs()
                if gpu_future is not None and gpu_future.done():
                    commit_vectors(gpu_future.result())
                    gpu_future = None
                if gpu_future is None and buffered and (buffered_count >= 1024 or (not futures and next_job == len(jobs))):
                    payload = buffered
                    buffered, buffered_count = [], 0
                    gpu_future = gpu_pool.submit(encode_buffer, payload, batch_size)
                if time.monotonic() - last_print > 15:
                    progress = {**{k: v for k, v in run.items() if k not in {"gpu_batches", "batch_calibration", "signature"}},
                                "elapsed_seconds": time.perf_counter() - started, "buffered_chunks": buffered_count,
                                "submitted": next_job, "resources": monitor.last, "status": store.status()}
                    atomic_json(directory / "progress.json", progress)
                    print(f"Progress CPU={run['cpu_completed']}/{len(jobs)} indexed={run['encoded_documents']} chunks={run['encoded_chunks']} buffered={buffered_count} resources={json.dumps(monitor.last)}", flush=True)
                    last_print = time.monotonic()
        run["resource_summary"] = {key: {"mean": float(np.mean([s[key] for s in monitor.samples if key in s])),
                                          "max": max(s[key] for s in monitor.samples if key in s)}
                                   for key in ["cpu_percent", "ram_used_gib", "pipeline_rss_gib", "gpu_percent", "gpu_memory_mib"]
                                   if any(key in s for s in monitor.samples)}
        del embedder
        torch.cuda.empty_cache()
    print("Publishing memory-mapped search snapshot ...", flush=True)
    if store.status()["chunks"]:
        store.refresh_snapshot()
    run.update(completed_at=utc_now(), elapsed_seconds=time.perf_counter() - started, final_status=store.status())
    atomic_json(directory / "last_build.json", run)
    atomic_json(directory / "progress.json", {"state": "complete", **run["final_status"], "completed_at": run["completed_at"]})
    store.close()
    print(json.dumps({key: run[key] for key in ["elapsed_seconds", "encoded_documents", "encoded_chunks", "failed_this_run", "quarantined_this_run", "final_status"]}, ensure_ascii=False, indent=2), flush=True)
    return run


def ingest_documents(root: Path, pdf_paths, *, index_dir: Path | None = None, title=None, **build_options):
    """外部程序可调用的增量接口；相同内容不重复编码，复制与建库共用一个写锁。"""
    root = root.resolve()
    directory = (index_dir or root / "RAG" / "library").resolve()
    with writer_lock(directory):
        ids = [register_pdf(root, Path(path), title=title) for path in pdf_paths]
        result = build_library(root, index_dir=directory, doc_ids=ids, _locked=True, **build_options)
    result['terminology'] = post_build_verification(root, directory)
    return result
