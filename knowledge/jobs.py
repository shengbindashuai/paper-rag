"""持久化串行入库队列；模型访问与检索共用锁，子进程退出释放 GPU。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
import uuid

from rag_core.io import atomic_json, utc_now
from rag_core.storage import writer_lock


class IngestJobs:
    def __init__(self, service, allowed_roots=None, runner=None):
        self.service = service
        self.root = service.config.root.resolve()
        self.index_dir = service.config.path("index_dir")
        self.directory = self.index_dir / "ingest_jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = [Path(p).resolve() for p in (allowed_roots or [self.root])]
        self.lock, self.closed = RLock(), False
        self.listeners = set()
        self.owner = writer_lock(self.directory / "owner")
        self.owner.__enter__()
        # 不自动重放崩溃时的写入；可用相同请求重试，哈希注册不会覆盖原件。
        for path in self.directory.glob("*.job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in {"queued", "running"}:
                job.update(status="interrupted", finished_at=utc_now(), error="服务重启中断，请重新提交")
                atomic_json(path, job)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-ingest")
        self.runner = runner or self._run_process

    def validate_path(self, value, suffixes):
        path = Path(value).resolve(strict=True)
        if not path.is_file() or not any(path.is_relative_to(root) for root in self.allowed_roots):
            raise ValueError("文件不在允许读取的目录；启动时可用 --allow-read-root 添加目录")
        if path.suffix.lower() not in suffixes or path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("文件格式不支持或超过 100 MiB")
        return path

    def submit(self, kind, request):
        from knowledge.sources import extract_document
        payload = request.model_dump(exclude_none=True)
        payload["source_type"] = kind
        if "path" in payload:
            path = self.validate_path(payload["path"], {".pdf"} if kind == "paper" else {".md", ".txt", ".docx"})
            payload["path"] = str(path)
            payload["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if kind != "paper":
                payload["content"] = extract_document(path)
                payload["original_path"] = payload.pop("path")
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            if self.closed:
                raise RuntimeError("入库队列已关闭")
            jobs = [json.loads(p.read_text(encoding="utf-8")) for p in self.directory.glob("*.job.json")]
            for job in jobs:
                if job["identity"] == identity and job["status"] in {"queued", "running", "completed"}:
                    return self._public(job)
            if sum(j["status"] in {"queued", "running"} for j in jobs) >= 32:
                raise RuntimeError("入库队列已满，请稍后重试")
            identifier = uuid.uuid4().hex
            payload["result_path"] = str(self.directory / f"{identifier}.result.json")
            atomic_json(self.directory / f"{identifier}.input.json", payload)
            job = {"job_id": identifier, "identity": identity, "source_type": kind,
                   "status": "queued", "created_at": utc_now()}
            atomic_json(self.directory / f"{identifier}.job.json", job)
            self.pool.submit(self._execute, identifier)
            return self._public(job)

    def _public(self, job):
        return {k: v for k, v in job.items() if k != "identity"}

    def subscribe(self, callback):
        with self.lock:
            self.listeners.add(callback)
        def unsubscribe():
            with self.lock:
                self.listeners.discard(callback)
        return unsubscribe

    def status(self, identifier):
        # 即使 Python 调用方绕过 schema，也不允许 job_id 用于目录穿越。
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("非法 job_id")
        with self.lock:
            path = self.directory / f"{identifier}.job.json"
            if not path.is_file():
                raise KeyError("入库任务不存在")
            return self._public(json.loads(path.read_text(encoding="utf-8")))

    def _execute(self, identifier):
        path = self.directory / f"{identifier}.job.json"
        with self.lock:
            job = json.loads(path.read_text(encoding="utf-8"))
            job.update(status="running", started_at=utc_now())
            atomic_json(path, job)
        try:
            with self.service.lock:
                self.service.reranker.close()
                self.runner(identifier)
                result = json.loads((self.directory / f"{identifier}.result.json").read_text(encoding="utf-8"))
                if not result.get("indexed"):
                    raise RuntimeError("入库进程未确认索引发布")
            job.update(status="completed", result=result)
        except Exception as exc:
            # 不向 MCP 返回底层异常正文（可能含凭证）；详细错误只写本地子进程日志。
            job.update(status="failed", error=f"{type(exc).__name__}: 入库失败，请查看本地任务日志")
            import traceback
            with (self.directory / f"{identifier}.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            result_path = self.directory / f"{identifier}.result.json"
            if result_path.exists():
                try:
                    job["result"] = json.loads(result_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
        job["finished_at"] = utc_now()
        job["log_file"] = str(self.directory / f"{identifier}.log")
        with self.lock:
            atomic_json(path, job)
            listeners = list(self.listeners)
        for callback in listeners:
            try:
                callback('jobs')
            except Exception:
                import logging
                logging.exception('入库完成通知失败')

    def _run_process(self, identifier):
        payload_path = self.directory / f"{identifier}.input.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        original = payload.get("path") or payload.get("original_path")
        if original and hashlib.sha256(Path(original).read_bytes()).hexdigest() != payload["input_sha256"]:
            raise ValueError("提交后源文件发生变化，请重新提交")
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        with (self.directory / f"{identifier}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "knowledge.worker", "--root", str(self.root),
                            "--index-dir", str(self.index_dir), "--payload", str(payload_path)],
                           cwd=Path(__file__).resolve().parents[1], env=env, stdout=log, stderr=log, check=True)

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True)
        self.owner.__exit__(None, None, None)
