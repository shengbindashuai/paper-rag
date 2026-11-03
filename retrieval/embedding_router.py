"""128 常驻 + 最多一个较大 NPU 档位；超 1024 临时使用 GPU。

计数包括指令与自动 EOS。锁覆盖模型切换和推理；并发调用不会卸载正在
使用的 InferRequest。NPU 不可用时回退 GPU，失败档位冷却后再尝试。
"""
from contextlib import redirect_stdout
import gc
import json
import sys
from threading import RLock
import time

from rag_core.io import file_sha256

NPU_UNAVAILABLE = (RuntimeError, OSError, ImportError, EOFError)


def choose_device(tokens: int, gpu_max_tokens=32768):
    if tokens < 1 or tokens > gpu_max_tokens:
        raise ValueError(f"完整查询需在 1～{gpu_max_tokens} token 内；拒绝截断")
    for length in (128, 256, 512, 1024):
        if tokens <= length:
            return "NPU", length
    return "GPU", None


class QueryEmbeddingRouter:
    def __init__(self, config, *, tokenizer=None, npu_factory=None, gpu_factory=None, model_sha256=None):
        self.config = config
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(config.path("embedding_dir")), local_files_only=True)
        self.tokenizer = tokenizer
        self.model_sha256 = model_sha256 or file_sha256(config.path("embedding_dir") / "model.safetensors")
        self.npu_factory = npu_factory or self._load_npu
        self.gpu_factory = gpu_factory or self._load_gpu
        self.models, self.optional_length = {}, None
        self.lock = RLock()
        self.started = False
        self.npu_failures = {}
        self.npu_retry_seconds = 60
        self.before_gpu = None  # Service 在此注册释放 GPU reranker 的回调。

    def _load_npu(self, length):
        from .npu_worker import IsolatedNpuEncoder
        directory = self.config.path("models_dir") / f"Qwen3-Embedding-0.6B-openvino-fp16-s{length}"
        conversion = json.loads((directory / "conversion.json").read_text(encoding="utf-8"))
        if conversion["source_model_sha256"] != self.model_sha256 or conversion["sequence_length"] != length:
            raise ValueError("NPU 导出版本与当前 embedding 源权重不一致")
        with redirect_stdout(sys.stderr):
            return IsolatedNpuEncoder(directory, self.config.path("embedding_dir"), self.config.instruction,
                                    self.config.path("cache_dir") / f"npu_fp16_s{length}")

    def _load_gpu(self):
        from rag_core.encoders.gpu import QwenEmbedder
        return QwenEmbedder(self.config.path("embedding_dir"), self.config.instruction)

    def _get_npu(self, length):
        if length not in self.models:
            if length != 128 and self.optional_length is not None:
                old = self.models.pop(self.optional_length)
                self.optional_length = None
                old.close()
                del old
                gc.collect()
            self.models[length] = self.npu_factory(length)
            if length != 128:
                self.optional_length = length
        return self.models[length]

    def start(self):
        with self.lock:
            if 128 not in self.models and self._failure(128) is None:
                try:
                    self._get_npu(128).encode_one("Warm-up scientific retrieval question.")
                except NPU_UNAVAILABLE as exc:
                    self._record_failure(128, exc)
            # 服务允许在没有 NPU 的机器上启动；GPU 权重仍按请求延迟加载。
            self.started = True

    def _failure(self, length):
        failure = self.npu_failures.get(length)
        if failure and time.monotonic() >= failure["retry_at"]:
            del self.npu_failures[length]
            return None
        return failure

    def _record_failure(self, length, exc):
        model = self.models.pop(length, None)
        if self.optional_length == length:
            self.optional_length = None
        if model is not None:
            try:
                model.close()
            except (RuntimeError, OSError):
                pass  # 故障进程仍需尽力清理，不能因此阻止 GPU 回退。
        self.npu_failures[length] = {"error_type": type(exc).__name__, "message": str(exc)[:500],
                                    "retry_at": time.monotonic() + self.npu_retry_seconds}
        gc.collect()

    def _encode_gpu(self, query, count, reason):
        if self.before_gpu is not None:
            self.before_gpu()
        gpu = self.gpu_factory()
        try:
            vectors, stats = gpu.encode([query], query=True, batch_size=1, max_tokens=self.config.gpu_max_tokens)
            return vectors[0], {**stats, "device": "GPU", "npu_length": None,
                               "actual_tokens": count, "reason": reason}
        finally:
            close = getattr(gpu, "close", None)
            if close:
                close()
            del gpu
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def token_length(self, query):
        return len(self.tokenizer(f"Instruct: {self.config.instruction}\nQuery:{query}")["input_ids"])

    def encode(self, query):
        started = time.perf_counter()
        count = self.token_length(query)
        device, length = choose_device(count, self.config.gpu_max_tokens)
        with self.lock:
            self.start()
            failure = self._failure(128) if device == "NPU" else None
            failure = failure or (self._failure(length) if device == "NPU" else None)
            if device == "NPU" and failure is None:
                cached = length in self.models
                try:
                    vector, stats = self._get_npu(length).encode_one(query)
                    route = {"device": device, "npu_length": length, "model_was_loaded": cached, **stats}
                except NPU_UNAVAILABLE as exc:
                    self._record_failure(length, exc)
                    failure = self._failure(length)
            if device == "GPU" or failure is not None:
                reason = "query_exceeds_1024_tokens" if device == "GPU" else "npu_unavailable"
                vector, route = self._encode_gpu(query, count, reason)
                if failure is not None:
                    route["requested_npu_length"] = length
                    route["fallback"] = {k: failure[k] for k in ["error_type", "message"]}
                    route["fallback"]["retry_after_seconds"] = max(0, failure["retry_at"] - time.monotonic())
            route["routing_total_ms"] = (time.perf_counter() - started) * 1000
            route["resident_npu_lengths"] = sorted(self.models)
            return vector, route

    def status(self):
        return {"started": self.started, "resident_npu_lengths": sorted(self.models),
                "policy": "128 pinned + one on-demand NPU tier; GPU above 1024 or when NPU unavailable",
                "npu_failures": {str(k): {"error_type": v["error_type"], "message": v["message"],
                    "retry_after_seconds": max(0, v["retry_at"] - time.monotonic())} for k, v in self.npu_failures.items()},
                "gpu_max_tokens": self.config.gpu_max_tokens, "model_sha256": self.model_sha256}

    def close(self):
        with self.lock:
            for model in self.models.values():
                model.close()
            self.models.clear()
            self.optional_length, self.started = None, False
            self.npu_failures.clear()
            gc.collect()
