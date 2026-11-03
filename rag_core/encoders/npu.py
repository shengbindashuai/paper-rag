"""OpenVINO NPU 问题编码；明确指定 NPU，失败时不静默回退 CPU。

Tokenizer、FP32 归一化和向量检索仍在 CPU 上；Transformer 图在 NPU 上执行。
采用与 GPU 实验相同的指令、结束标记及 last-token pooling，固定左 padding。
"""
from pathlib import Path
import time

import numpy as np
import openvino as ov
from transformers import AutoTokenizer


class NpuQueryEmbedder:
    def __init__(self, model_dir: Path, tokenizer_dir: Path, instruction: str, cache_dir: Path):
        self.core = ov.Core()
        if "NPU" not in self.core.available_devices:
            raise RuntimeError(f"未检测到 NPU：{self.core.available_devices}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                                       trust_remote_code=False, padding_side="left")
        self.instruction = instruction
        model = self.core.read_model(model_dir / "openvino_model.xml")
        self.sequence_length = int(model.input("input_ids").shape[1])
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print("Compiling/loading static FP16 graph on NPU ...", flush=True)
        self.compiled = self.core.compile_model(model, "NPU", {
            "CACHE_DIR": str(cache_dir), "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16", "NPU_COMPILER_DYNAMIC_QUANTIZATION": False})
        self.compile_seconds = time.perf_counter() - started
        reported_devices = self.compiled.get_property("EXECUTION_DEVICES")
        devices = [reported_devices] if isinstance(reported_devices, str) else list(reported_devices)
        if not devices or any(not dev.startswith("NPU") for dev in devices):
            raise RuntimeError(f"执行设备不符合 NPU 要求：{devices}")
        self.device_info = {"name": self.core.get_property("NPU", "FULL_DEVICE_NAME"),
                            "driver_version": str(self.core.get_property("NPU", "NPU_DRIVER_VERSION")),
                            "execution_devices": devices, "openvino_version": ov.__version__,
                            "compile_or_cache_load_seconds": self.compile_seconds,
                            "sequence_length": self.sequence_length, "batch_size": 1,
                            "requested_inference_precision": "f16", "dynamic_quantization": False}
        self.request = self.compiled.create_infer_request()

    def encode_one(self, question: str) -> tuple[np.ndarray, dict]:
        started = time.perf_counter()
        text = f"Instruct: {self.instruction}\nQuery:{question}"
        count = len(self.tokenizer(text)["input_ids"])
        if count > self.sequence_length:
            raise ValueError(f"问题含指令共 {count} token，超过静态长度 {self.sequence_length}；拒绝截断")
        inputs = dict(self.tokenizer(text, padding="max_length", max_length=self.sequence_length,
                                     truncation=False, return_tensors="np"))
        # 只移除 pad 对位置计数的贡献；有效 token 的相对位置与原模型一致。
        positions = np.cumsum(inputs["attention_mask"], axis=1) - 1
        positions[inputs["attention_mask"] == 0] = 0
        inputs["position_ids"] = positions.astype(np.int64)
        inference_started = time.perf_counter()
        result = self.request.infer(inputs)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        raw = np.asarray(result[self.compiled.output(0)], dtype=np.float32).copy()
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        if not np.isfinite(raw).all() or np.any(norm == 0):
            raise RuntimeError("NPU 生成无效/零向量")
        vector = raw / norm
        if vector.shape != (1, 1024):
            raise RuntimeError(f"NPU 向量维度异常：{vector.shape}")
        return vector[0], {"end_to_end_ms": (time.perf_counter() - started) * 1000,
                           "inference_ms": inference_ms, "actual_tokens": count,
                           "padded_tokens": self.sequence_length}

    def close(self):
        """解除请求/编译图引用，供在线路由器释放按需档位。"""
        self.request = None
        self.compiled = None
        self.core = None
