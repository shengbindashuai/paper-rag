"""本地 Qwen 推理后端：固定输入规范，只生成向量，不更新参数。

将模型推理与检索评测分开，方便之后替换其他本地模型或 API 后端。
GPU 指标是 PyTorch 进程分配的显存，不是整机所有程序的显存总量。
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class QwenEmbedder:
    def __init__(self, model_dir: Path, instruction: str, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前解释器 CUDA 不可用")
        self.device = device
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False, padding_side="left")
        start = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.synchronize()
        self.load_seconds = time.perf_counter() - start
        provenance = model_dir / "provenance.json"
        self.provenance = json.loads(provenance.read_text()) if provenance.exists() else {}

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def encode(self, texts: list[str], *, query: bool = False, batch_size: int = 2,
               max_tokens: int = 32768, progress: bool = False) -> tuple[np.ndarray, dict]:
        if not texts or batch_size < 1:
            raise ValueError("需要非空文本列表及正 batch size")
        prepared = [f"Instruct: {self.instruction}\nQuery:{text}" for text in texts] if query else texts
        self.synchronize()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs, counts, padded_tokens = [], [], 0
        for offset in range(0, len(prepared), batch_size):
            inputs = self.tokenizer(prepared[offset:offset + batch_size], padding=True,
                                    truncation=False, return_tensors="pt")
            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            if max(lengths) > max_tokens:
                raise ValueError(f"输入长度 {max(lengths)} 超过 {max_tokens}；拒绝静默截断")
            counts.extend(lengths)
            padded_tokens += inputs["input_ids"].numel()
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                hidden = self.model(**inputs, use_cache=False).last_hidden_state
                # 左 padding 保证最后一列是有效 token，采用官方 last-token pooling。
                if not bool(inputs["attention_mask"][:, -1].all()):
                    raise RuntimeError("最后一列出现 padding，不能使用 last-token pooling")
                raw = hidden[:, -1].float()
                if not torch.isfinite(raw).all() or (raw.norm(dim=1) == 0).any():
                    raise RuntimeError("模型生成了无效/零向量")
                outputs.append(F.normalize(raw, p=2, dim=1).cpu().numpy())
            del hidden, raw, inputs
            if progress and ((offset // batch_size) % 25 == 0 or offset + batch_size >= len(prepared)):
                print(f"  encoded {min(offset + batch_size, len(prepared))}/{len(prepared)}", flush=True)
        self.synchronize()
        vectors = np.concatenate(outputs)
        norm_error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)))
        if not np.isfinite(vectors).all() or norm_error > 1e-4:
            raise RuntimeError("向量数值或归一化检查失败")
        return vectors, {
            "encoding_seconds": time.perf_counter() - started, "count": len(texts),
            "dimension": int(vectors.shape[1]), "batch_size": batch_size,
            "actual_input_tokens": int(sum(counts)), "padded_input_tokens": padded_tokens,
            "input_token_min": min(counts), "input_token_median": float(np.median(counts)),
            "input_token_max": max(counts), "max_norm_error": norm_error,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if self.device == "cuda" else None,
        }
