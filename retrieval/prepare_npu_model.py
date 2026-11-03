"""把本地同一份权重转换为 OpenVINO 静态图；不下载、不训练、不做 INT4 量化。

python -m retrieval.prepare_npu_model

静态 batch=1，长度可通过 --sequence-length 选择。权重以 FP16 保存；保留原始权重。
显式构造与左 padding 对应的因果 mask，避免追踪动态 mask 分支时产生错误。
"""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import openvino as ov
import torch
from transformers import AutoModel, AutoTokenizer

from rag_core.config import Workspace, default_root
from rag_core.io import file_sha256
from rag_core.io import atomic_json as write_json


class StaticQueryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        # 有效 token 对有效 token 的注意力与原模型一致；最后一个 token 是 EOS。
        length = input_ids.shape[1]
        causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=input_ids.device))
        allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()
        bias = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        hidden = self.model(input_ids=input_ids, attention_mask=bias, position_ids=position_ids,
                            use_cache=False, return_dict=False)[0]
        return hidden[:, -1, :]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, choices=[128, 256, 512, 1024, 2048, 4096, 8192], default=128)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args()
    workspace = Workspace(args.root.resolve())
    destination = workspace.rag / "models" / f"Qwen3-Embedding-0.6B-openvino-fp16-s{args.sequence_length}"
    destination.mkdir(parents=True, exist_ok=True)
    print("Loading same local weights on CPU for conversion ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(workspace.tokenizer), local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(str(workspace.tokenizer), local_files_only=True, dtype=torch.float32,
                                     attn_implementation="eager").eval()
    model.requires_grad_(False)
    example = tokenizer("A scientific retrieval question.", padding="max_length", max_length=args.sequence_length,
                        truncation=False, return_tensors="pt")
    positions = example["attention_mask"].cumsum(-1) - 1
    positions.masked_fill_(example["attention_mask"] == 0, 0)
    inputs = (example["input_ids"], example["attention_mask"], positions)
    wrapper = StaticQueryEncoder(model).eval()
    # 转换前先验证手工 4D mask 与 transformers 默认 mask 的有效最后 token 一致。
    with torch.inference_mode():
        expected = model(**example, position_ids=positions, use_cache=False).last_hidden_state[:, -1].numpy()
        explicit = wrapper(*inputs).numpy()
    error = float(np.max(np.abs(expected - explicit)))
    if not np.allclose(expected, explicit, atol=1e-4, rtol=1e-4):
        raise RuntimeError(f"显式因果 mask 与默认结果不一致：{error}")
    print(f"Mask equivalence passed (max_abs={error:.8f}); converting ...", flush=True)
    started = time.perf_counter()
    converted = ov.convert_model(wrapper, example_input=inputs, input=[
        ("input_ids", [1, args.sequence_length], ov.Type.i64),
        ("attention_mask", [1, args.sequence_length], ov.Type.i64),
        ("position_ids", [1, args.sequence_length], ov.Type.i64)])
    converted.output(0).get_tensor().set_names({"last_token_hidden_state"})
    ov.save_model(converted, destination / "openvino_model.xml", compress_to_fp16=True)
    info = {"model": "Qwen/Qwen3-Embedding-0.6B", "precision": "FP16 weight storage; no INT4/INT8 quantization",
            "source_model_sha256": file_sha256(workspace.tokenizer / "model.safetensors"),
            "sequence_length": args.sequence_length, "batch_size": 1,
            "conversion_seconds": time.perf_counter() - started, "mask_equivalence_max_abs": error,
            "openvino_version": ov.__version__, "xml_sha256": file_sha256(destination / "openvino_model.xml"),
            "bin_sha256": file_sha256(destination / "openvino_model.bin")}
    write_json(destination / "conversion.json", info)
    print(f"Exported: {destination}", flush=True)


if __name__ == "__main__":
    main()
