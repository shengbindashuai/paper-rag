"""Qwen3-Reranker-0.6B：只对召回候选打分，分数不作为校准后的置信度。

按官方 yes/no 模板计算 P(yes | yes/no)，不生成自由文本。只计算末尾 logits，
避免长输入为每个 token 生成整个词表的 logits。超限时显式报错，不截断问题。
"""
import gc
import time


PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class QwenReranker:
    def __init__(self, model_dir, instruction, batch_size=2, max_tokens=32768):
        self.directory, self.instruction = model_dir, instruction
        self.batch_size, self.max_tokens = batch_size, max_tokens
        self.model = self.tokenizer = None

    def _load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not torch.cuda.is_available():
                raise RuntimeError("重排需要 CUDA GPU；不会静默改用其他设备")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory), local_files_only=True, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(str(self.directory), local_files_only=True,
                use_safetensors=True, trust_remote_code=False, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
            self.model.requires_grad_(False)
            self.no_id = self.tokenizer.convert_tokens_to_ids("no")
            self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")

    def _prepare_batches(self, query, hits):
        """Fast tokenizer 一次完成批量编码和左 padding，不拆成 encode + pad。

        预先检查所有 CPU 批次，超限时在任何 GPU 推理前报错。
        """
        batches, longest = [], 0
        for offset in range(0, len(hits), self.batch_size):
            prompts = [
                f"{PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: "
                f"{hit.get('embedding_text', hit['text'])}{SUFFIX}"
                for hit in hits[offset:offset + self.batch_size]]
            batch = self.tokenizer(prompts, add_special_tokens=False, padding=True,
                                   truncation=False, return_tensors="pt",
                                   return_attention_mask=True, return_token_type_ids=False)
            length = int(batch["attention_mask"].sum(dim=1).max().item())
            if length > self.max_tokens:
                raise ValueError(f"问题＋文档片段＋重排模板达 {length} token，超过 {self.max_tokens}；拒绝截断")
            longest = max(longest, length)
            batches.append(batch)
        return batches, longest

    def rerank(self, query, hits):
        if not hits:
            return [], {"model": "Qwen/Qwen3-Reranker-0.6B", "count": 0}
        import torch
        started = time.perf_counter()
        self._load()
        batches, longest = self._prepare_batches(query, hits)
        scores = []
        for cpu_batch in batches:
            # 不对列表中的 BatchEncoding 原地 .to()，避免已处理批次一直占用显存。
            batch = {name: tensor.to("cuda") for name, tensor in cpu_batch.items()}
            with torch.inference_mode():
                logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                pair = logits[:, [self.no_id, self.yes_id]].float()
                values = torch.softmax(pair, dim=1)[:, 1]
                if not torch.isfinite(values).all():
                    raise RuntimeError("重排生成无效分数")
                scores.extend(values.cpu().tolist())
            del batch, logits, pair, values
        ordered = [{**hit, "rerank_score": float(score), "dense_rank": hit.get("dense_rank", rank + 1)}
                   for rank, (hit, score) in enumerate(zip(hits, scores))]
        ordered.sort(key=lambda hit: (-hit["rerank_score"], hit.get("rrf_rank", hit["dense_rank"])))
        return ordered, {"model": "Qwen/Qwen3-Reranker-0.6B", "device": "GPU", "count": len(hits),
                         "batch_size": self.batch_size, "total_ms": (time.perf_counter() - started) * 1000,
                         "input_token_max": longest, "score_type": "yes_no_softmax_not_calibrated"}

    def close(self):
        self.model = self.tokenizer = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
