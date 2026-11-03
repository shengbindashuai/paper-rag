"""RRF 融合与只调整顺序的 MMR；正文和原始分数始终保留。"""
import math
import numpy as np


def identity(hit):
    return hit["doc_id"], hit["chunk_id"]


def reciprocal_rank_fusion(dense, keyword, k=60):
    combined, first_seen = {}, {}
    for channel, hits in (("dense", dense), ("keyword", keyword)):
        seen = set()
        for rank, hit in enumerate(hits, 1):
            key = identity(hit)
            if key in seen:
                continue
            seen.add(key)
            if key not in combined:
                combined[key] = {**hit, "dense_rank": None, "keyword_rank": None, "rrf_score": 0.0}
                first_seen[key] = len(first_seen)
            result = combined[key]
            result["rrf_score"] += 1.0 / (k + rank)
            result[channel + "_rank"] = rank
            if channel == "keyword":
                result["keyword_score"] = hit["keyword_score"]
    ordered = sorted(combined.values(), key=lambda h: (-h["rrf_score"], first_seen[identity(h)]))
    return [{**h, "rrf_rank": rank} for rank, h in enumerate(ordered, 1)]


def mmr_reorder(hits, vectors, weight=0.7):
    """完整排列：永远迭代到所有输入片段都已输出，不按分数阈值删除。"""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("MMR lambda 必须在 0～1")
    if len({identity(h) for h in hits}) != len(hits):
        raise ValueError("MMR 输入存在重复文档/片段 ID")
    if not hits:
        return []
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(hits) or not np.isfinite(matrix).all():
        raise ValueError("MMR 向量数量或数值错误")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0):
        raise ValueError("MMR 不允许零向量")
    matrix = matrix / norms[:, None]
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    relevance = np.asarray([h.get("term_rerank_score", h.get("rerank_score", h.get("rrf_score", h["dense_score"]))) for h in hits], dtype=float)
    if not np.isfinite(relevance).all():
        raise ValueError("MMR 相关性分数无效")
    # 仅用于内部计算；保留输入的 dense_score / rerank_score / rrf_score 原值。
    span = float(np.ptp(relevance))
    relevance = (relevance - relevance.min()) / span if span > 0 else np.ones(len(hits))
    remaining, selected, output = list(range(len(hits))), [], []
    while remaining:
        def marginal(i):
            redundancy = max(0.0, float(similarity[i, selected].max())) if selected else 0.0
            return weight * float(relevance[i]) - (1.0 - weight) * redundancy
        # 同分保留重排顺序，避免平台相关的集合遍历差异。
        chosen = max(remaining, key=lambda i: (marginal(i), -i))
        output.append({**hits[chosen], "mmr_rank": len(output) + 1, "mmr_score": marginal(chosen)})
        selected.append(chosen)
        remaining.remove(chosen)
    if {identity(h) for h in output} != {identity(h) for h in hits} or len(output) != len(hits):
        raise RuntimeError("MMR 改变了候选集合")
    return output
