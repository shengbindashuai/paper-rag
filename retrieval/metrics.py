"""以固定原文字符区间判断命中，避免给不同分块长度硬套相同 chunk ID。

只计入“完整包含已标注证据”的片段。未标注的同义证据可能也正确，因此
这里的 Hit/MRR 是已知证据命中指标，不等价于穷尽标注后的完整相关性评估。
"""
import numpy as np


def contains_evidence(chunk: dict, evidence: dict) -> bool:
    if chunk["doc_id"] != evidence["doc_id"]:
        return False
    intervals = sorted(
        (s["segment_char_start"], s["segment_char_end"])
        for s in chunk["source_spans"] if s["segment_id"] == evidence["segment_id"]
    )
    # 一条证据可能横跨两个句子单元；仅允许间隔是空白，不能跨越遗漏的正文。
    cursor = evidence["start"]
    for start, end in intervals:
        if end <= cursor:
            continue
        if start > cursor and evidence["segment_text"][cursor:min(start, evidence["end"])].strip():
            return False
        cursor = max(cursor, end)
        if cursor >= evidence["end"]:
            return True
    return False


def evaluate(scores: np.ndarray, chunks: list[dict], questions: list[dict], budget: int) -> dict:
    rows = []
    for index, question in enumerate(questions):
        # 稳定排序保证同分时实验可复现，不依赖随机打破并列。
        order = np.argsort(-scores[index], kind="stable").tolist()
        positives = {i for i, c in enumerate(chunks) if contains_evidence(c, question["evidence"])}
        ranks = [rank for rank, i in enumerate(order, 1) if i in positives]
        rank = min(ranks) if ranks else None
        selected, used_tokens = [], 0
        for i in order:
            cost = chunks[i]["token_count"]
            if used_tokens + cost > budget:
                break  # 固定预算下按排名顺序装入完整块，不偷偷截断块。
            selected.append(i)
            used_tokens += cost
        rows.append({
            "query_id": question["query_id"], "language": question["language"],
            "question": question["question"], "evidence_id": question["evidence_id"],
            "expected_doc_id": question["evidence"]["doc_id"],
            "evidence_available": bool(positives), "first_relevant_rank": rank,
            "hit1": rank == 1, "hit3": rank is not None and rank <= 3,
            "hit5": rank is not None and rank <= 5,
            "reciprocal_rank": 1 / rank if rank else 0,
            "budget_hit": bool(set(selected) & positives),
            "budget_used_tokens": used_tokens, "budget_chunk_count": len(selected),
            "top3_context_tokens": sum(chunks[i]["token_count"] for i in order[:3]),
            "top5": [{"chunk_id": chunks[i]["chunk_id"], "doc_id": chunks[i]["doc_id"],
                      "section": chunks[i]["section"], "pages": chunks[i]["pages"],
                      "score": float(scores[index, i]), "token_count": chunks[i]["token_count"],
                      "contains_known_evidence": i in positives,
                      "text_preview": chunks[i]["text"][:600]} for i in order[:5]],
        })

    def aggregate(subset):
        return {key: float(np.mean([r[key] for r in subset])) for key in
                ["evidence_available", "hit1", "hit3", "hit5", "reciprocal_rank", "budget_hit", "budget_used_tokens", "budget_chunk_count", "top3_context_tokens"]}

    return {"overall": aggregate(rows), "by_language": {
        lang: aggregate([r for r in rows if r["language"] == lang]) for lang in sorted({r["language"] for r in rows})
    }, "queries": rows}
