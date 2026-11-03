"""按章节和句子打包片段，并记录每一段文本来自哪里。

这里仅加载 Tokenizer 计数，不加载 Qwen 模型权重，不生成 embedding。
text 是清洗后的证据正文；embedding_text 在正文前补充论文标题和章节。
"""
import re

from transformers import AutoTokenizer

from rag_core.config import ProcessingConfig


def load_tokenizer(path):
    if not (path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"本地 Tokenizer 不完整：{path}")
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


def token_count(tokenizer, text: str) -> int:
    # 与实际推理一致：Qwen 会自动附加一个结束标记，它也占输入预算。
    return len(tokenizer.encode(text, add_special_tokens=True))


def _header(title: str, section_path: list[str]) -> str:
    return f"Title: {title}\nSection: {' > '.join(section_path)}\n\n"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while start < end and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    # 这是透明的规则切句，不是语义模型；缩写可能产生较短单元，随后会重新打包。
    boundaries = [0] + [m.end() for m in re.finditer(r"[.!?。！？][\"'’”)]*\s+|\n+", text)] + [len(text)]
    return [(a, b) for a, b in (_trim(text, a, b) for a, b in zip(boundaries, boundaries[1:])) if a < b]


def _pieces(segment: dict, header: str, tokenizer, config: ProcessingConfig) -> list[dict]:
    """先保留整句；单句仍超长时，在空白边界二分，最后才使用字符边界。

    不截断 tokenizer 输出，也不 decode 半个 token，避免中文字符被破坏。
    每个 piece 的 start/end 都指向 cleaned JSON 中对应 segment 的字符区间。
    """
    text = segment["text"]
    result = []
    for start, end in _sentence_spans(text):
        while start < end:
            cut, hard_split = end, False
            if token_count(tokenizer, header + text[start:end]) > config.max_tokens:
                low, high = start + 1, end
                # 找一个满足 token 上限的字符前缀；之后还会检查实际 token 数。
                while low < high:
                    middle = (low + high + 1) // 2
                    if token_count(tokenizer, header + text[start:middle]) <= config.max_tokens:
                        low = middle
                    else:
                        high = middle - 1
                cut = low
                words = list(re.finditer(r"\s+", text[start:cut]))
                if words and words[-1].start() > (cut - start) // 2:
                    cut = start + words[-1].start()
                if cut <= start or token_count(tokenizer, header + text[start:cut]) > config.max_tokens:
                    raise ValueError("标题/章节占用过大，无法安全切出任何正文")
                hard_split = True
            a, b = _trim(text, start, cut)
            if a < b:
                result.append({"segment_id": segment["segment_id"], "start": a, "end": b,
                               "text": text[a:b], "hard_split": hard_split})
            start, _ = _trim(text, cut, end)
    return result


def _body(pieces: list[dict], segments: dict) -> str:
    """同段连续部分沿用原间隔；跨段插空行，避免两个词被意外黏在一起。"""
    parts = []
    for i, piece in enumerate(pieces):
        if i:
            previous = pieces[i - 1]
            if previous["segment_id"] == piece["segment_id"]:
                original = segments[piece["segment_id"]]["text"]
                parts.append(original[previous["end"]:piece["start"]])
            else:
                parts.append("\n\n")
        parts.append(piece["text"])
    return "".join(parts)


def chunk_document(cleaned: dict, tokenizer, config: ProcessingConfig) -> list[dict]:
    segment_map = {s["segment_id"]: s for s in cleaned["segments"]}
    groups: list[list[dict]] = []
    for segment in cleaned["segments"]:
        if segment["kind"] in {"heading", "front_matter"} or not segment["text"]:
            continue
        # 不把参考文献、公式碎块和图表坐标混进普通正文 chunk。
        family = "prose" if segment["kind"] == "paragraph" else segment["kind"]
        key = (segment["parent_id"], family, segment["retrieval_eligible"])
        previous_key = groups[-1][0]["_group_key"] if groups else None
        tagged = {**segment, "_group_key": key}
        if key != previous_key:
            groups.append([tagged])
        else:
            groups[-1].append(tagged)
    chunks = []
    for group in groups:
        header = _header(cleaned["title"], group[0]["section_path"])
        if token_count(tokenizer, header) >= config.max_tokens - 16:
            raise ValueError("标题/章节过长，正文预算不足")
        pieces = [p for s in group for p in _pieces(s, header, tokenizer, config)]
        current, overlap_count = [], 0

        def emit():
            if not current or len(current) <= overlap_count:
                return  # 不能只把上一块的重叠内容再输出成一个新块。
            text = _body(current, segment_map)
            used_segments = [segment_map[sid] for sid in dict.fromkeys(p["segment_id"] for p in current)]
            sources = []
            for piece in current:
                segment = segment_map[piece["segment_id"]]
                # 跨页段落可能在切块后只命中其中一页，不能把整段的所有页码照搬。
                anchors = [a for a in segment["sources"]
                           if a.get("segment_char_end", len(segment["text"])) > piece["start"]
                           and a.get("segment_char_start", 0) < piece["end"]]
                sources.append({"segment_id": piece["segment_id"],
                                "segment_char_start": piece["start"], "segment_char_end": piece["end"],
                                "text": piece["text"], "pdf_blocks": anchors})
            flags = {f for s in used_segments for f in s["quality_flags"]}
            if any(p["hard_split"] for p in current):
                flags.add("long_sentence_split")
            chunks.append({
                "schema_version": 1, "chunk_id": f"{cleaned['doc_id']}_{len(chunks) + 1:04}",
                "doc_id": cleaned["doc_id"], "parent_id": group[0]["parent_id"],
                "section": " > ".join(group[0]["section_path"]),
                "section_path": group[0]["section_path"], "kind": group[0]["_group_key"][1],
                "pages": sorted({a["page"] for source in sources for a in source["pdf_blocks"]}),
                "text": text, "embedding_text": header + text,
                "token_count": token_count(tokenizer, header + text),
                "body_token_count": token_count(tokenizer, text),
                "overlap_piece_count": overlap_count,
                "source_file": cleaned["source_file"], "source_sha256": cleaned["source_sha256"],
                "source_spans": sources, "quality_flags": sorted(flags),
                "related_segment_ids": sorted({sid for s in used_segments for sid in s.get("related_segment_ids", [])}),
                "retrieval_eligible": all(s["retrieval_eligible"] for s in used_segments),
                "pipeline_version": config.pipeline_version,
                "tokenizer": {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": config.tokenizer_revision},
            })

        for piece in pieces:
            proposal = current + [piece]
            count = token_count(tokenizer, header + _body(proposal, segment_map))
            over_target = current and len(current) > overlap_count and token_count(tokenizer, header + _body(current, segment_map)) >= config.target_tokens
            if current and (count > config.max_tokens or over_target):
                emit()
                # 只复制末尾完整句子，并对重叠本身计 token。图表/算法不复制重叠。
                tail = []
                if group[0]["_group_key"][1] == "prose":
                    for old in reversed(current):
                        candidate = [old] + tail
                        if token_count(tokenizer, _body(candidate, segment_map)) > config.overlap_tokens:
                            break
                        tail = candidate
                while tail and token_count(tokenizer, header + _body(tail + [piece], segment_map)) > config.max_tokens:
                    tail = tail[1:]
                current, overlap_count = tail, len(tail)
            current.append(piece)
        emit()
    for i, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[i - 1]["chunk_id"] if i else None
        chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i + 1 < len(chunks) else None
    return chunks
