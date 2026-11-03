"""保守清洗与结构分类。

原则：删除的块要有日志；正文修改要有记录；无法还原的公式、表格和图形文字
保留来源并标记人工复核。该版本不尝试把二维公式“猜”成准确的 LaTeX。
"""
from collections import Counter
import re
import unicodedata


LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
MATH_FONT = re.compile(r"CMMI|CMSY|CMEX|MSAM|MSBM")
CAPTION = re.compile(r"^(Figure|Fig\.|Table)\s+(\d+)(?:\s*[:.]|\s+)")
SECTION = re.compile(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+([A-Z].{1,110})$")


def _normal(text: str) -> str:
    # 不对整段使用 NFKC：它可能把上标等有数学含义的字符改写掉。
    return unicodedata.normalize("NFC", text.translate(LIGATURES)).replace("\u00ad", "")


def _flat(block: dict) -> str:
    return re.sub(r"\s+", " ", _normal(block["text"])).strip()


def _spans(block: dict) -> list[dict]:
    return [span for line in block["lines"] for span in line["spans"]]


def _anchor(page: dict, block: dict) -> dict:
    return {"page": page["page"], "block_id": block["block_id"],
            "bbox": block["bbox"], "line_ids": [line["line_id"] for line in block["lines"]]}


def _heading(block: dict, body_font: float) -> tuple[str, int] | None:
    text = _flat(block)
    special = {label.lower(): label for label in ["Abstract", "References", "Bibliography", "Appendix", "Acknowledgements", "Acknowledgments", "摘要", "参考文献", "致谢"]}
    if text.lower() in special:
        return special[text.lower()], 1
    spans = _spans(block)
    # Nimbus 的 Medi 也是粗体字重，ADOPT 论文使用此命名。
    bold = any(re.search(r"BX|Bold|Medi|Demi", span["font"]) for span in spans)
    size = max(span["size"] for span in spans)
    match = SECTION.fullmatch(text)
    if match and bold and size >= body_font * .98 and len(text.split()) <= 16:
        label = match[1]
        return text, label.count(".") + 1
    return None


def clean_lines(lines: list[dict], lexicon: set[str], preserve_lines: bool) -> tuple[str, list[str], list[dict]]:
    """合并排版换行；断词只在文档内存在完整拼写时移除行末连字符。"""
    changes, flags, output = [], set(), ""
    for line in lines:
        original = line["text"]
        text = _normal(original).replace("\u00a0", " ").strip()
        if text != original.strip():
            changes.append({"action": "normalize_ligature_or_space", "line_id": line["line_id"]})
        controls = sorted({ord(char) for char in text if unicodedata.category(char) == "Cc"})
        if controls:
            flags.add("unmapped_pdf_glyph")
            text = "".join(f"[PDF_GLYPH_{ord(c):04X}]" if unicodedata.category(c) == "Cc" else c for c in text)
        if not text:
            continue
        if not output:
            output = text
        elif preserve_lines:
            output += "\n" + text
        elif output.endswith("-") and re.match(r"[a-z]", text):
            left = re.search(r"([A-Za-z]+)-$", output)
            right = re.match(r"([a-z]+)", text)
            whole_word = (left[1] + right[1]).lower() if left and right else ""
            direct_evidence = whole_word in lexicon
            # including 的原形 include 若在文档内出现，也可作为断词合并证据。
            # 仅支持这一条明确的形态规则，不使用自动拼写纠错改写论文原词。
            morphology_evidence = bool(right and right[1] == "ing" and left and (left[1].lower() + "e") in lexicon)
            if left and right and (direct_evidence or morphology_evidence):
                output = output[:-1] + text
                action = "dehyphenate_with_document_evidence" if direct_evidence else "dehyphenate_with_document_morphology"
                changes.append({"action": action, "line_id": line["line_id"],
                                "word": left[1] + right[1]})
            else:
                # 保留 fine-tuning 等真实连字符，宁可标记，也不凭字形猜词。
                output += text
                flags.add("line_end_hyphen_preserved")
        else:
            output += " " + text
    return output.strip(), sorted(flags), changes


def _layout_kind(block: dict, body_font: float) -> str | None:
    """只做可解释的版面启发式识别，不宣称识别了图中的实际数据。"""
    text = _flat(block)
    spans = _spans(block)
    total = sum(len(span["text"]) for span in spans) or 1
    math_fraction = sum(len(s["text"]) for s in spans if MATH_FONT.search(s["font"])) / total
    words = re.findall(r"[A-Za-z]{3,}", text)
    sizes = [s["size"] for s in spans]
    line_lengths = [len(line["text"].strip()) for line in block["lines"]]
    # 显示公式的碎块/编号；有大量普通词的段落仍保留为正文，并另标 inline math。
    if re.fullmatch(r"[,.]?\s*\(\d+\)", text) or (math_fraction > 0.25 and len(words) < 5):
        return "equation"
    if len(line_lengths) >= 6 and sum(n < 45 for n in line_lengths) / len(line_lengths) > 0.8:
        return "table_candidate"
    if max(sizes) < body_font * 0.95 and len(words) < 8:
        return "figure_labels"
    return None


def clean_document(raw: dict, title: str) -> dict:
    body = raw["body_font_size"]
    all_text = "\n".join(_normal(b["text"]) for p in raw["pages"] for b in p["blocks"])
    lexicon = {word.lower() for word in re.findall(r"\b[A-Za-z]{3,}\b", all_text)}
    # 只在页边重复出现的块上应用重复页眉/页脚规则，正文重复内容不删除。
    marginal: Counter = Counter()
    for page in raw["pages"]:
        seen = {_flat(b) for b in page["blocks"]
                if b["bbox"][3] < page["height"] * 0.1 or b["bbox"][1] > page["height"] * 0.9}
        marginal.update(seen)
    segments, removed, sections = [], [], []
    section_path, parent_id = ["Front matter"], f"{raw['doc_id']}:front"
    before_abstract, in_references, in_algorithm = True, False, False
    layout_warnings = []
    for page in raw["pages"]:
        # 长正文块若明显左右并排，本版本只报告，不伪装成支持通用多栏解析。
        narrative = [b for b in page["blocks"] if len(re.findall(r"[A-Za-z]{3,}", _flat(b))) >= 15
                     and page["width"] * 0.3 < b["bbox"][2] - b["bbox"][0] < page["width"] * 0.65]
        if any(a["bbox"][2] < b["bbox"][0] and min(a["bbox"][3], b["bbox"][3]) > max(a["bbox"][1], b["bbox"][1])
               for a in narrative for b in narrative):
            layout_warnings.append({"page": page["page"], "reason": "possible_multicolumn_reading_order"})
        for block in page["blocks"]:
            text, bbox = _flat(block), block["bbox"]
            rotated = any(abs(line["direction"][1]) > 0.1 for line in block["lines"])
            reason = None
            if not text:
                reason = "empty_block"
            elif rotated and "arxiv:" in text.lower() and bbox[2] < page["width"] * 0.15:
                reason = "arxiv_rotated_margin"
            elif text.isdigit() and bbox[1] > page["height"] * 0.88:
                reason = "page_number"
            elif marginal[text] >= 3:
                reason = "repeated_margin_block"
            if reason:
                removed.append({**_anchor(page, block), "reason": reason, "raw_text": block["text"]})
                continue

            heading = _heading(block, body)
            if heading:
                label, level = heading
                before_abstract = False
                in_algorithm = False
                in_references = label in {"References", "Bibliography", "参考文献"}
                section_path = section_path[:level - 1] + [label]
                parent_id = f"{raw['doc_id']}:section:{len(sections) + 1:02}"
                sections.append({"parent_id": parent_id, "path": list(section_path), "start_page": page["page"]})
                kind = "heading"
            elif before_abstract and re.match(r"^(?:Abstract|摘要)[\s:：.—-]", text, re.I):
                # 有些 PDF 把“Abstract”和摘要正文放在同一个块内；全文保留。
                before_abstract = False
                section_path = ["Abstract"]
                parent_id = f"{raw['doc_id']}:section:{len(sections) + 1:02}"
                sections.append({"parent_id": parent_id, "path": list(section_path), "start_page": page["page"]})
                kind = "paragraph"
            elif before_abstract:
                kind = "front_matter"
            elif in_references:
                kind = "reference"
            elif re.match(r"^Algorithm\s+\d+", text):
                in_algorithm, kind = True, "algorithm"
            elif in_algorithm:
                kind = "algorithm"
            elif CAPTION.match(text) and (
                re.match(r"^(?:Figure|Fig\.|Table)\s+\d+\s*[:.]", text)
                or max(s["size"] for s in _spans(block)) < body * 0.95
                or re.search(r"Bold|Medi|Demi|BX", _spans(block)[0]["font"])
            ):
                kind = "table_caption" if text.startswith("Table") else "figure_caption"
            else:
                kind = _layout_kind(block, body) or "paragraph"
                # 小字号的长脚注仍是有意义的文本，不能当作图表刻度删除。
                if kind == "paragraph" and max(s["size"] for s in _spans(block)) < body * 0.9 and bbox[1] > page["height"] * 0.8:
                    kind = "footnote"
            preserve = kind in {"equation", "table_candidate", "figure_labels", "algorithm"}
            cleaned, flags, changes = clean_lines(block["lines"], lexicon, preserve)
            if any(MATH_FONT.search(s["font"]) for s in _spans(block)) and kind in {"paragraph", "footnote"}:
                flags = sorted(set(flags) | {"inline_math_plain_text"})
                # 分式上下堆叠时，两个不同“行”的纵向范围会交叠。
                # 仅按换行拼接会丢失分子/分母关系，因此整段需要查 PDF。
                line_boxes = [line["bbox"] for line in block["lines"]]
                stacked = any(min(a[3], b[3]) - max(a[1], b[1]) > 3
                              and abs(a[1] - b[1]) > 3 for a, b in zip(line_boxes, line_boxes[1:]))
                if stacked:
                    flags = sorted(set(flags) | {"inline_math_layout_requires_review"})
            if kind in {"equation", "table_candidate", "figure_labels", "algorithm"}:
                flags = sorted(set(flags) | {"layout_requires_visual_review"})
            item = {"segment_id": f"{raw['doc_id']}:s{len(segments) + 1:04}", "kind": kind,
                    "parent_id": parent_id, "section_path": list(section_path), "text": cleaned,
                    "sources": [{**_anchor(page, block), "segment_char_start": 0, "segment_char_end": len(cleaned)}],
                    "quality_flags": flags, "changes": changes,
                    "retrieval_eligible": kind in {"paragraph", "footnote", "figure_caption", "table_caption"}
                    and not {"unmapped_pdf_glyph", "inline_math_layout_requires_review"}.intersection(flags)}
            # 同一公式、图表或算法常有多个紧邻碎块，合并便于人工复核。
            previous = segments[-1] if segments else None
            same_group = previous and previous["kind"] == kind and previous["parent_id"] == parent_id
            same_page = previous and previous["sources"][-1]["page"] == page["page"]
            if same_group and same_page and preserve:
                offset = len(previous["text"]) + 1
                previous["text"] += "\n" + cleaned
                previous["sources"].extend({**a, "segment_char_start": a["segment_char_start"] + offset,
                                             "segment_char_end": a["segment_char_end"] + offset} for a in item["sources"])
                previous["quality_flags"] = sorted(set(previous["quality_flags"] + flags))
                previous["changes"].extend(changes)
            else:
                # 数学上下标可能使一段正文被 PDF 拆成纵向交叠的几个块。
                overlapping_prose = same_group and same_page and kind == "paragraph"
                # 左栏末尾和右栏开头在 y 方向也会“交叠”，必须同时检查横向交集。
                previous_box = previous["sources"][-1]["bbox"] if previous else bbox
                horizontal_overlap = min(bbox[2], previous_box[2]) - max(bbox[0], previous_box[0])
                overlapping_prose = overlapping_prose and horizontal_overlap > min(bbox[2] - bbox[0], previous_box[2] - previous_box[0]) * .4
                overlapping_prose = overlapping_prose and bbox[1] < max(a["bbox"][3] for a in previous["sources"]) - 2
                # 跨页连续正文仅在上一段无句末标点、下一段小写起头时合并。
                continuation = same_group and kind == "paragraph" and not same_page
                continuation = continuation and page["page"] == previous["sources"][-1]["page"] + 1
                continuation = continuation and previous["text"][-1:] not in ".!?。！？:;"
                continuation = continuation and bool(re.match(r"^[a-z]", cleaned))
                if continuation or overlapping_prose:
                    offset = len(previous["text"]) + 1
                    previous["text"] += " " + cleaned
                    previous["sources"].extend({**a, "segment_char_start": a["segment_char_start"] + offset,
                                                 "segment_char_end": a["segment_char_end"] + offset} for a in item["sources"])
                    previous["quality_flags"] = sorted(set(previous["quality_flags"] + flags))
                    action = "join_cross_page_continuation" if continuation else "join_overlapping_prose_blocks"
                    previous["changes"].extend(changes + [{"action": action}])
                    previous["retrieval_eligible"] = previous["retrieval_eligible"] and item["retrieval_eligible"]
                    if overlapping_prose and "inline_math_plain_text" in previous["quality_flags"]:
                        previous["quality_flags"] = sorted(set(previous["quality_flags"]) | {"inline_math_layout_requires_review"})
                        previous["retrieval_eligible"] = False
                else:
                    segments.append(item)
    # 图表说明与邻接的版面块建立关联；后续补读父段时不会只取表值而漏掉标题。
    for index, segment in enumerate(segments):
        segment["related_segment_ids"] = []
        if segment["kind"] not in {"figure_caption", "table_caption"}:
            continue
        cursor = index - 1
        while cursor >= 0:
            previous = segments[cursor]
            if previous["kind"] not in {"equation", "table_candidate", "figure_labels"} or previous["parent_id"] != segment["parent_id"]:
                break
            if previous["sources"][-1]["page"] != segment["sources"][0]["page"]:
                break
            segment["related_segment_ids"].append(previous["segment_id"])
            previous.setdefault("related_segment_ids", []).append(segment["segment_id"])
            cursor -= 1
    return {"schema_version": 1, "doc_id": raw["doc_id"], "title": title,
            "source_file": raw["source_file"], "source_sha256": raw["source_sha256"],
            "sections": sections, "segments": segments, "removed_blocks": removed,
            "layout_warnings": layout_warnings,
            "limitations": ["Single-column native-text PDF pilot; no OCR or verified math/table reconstruction.",
                            "Math, tables, figure labels and pseudocode are retained with review flags."]}
