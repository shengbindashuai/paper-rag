"""全库用的保守阅读顺序和入库筛选；不把英文词数规则套到中文上。"""
from collections import Counter
from copy import deepcopy
import re

from .chunking import token_count


def order_columns(raw: dict) -> tuple[dict, dict]:
    """识别常见双栏，按跨栏块分带，再在每带内先左后右。

    只改变交给清洗器的块顺序，原始提取另行完整保留。三栏/跨栏混合块
    无法仅靠这些规则保证还原，记录风险并在筛选阶段排除相应块。
    """
    ordered = deepcopy(raw)
    reordered, mixed = [], []
    for page in ordered["pages"]:
        width, mid = page["width"], page["width"] / 2
        blocks = page["blocks"]
        narrative = [b for b in blocks if len(b["text"]) > 100
                     and width * .23 < b["bbox"][2] - b["bbox"][0] < width * .6]
        left = [b for b in narrative if b["bbox"][2] < mid + width * .04]
        right = [b for b in narrative if b["bbox"][0] > mid - width * .04]
        if len(left) < 2 or len(right) < 2:
            continue
        reordered.append(page["page"])
        wide, narrow = [], []
        for block in blocks:
            bbox = block["bbox"]
            if bbox[2] - bbox[0] > width * .7:
                wide.append(block)
                # 同一原始块若同时混入左右栏的短行，不能当成普通跨栏段落。
                ls = [l for l in block["lines"] if len(l["text"].strip()) > 20]
                if (any(l["bbox"][2] < mid + width * .04 for l in ls)
                        and any(l["bbox"][0] > mid - width * .04 for l in ls)):
                    mixed.append(block["block_id"])
            else:
                narrow.append(block)

        def columns(items):
            return sorted(items, key=lambda b: (int((b["bbox"][0] + b["bbox"][2]) / 2 >= mid), b["bbox"][1], b["bbox"][0]))

        result = []
        for separator in sorted(wide, key=lambda b: b["bbox"][1]):
            cutoff = (separator["bbox"][1] + separator["bbox"][3]) / 2
            preceding = [b for b in narrow if (b["bbox"][1] + b["bbox"][3]) / 2 < cutoff]
            result.extend(columns(preceding))
            result.append(separator)
            used = {b["block_id"] for b in preceding}
            narrow = [b for b in narrow if b["block_id"] not in used]
        result.extend(columns(narrow))
        page["blocks"] = result
    return ordered, {"two_column_reordered_pages": reordered, "mixed_column_block_ids": mixed,
                     "method": "two-column-bands-v1; native order retained on other pages"}


def screen_segments(cleaned: dict, tokenizer, layout: dict) -> dict:
    reasons = Counter()
    mixed = set(layout["mixed_column_block_ids"])
    for segment in cleaned["segments"]:
        reason = None
        text = segment["text"]
        path = " ".join(segment["section_path"])
        if not segment["retrieval_eligible"]:
            reason = "parser_marked_ineligible"
        elif any(a["block_id"] in mixed for a in segment["sources"]):
            reason = "mixed_column_block"
        elif re.search(r"acknowledg|bibliography|references|author contributions|funding|致谢|参考文献", path, re.I):
            reason = "non_evidence_section"
        elif re.match(r"(?:We (?:thank|are grateful)|Acknowledg|©|Copyright)", text, re.I):
            reason = "acknowledgement_or_copyright"
        elif token_count(tokenizer, text) < 20:
            reason = "short_fragment"
        else:
            chinese = len(re.findall(r"[\u3400-\u9fff]", text))
            words = re.findall(r"[A-Za-z]+", text.lower())
            function = set("the of to in for with and a an is are which that we this by from as on it be".split())
            if chinese < 12 and (len(words) < 8 or sum(w in function for w in words) < 2):
                reason = "non_narrative_fragment"
        if reason:
            segment["retrieval_eligible"] = False
            reasons[reason] += 1
        segment["library_exclusion_reason"] = reason
    return dict(reasons)
