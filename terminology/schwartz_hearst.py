"""依据 Schwartz & Hearst (PSB 2003) 的双向括号候选与反向字符对齐。

本实现从连续清洗段落提取，不跨段拼接；字符区间指向输入的原始 Python 字符串。
来源：https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf
"""
import re


def valid_short_form(text):
    return (2 <= len(text) <= 10 and len(text.split()) <= 2
            and text[0].isalnum() and any(c.isalpha() for c in text))


def align_long_form(short_form, long_form):
    """返回最短匹配全称的开始位置和字符对齐；不改写、归一化源字符串。"""
    if not valid_short_form(short_form):
        return None
    cursor, alignment = len(long_form) - 1, []
    for short_index in range(len(short_form) - 1, -1, -1):
        char = short_form[short_index]
        if not char.isalnum():
            continue
        while cursor >= 0:
            same = long_form[cursor].lower() == char.lower()
            at_boundary = short_index != 0 or cursor == 0 or not long_form[cursor - 1].isalnum()
            if same and at_boundary:
                break
            cursor -= 1
        if cursor < 0:
            return None
        alignment.append((short_index, cursor))
        cursor -= 1
    # 对齐到连字符词的内部边界时，仍保留该完整的空白分隔词。
    start = cursor + 1
    while start > 0 and not long_form[start - 1].isspace():
        start -= 1
    matched = long_form[start:]
    if len(matched) < len(short_form):
        return None
    if short_form.casefold() in [w.casefold() for w in matched.split()]:
        return None
    if len(matched.split()) > min(len(short_form) + 5, len(short_form) * 2):
        return None
    return start, list(reversed(alignment))


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def extract_pairs(text):
    """保留每次定义出现；重复配对交由候选表按来源区分，而不是全局覆盖。"""
    stack, brackets = [], []
    for position, char in enumerate(text):
        if char in "(（":
            stack.append((position, char))
        elif char in ")）" and stack:
            opening, kind = stack[-1]
            if (kind, char) in {("(", ")"), ("（", "）")}:
                stack.pop()
                brackets.append((opening, position))
    results, seen = [], set()
    for opening, closing in sorted(brackets):
        inner_start, inner_end = _trim(text, opening + 1, closing)
        inside = text[inner_start:inner_end]
        if not inside:
            continue
        # 不跨明确句末/空段落搜全称。单个换行可来自 PDF 的同段换行。
        boundary = 0
        for match in re.finditer(r"[.!?。！？]\s+|\n\s*\n", text[:opening]):
            boundary = match.end()
        words = list(re.finditer(r"\S+", text[boundary:opening]))
        candidates = []
        if valid_short_form(inside) and words:
            maximum = min(len(inside) + 5, len(inside) * 2)
            left = boundary + words[-maximum].start() if len(words) >= maximum else boundary + words[0].start()
            lf_start, lf_end = _trim(text, left, opening)
            candidates.append((inner_start, inner_end, lf_start, lf_end, "long(short)"))
        # 反向形式：括号内全称至少三词，或已超出合法缩写长度。
        if (len(inside.split()) > 2 or not valid_short_form(inside)) and words:
            sf_start, sf_end = _trim(text, boundary + words[-1].start(), boundary + words[-1].end())
            candidates.append((sf_start, sf_end, inner_start, inner_end, "short(long)"))
        for sf_start, sf_end, lf_start, lf_end, pattern in candidates:
            short_form = text[sf_start:sf_end]
            match = align_long_form(short_form, text[lf_start:lf_end])
            if match is None:
                continue
            relative, alignment = match
            lf_start += relative
            key = (sf_start, sf_end, lf_start, lf_end)
            if key in seen:
                continue
            seen.add(key)
            long_form = text[lf_start:lf_end]
            flags = []
            if not any(c.isupper() for c in short_form):
                flags.append("lowercase_short_form")
            if any(c.isdigit() for c in short_form):
                flags.append("contains_digits")
            if len(long_form.split()) == 1:
                flags.append("single_word_long_form")
            if any(c in long_form for c in "()（）=∑∫"):
                flags.append("parenthesis_or_math_in_long_form")
            results.append({"short_form": short_form, "long_form": long_form,
                            "short_start": sf_start, "short_end": sf_end,
                            "long_start": lf_start, "long_end": lf_end,
                            "pattern": pattern, "flags": flags,
                            "alignment": [{"short_index": s, "segment_index": lf_start - relative + l}
                                          for s, l in alignment]})
    return results
