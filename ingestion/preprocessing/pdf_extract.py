"""只负责提取，不做清洗：保留 PDF 中的原始文本、字体和页面坐标。

PDF 文本块不是可靠的语义段落。特别是公式可能被拆成多个块，图中的刻度
也会被提取出来。因此后续阶段需要分类，不能将这里的全部文本直接入库。
"""
from collections import Counter
from pathlib import Path

import pymupdf
from rag_core.io import file_sha256


def extract_pdf(pdf_path: Path, doc_id: str, source_file: str) -> dict:
    pages = []
    body_fonts: Counter = Counter()
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("PDF 已加密，需要先解锁")
        for page_index, page in enumerate(document):
            blocks = []
            # 保留原生块顺序，不直接用 sort=True 将二维公式的碎块按 y 坐标打乱。
            # 该版本的清洗逻辑面向单栏论文；多栏情况会在报告中标记。
            raw = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT, sort=False)
            for block_number, block in enumerate(raw["blocks"]):
                if block["type"] != 0:
                    continue
                block_id = f"{doc_id}:p{page_index + 1:02}:b{block_number:03}"
                lines = []
                for line_number, line in enumerate(block["lines"]):
                    spans = [
                        {"text": span["text"], "font": span["font"],
                         "size": round(span["size"], 4), "bbox": list(span["bbox"])}
                        for span in line["spans"]
                    ]
                    lines.append({
                        "line_id": f"{block_id}:l{line_number:03}",
                        "bbox": list(line["bbox"]), "direction": list(line["dir"]),
                        "text": "".join(span["text"] for span in spans), "spans": spans,
                    })
                    # 按字符数加权找正文常见字号；小图表刻度虽多，正文字符通常更多。
                    for span in spans:
                        if line["dir"][0] > 0.9 and 8 <= span["size"] <= 13:
                            body_fonts[round(span["size"], 1)] += len(span["text"])
                blocks.append({"block_id": block_id, "bbox": list(block["bbox"]),
                               "text": "\n".join(line["text"] for line in lines),
                               "lines": lines})
            pages.append({"page": page_index + 1, "width": page.rect.width,
                          "height": page.rect.height, "blocks": blocks})
        pdf_metadata = document.metadata
    if not body_fonts:
        raise ValueError("没有足够的文字层；该流程不自动执行 OCR")
    return {
        "schema_version": 1, "doc_id": doc_id, "source_file": source_file,
        "source_sha256": file_sha256(pdf_path), "page_count": len(pages),
        "extractor": {"name": "PyMuPDF", "version": pymupdf.VersionBind},
        "pdf_metadata": pdf_metadata,
        "body_font_size": body_fonts.most_common(1)[0][0], "pages": pages,
    }


def render_previews(pdf_path: Path, output_dir: Path, page_numbers: list[int]) -> list[str]:
    """渲染原页面供人工核查；不改写 PDF，不把渲染图冒充 OCR 结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(pdf_path) as document:
        for number in page_numbers:
            if not 1 <= number <= len(document):
                raise ValueError(f"不存在的页码：{number}")
            path = output_dir / f"page_{number:02}.png"
            document[number - 1].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(path)
            paths.append(str(path))
    return paths
