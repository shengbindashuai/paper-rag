"""Create a synthetic PDF to exercise ingestion without distributing research papers."""
import argparse
from pathlib import Path
import sys
import textwrap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.config import default_root


PARAGRAPHS = [
    "This document is a synthetic software demonstration. It is not a scientific publication and it does not report experimental results. The text exists only to check document extraction, indexing and retrieval in a fresh local workspace.",
    "In this demonstration, retrieval augmented generation (RAG) starts with a document collection. A preprocessing step extracts the text and retains source locations. The system splits the text into passages that can be retrieved and inspected independently.",
    "The demonstration library stores the source identifier, passage identifier and page number with every passage. A reader can use those fields to return to the source text. The archived document remains available for checking the original wording.",
    "The local embedding model converts each passage into a normalized vector. A query is converted with the same model weights. The index compares the query vector with document vectors and returns candidate passages for the next retrieval stage.",
    "A keyword channel can retrieve passages containing the words in the query. The demonstration combines the vector and keyword rankings before reranking the candidates. These software steps do not establish the scientific validity of the retrieved text.",
    "The quality gate is a separate optional external service. For the first local demonstration, the command explicitly disables that gate so no API key is needed. The output then identifies the evidence as not reviewed by the external quality gate.",
    "The service returns evidence passages and source metadata. A connected assistant can use the evidence to prepare a response. This repository does not include a separate local answer generation model, and the demonstration does not fabricate a research conclusion.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args()
    destination = args.root.resolve() / "RAG/inbox/demo.pdf"
    if destination.exists():
        raise FileExistsError(f"Demo already exists: {destination}")
    import pymupdf
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as document:
        page = document.new_page()
        y = 55
        page.insert_text((50, y), "Synthetic RAG demonstration", fontsize=16)
        y += 35
        page.insert_text((50, y), "Abstract", fontsize=12)
        y += 25
        for paragraph in PARAGRAPHS:
            lines = textwrap.wrap(paragraph, width=88)
            page.insert_text((50, y), "\n".join(lines), fontsize=10, lineheight=1.4)
            y += len(lines) * 14 + 17
        document.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
