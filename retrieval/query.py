"""Python 便捷入口；长驻应用复用同一个 PaperRetriever 实例以保持 128 常驻。"""
from pathlib import Path
from rag_core.config import config_file
from .config import RetrievalConfig
from .service import RetrievalService


class PaperRetriever(RetrievalService):
    def __init__(self, root: Path, index_dir: Path | None = None, config_path: Path | None = None):
        root = root.resolve()
        config = RetrievalConfig.load(config_path or config_file(root, "retrieval.json"), root)
        if index_dir is not None:
            config.index_dir = str(index_dir.resolve())
        super().__init__(config)
        self.start()

    def search(self, query, top_k=5, **options):
        if isinstance(query, str):
            return super().search({"query": query, "top_k": top_k, **options})
        return super().search(query)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
