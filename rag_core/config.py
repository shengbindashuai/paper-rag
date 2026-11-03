"""集中保存路径和处理参数，避免把可调规则散落在处理代码里。"""
from dataclasses import asdict, dataclass
from pathlib import Path
import os


CODE_ROOT = Path(__file__).resolve().parents[1]


def default_root() -> Path:
    """默认将数据放在仓库内；旧工作区可用 --root 或 RAG_WORKSPACE 指定。"""
    return Path(os.environ.get("RAG_WORKSPACE", str(CODE_ROOT))).expanduser().resolve()


def config_file(root: Path, name: str) -> Path:
    """工作区覆盖配置优先，兼容旧版 RAG_code/config 布局。"""
    for directory in (root / "config", root / "RAG_code" / "config", CODE_ROOT / "config"):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return CODE_ROOT / "config" / name


def resource_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "config":
        return config_file(root, str(Path(*path.parts[1:])))
    if len(path.parts) > 1 and path.parts[:2] == ("RAG_code", "config"):
        return config_file(root, str(Path(*path.parts[2:])))
    return (root / path).resolve()


@dataclass(frozen=True)
class ProcessingConfig:
    # 四篇论文实验后的基线：384 是期望大小，512 是完整输入的硬上限。
    # 采用已下载的 Qwen Tokenizer 计数，不把“字符数”当作 token 数。
    target_tokens: int = 384
    max_tokens: int = 512
    overlap_tokens: int = 64
    # 重叠以完整句子为单位，因此这是上限，不承诺每块恰好重叠 64 token。
    pipeline_version: str = "single-paper-v2"
    tokenizer_revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

    def validate(self) -> None:
        if not 0 <= self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("要求 0 <= overlap < target <= max_tokens")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Workspace:
    root: Path
    model_dir: Path | None = None

    @property
    def rag(self) -> Path:
        return self.root / "RAG"

    @property
    def tokenizer(self) -> Path:
        if self.model_dir is not None:
            return self.model_dir.resolve()
        canonical = self.rag / "models" / "Qwen3-Embedding-0.6B"
        legacy = self.rag / "embedding_step1" / "model"
        return legacy if not canonical.exists() and legacy.exists() else canonical

    def outputs(self, doc_id: str) -> dict[str, Path]:
        return {
            "extracted": self.rag / "extracted" / f"{doc_id}.json",
            "cleaned": self.rag / "cleaned" / f"{doc_id}.json",
            "chunks": self.rag / "chunks" / f"{doc_id}.jsonl",
            "report": self.rag / "metadata" / f"{doc_id}.processing.json",
        }
