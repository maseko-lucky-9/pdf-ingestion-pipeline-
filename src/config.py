from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    collections_root: str = "./collections"
    cache_dir: str = "./.cache"


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    batch_size: int = 64


class ChunkerConfig(BaseModel):
    max_tokens: int = 900
    overlap_pct: float = 0.15


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "auto"


class RetrievalConfig(BaseModel):
    bm25_k: int = 50
    vector_k: int = 50
    rrf_k: int = 60
    final_k: int = 10


class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    ollama: OllamaConfig = OllamaConfig()
    chunker: ChunkerConfig = ChunkerConfig()
    rerank: RerankConfig = RerankConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    def collection_db_path(self, name: str) -> Path:
        return Path(self.paths.collections_root) / name / "index.db"

    def collection_error_log(self, name: str) -> Path:
        return Path(self.paths.collections_root) / name / "ingest_errors.log"


_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_cached: Config | None = None


def load_config(path: Path | None = None) -> Config:
    global _cached
    if _cached is None:
        p = path or _CONFIG_PATH
        if p.exists():
            raw = yaml.safe_load(p.read_text())
            _cached = Config.model_validate(raw)
        else:
            _cached = Config()
    return _cached
