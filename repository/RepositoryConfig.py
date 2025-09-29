from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class VectorDatabaseConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "openclip_multimodal"
    dim: int = 768
    text_max_len: int = 2048
    path_max_len: int = 1024
    metric_type: str = "IP"
    index_type: str = "HNSW"
    index_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.index_params is None:
            self.index_params = {"M": 16, "efConstruction": 200}
