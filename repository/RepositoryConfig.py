from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WitImagesStorageConfig:  # MinIO 연결 설정을 담는 데이터 클래스
    """MinIO 연결과 버킷 정보를 담는 설정 클래스"""

    endpoint: str = (
        "localhost:9000"  # MinIO 서버 주소 (도커 컴포즈 내부면 "minio:9000")
    )
    access_key: str = "minioadmin"  # MinIO 액세스 키
    secret_key: str = "minioadmin"  # MinIO 시크릿 키
    secure: bool = False  # HTTP 사용 시 False, HTTPS 사용 시 True
    bucket_name: str = (
        "wit-images"  # 사용할 버킷 이름 (MinIO 규칙에 맞는 소문자+하이픈)
    )


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
