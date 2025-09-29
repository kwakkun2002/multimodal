from dataclasses import dataclass


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
