from dataclasses import dataclass


@dataclass
class DocWitConfig:
    """DocWit 벡터 저장소의 연결 및 컬렉션 설정을 담는 클래스"""

    milvus_host: str = "localhost"  # Milvus 서버 호스트 주소
    milvus_port: int = 19532  # Milvus 서버 포트 (도커 포트 매핑 19530->19532)
    collection_name: str = "doc_wit_documents"  # 사용할 컬렉션 이름
    embedding_dim: int = 1024  # BGE-M3 임베딩 차원
    chunk_size: int = 256  # 텍스트 청크 크기 (토큰 단위)
    chunk_overlap: int = 20  # 청크 간 중복 크기
    text_max_len: int = 8192  # 텍스트 최대 길이
    image_ids_max_len: int = 2048  # 이미지 ID 리스트 최대 길이
