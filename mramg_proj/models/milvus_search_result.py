from dataclasses import dataclass
from typing import List


@dataclass
class MilvusSearchResult:
    """
    Milvus 검색 결과를 담는 데이터 클래스

    Attributes:
        id: Milvus에서 자동 생성된 고유 ID
        original_id: 원본 문서의 ID
        text: 검색된 텍스트 내용
        image_ids: 문서에 포함된 이미지 ID 리스트
        score: 검색 유사도 점수 (높을수록 더 유사함)
    """

    # Milvus에서 자동 생성된 고유 ID
    id: int

    # 원본 문서의 ID
    original_id: int

    # 검색된 텍스트 내용
    text: str

    # 문서에 포함된 이미지 ID 리스트
    image_ids: List[str]

    # 검색 유사도 점수 (높을수록 더 유사함)
    score: float
