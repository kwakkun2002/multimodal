from dataclasses import dataclass
from typing import List


# 샘플별 검색된 이미지 ID를 담는 소형 데이터클래스
@dataclass
class RetrievedImageIds:
    """
    단일 샘플에 대해 검색된 이미지 ID들을 담는 클래스

    Attributes:
        image_ids: 검색된 이미지 ID 리스트
    """

    # 벡터 검색 결과로 얻은 이미지 ID들
    image_ids: List[str]
