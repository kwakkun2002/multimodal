from dataclasses import dataclass
from typing import List


# 테스트 샘플 데이터를 담는 소형 데이터클래스
@dataclass
class TestSample:
    """
    JSONL에서 로드한 단일 테스트 샘플을 담는 클래스
    Attributes:
        id: 샘플 고유 식별자
        question: 사용자 질문
        ground_truth: 정답 (참조 답변)
        images_list: 정답에 해당하는 이미지 ID 리스트
    """

    # 샘플의 고유 ID
    id: str
    # 사용자가 입력한 질문
    question: str
    # 모델이 생성해야 할 정답
    ground_truth: str
    # 정답과 관련된 이미지 ID들
    images_list: List[str]