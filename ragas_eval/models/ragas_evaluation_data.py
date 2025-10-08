from dataclasses import dataclass
from typing import List


# RAGAS 평가용 데이터를 담는 소형 데이터클래스
@dataclass
class RagasEvaluationData:
    """
    RAGAS 평가를 위한 단일 데이터 항목을 담는 클래스

    Attributes:
        user_input: 사용자 질문
        retrieved_contexts: 검색된 텍스트 컨텍스트 리스트
        reference: 정답 (참조 답변)
    """

    # 평가할 사용자 질문
    user_input: str
    # 벡터 스토어에서 검색된 컨텍스트들
    retrieved_contexts: List[str]
    # 정답 참조 텍스트
    reference: str