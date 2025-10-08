from dataclasses import dataclass


# Context Recall 평가 점수 결과를 담는 데이터클래스
@dataclass
class ContextRecallScoreResult:
    """
    Context Recall 평가 점수 결과를 담는 클래스

    Attributes:
        average_context_recall_at_k: 평균 Context Recall 점수 (0.0 ~ 1.0)
        k: 검색에 사용된 상위 k개의 결과 수
        num_samples: 평가에 사용된 샘플 개수
    """

    # 모든 샘플에 대한 평균 Context Recall 점수
    average_context_recall_at_k: float
    # 검색 시 사용한 top-k 값
    k: int
    # 평가에 포함된 샘플의 총 개수
    num_samples: int
