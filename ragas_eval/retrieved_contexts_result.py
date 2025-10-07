from dataclasses import dataclass
from typing import List

from ragas_eval.ragas_evaluation_data import RagasEvaluationData
from ragas_eval.sample_image_ids import SampleImageIds
from ragas_eval.test_sample import TestSample


# 컨텍스트 검색 결과를 담는 데이터클래스
@dataclass
class RetrievedContextsResult:
    """
    테스트 샘플에 대한 컨텍스트 검색 결과를 담는 클래스

    Attributes:
        original_test_samples: JSONL에서 로드한 원본 테스트 샘플 리스트
        ragas_evaluation_data: RAGAS 평가를 위해 변환된 데이터셋 리스트
        retrieved_image_ids_per_sample: 각 샘플별로 검색된 이미지 ID 객체 리스트
    """

    # 원본 테스트 샘플 데이터 (TestSample 객체들)
    original_test_samples: List[TestSample]
    # RAGAS 평가용으로 구성된 데이터 (RagasEvaluationData 객체들)
    ragas_evaluation_data: List[RagasEvaluationData]
    # 각 샘플마다 검색된 이미지 ID들 (SampleImageIds 객체들)
    retrieved_image_ids_per_sample: List[SampleImageIds]
