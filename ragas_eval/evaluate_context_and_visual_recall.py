import os

from ragas_eval.context_recall_evaluator import (
    ContextRecallEvaluator,
    RetrievedContextsResult,
)
from ragas_eval.visual_recall_evaluator import VisualRecallEvaluator

# 평가 데이터 경로를 환경 변수로부터 읽기(기본값 제공)
jsonl_path = os.environ.get(
    "WIT_MQA_JSONL",
    # 기본 경로 설정
    "/home/kun/Desktop/multimodal/data/MRAMG-Bench/wit_mqa.jsonl",
)


# Context Recall 평가기 인스턴스 생성
context_evaluator = ContextRecallEvaluator()

# 1단계: 테스트 샘플들에 대해 컨텍스트 검색 및 데이터 준비
retrieval_result: RetrievedContextsResult = (
    context_evaluator.retrieve_contexts_for_samples(jsonl_path=jsonl_path, limit=5)
)

# 2단계: 준비된 RAGAS 데이터셋으로 Context Recall 점수 계산
context_recall_result = context_evaluator.compute_context_recall_score(
    retrieval_result.ragas_evaluation_data
)

print(f"Average Context Recall: {context_recall_result.average_context_recall_at_k}")
print(f"Top-K: {context_recall_result.k}")
print(f"Number of Samples: {context_recall_result.num_samples}")


# Visual Recall 평가기 인스턴스 생성
visual_evaluator = VisualRecallEvaluator()

# 3단계: 검색된 이미지 ID를 사용하여 Visual Recall 점수 계산
# TestSample 객체 리스트와 SampleImageIds 객체 리스트를 전달
avg_visual_recall = visual_evaluator.evaluate_visual_recall_at_k(
    retrieval_result.original_test_samples,
    retrieval_result.retrieved_image_ids_per_sample,
)

print(f"Average Visual Recall: {avg_visual_recall}")
