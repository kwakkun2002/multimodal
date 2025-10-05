import json
import os

from ragas_eval.context_recall_evaluator import ContextRecallEvaluator
from ragas_eval.visual_recall_evaluator import VisualRecallEvaluator


def main():
    """
    CLI 진입점 함수(메인 실행)
    """
    # 평가 데이터 경로를 환경 변수로부터 읽기(기본값 제공)
    jsonl_path = os.environ.get(
        "WIT_MQA_JSONL",
        # 기본 경로 설정
        "/home/kun/Desktop/multimodal/data/MRAMG-Bench/wit_mqa.jsonl",
    )

    # Context Recall 평가기 인스턴스 생성
    context_evaluator = ContextRecallEvaluator()
    # 평균 점수 및 기본 상세 결과, 보조 데이터 계산
    avg_recall, details, samples, per_sample_retrieved_image_ids = (
        context_evaluator.evaluate_context_recall_at_k(
            jsonl_path=jsonl_path
        )
    )

    # Visual Recall 평가기 인스턴스 생성
    visual_evaluator = VisualRecallEvaluator()
    # Visual Recall@K 점수를 상세 결과에 추가
    visual_evaluator.add_visual_recall_to_details(
        details, samples, per_sample_retrieved_image_ids
    )

    # 구분선 출력
    print("\n==============================")
    # 평균 점수 출력
    print(f"Context Recall@10: {avg_recall:.4f}")
    # 구분선 출력
    print("==============================\n")
    # 상세 결과 JSON 예쁘게 출력
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 스크립트 직접 실행 시 진입점
    main()
