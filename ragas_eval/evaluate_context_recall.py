# 결과 저장과 로깅을 위한 json 모듈 임포트
import json

# 환경 변수와 경로 사용을 위한 os 모듈 임포트
import os

# 타입 힌트를 위한 typing 임포트
from typing import Any, Dict, List, Tuple

# LLM 준비는 클래스로 이동됨
from ragas_eval.context_recall_evaluator import ContextRecallEvaluator

# OpenAI LLM을 LangChain으로 사용 (이 파일에서는 직접 사용하지 않음)
# from langchain_openai import ChatOpenAI
# RAGAS 관련 임포트는 클래스로 이동하여 이 파일에선 사용하지 않음
# 벡터 저장소 구성은 클래스로 이동
from ragas_eval.dataset_utils import (
    load_test_samples,  # noqa: F401 (다른 엔트리에서 재사용 가능성 고려)
)

# 컨텍스트 생성은 별도 유틸로 분리됨
# 데이터셋 변환은 클래스로 이동됨
from ragas_eval.visual_metrics import compute_visual_recall


def evaluate_context_recall_at_k(
    jsonl_path: str,
    top_k: int = 10,
    sample_limit: int | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Context Recall@K 평가를 수행하는 주 함수(클래스 호출 래퍼)
    """
    # 평가기 인스턴스 생성
    evaluator = ContextRecallEvaluator()
    # 평균 점수 및 기본 상세 결과, 보조 데이터 계산
    avg_recall, details, samples, per_sample_retrieved_image_ids = (
        evaluator.evaluate_context_recall_at_k(
            jsonl_path=jsonl_path, top_k=top_k, sample_limit=sample_limit
        )
    )
    # Visual Recall@K 계산(정답 이미지 대비 검색된 이미지 비율)
    visual_scores: List[float] = []
    for idx, sample in enumerate(samples):
        gt_imgs = [str(x) for x in (sample.get("images_list") or [])]
        ret_imgs = (
            per_sample_retrieved_image_ids[idx]
            if idx < len(per_sample_retrieved_image_ids)
            else []
        )
        score = compute_visual_recall(gt_imgs, ret_imgs)
        visual_scores.append(score)
    avg_visual_recall = (
        float(sum(visual_scores) / len(visual_scores)) if visual_scores else 0.0
    )
    details["average_visual_recall@k"] = avg_visual_recall
    return avg_recall, details


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
    # K 값 환경 변수로 설정(기본 10)
    top_k = int(os.environ.get("EVAL_TOP_K", "10"))
    # 샘플 제한 환경 변수
    sample_limit_env = os.environ.get("EVAL_SAMPLE_LIMIT")
    # 정수 변환
    sample_limit = int(sample_limit_env) if sample_limit_env else None

    # 평가 실행
    avg_recall, details = evaluate_context_recall_at_k(
        jsonl_path=jsonl_path,
        top_k=top_k,
        # 인자 전달
        sample_limit=sample_limit,
    )

    # 구분선 출력
    print("\n==============================")
    # 평균 점수 출력
    print(f"Context Recall@{top_k}: {avg_recall:.4f}")
    # 구분선 출력
    print("==============================\n")
    # 상세 결과 JSON 예쁘게 출력
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 스크립트 직접 실행 시 진입점
    main()
