"""
데이터셋 유틸리티: JSONL 테스트 샘플 로드 함수만을 포함
"""

# 타입 힌트를 위한 typing 임포트
# JSON 파싱을 위한 json 모듈 임포트
import json
from typing import Any, Dict, List


def load_test_samples(
    jsonl_path: str, limit: int | None = None
) -> List[Dict[str, Any]]:
    """
    JSONL 테스트 샘플 로드 함수(테스트 데이터 읽기)
    """
    # 샘플 리스트 초기화
    samples: List[Dict[str, Any]] = []
    # JSONL 파일 열기
    with open(jsonl_path, "r", encoding="utf-8") as f:
        # 각 라인을 순회하며 처리
        for idx, line in enumerate(f):
            # JSON 파싱
            data = json.loads(line)
            # 스키마 명시: id, question, ground_truth, images_list
            samples.append(
                {
                    # 샘플 고유 ID 저장
                    "id": data.get("id"),
                    # 질문 텍스트 저장
                    "question": data.get("question"),
                    # 정답(참조) 저장
                    "ground_truth": data.get("ground_truth"),
                    # 정답 이미지 ID 리스트 저장(시각 리콜 계산용)
                    "images_list": data.get("images_list", []),
                }
            )
            # 개수 제한이 있으면 조기 종료
            if limit is not None and len(samples) >= limit:
                # 루프 중단
                break
    # 샘플 리스트 반환
    return samples
