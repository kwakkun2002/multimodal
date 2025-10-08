"""
시각 메트릭: Visual Recall 계산 함수만을 포함
"""

# 타입 힌트를 위한 typing 임포트
from typing import List


def compute_visual_recall(
    ground_truth_image_ids: List[str], retrieved_image_ids: List[str]
) -> float:
    """
    시각 리콜(Visual Recall) 단일 샘플 계산 함수
    """
    # 정답 이미지 집합화(문자열 통일)
    gt_set = set(str(x) for x in (ground_truth_image_ids or []))
    # 정답 이미지가 없다면 리콜 정의상 0으로 반환
    if len(gt_set) == 0:
        return 0.0
    # 검색 이미지 집합화(문자열 통일)
    ret_set = set(str(x) for x in (retrieved_image_ids or []))
    # 교집합 크기(일치하는 이미지 수)
    hit = len(gt_set & ret_set)
    # 정답 대비 검색된 비율 반환
    return float(hit) / float(len(gt_set))
