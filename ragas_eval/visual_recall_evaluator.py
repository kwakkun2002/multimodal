from typing import List

from ragas_eval.models.sample_image_ids import RetrievedImageIds
from ragas_eval.models.test_sample import Mqa
from ragas_eval.utiles.visual_metrics import compute_visual_recall


class VisualRecallEvaluator:
    """
    Visual Recall@K 평가를 담당하는 클래스
    정답 이미지와 검색된 이미지를 비교하여 시각적 회상 정확도를 계산합니다.
    """

    def __init__(self):
        """
        VisualRecallEvaluator 인스턴스를 초기화합니다.
        """
        pass

    def evaluate_visual_recall_at_k(
        self,
        samples: List[Mqa],
        per_sample_retrieved_image_ids: List[RetrievedImageIds],
    ) -> float:
        """
        주어진 샘플들에 대해 Visual Recall@K 점수를 계산합니다.

        Args:
            samples: 평가할 TestSample 객체들의 리스트
            per_sample_retrieved_image_ids: 각 샘플별로 검색된 이미지 ID를 담은 SampleImageIds 객체 리스트

        Returns:
            float: 평균 Visual Recall@K 점수
        """

        # 각 샘플별 Visual Recall 점수를 저장할 리스트 초기화
        visual_scores: List[float] = []

        # 각 샘플을 순회하며 Visual Recall 점수 계산
        for idx, sample in enumerate(samples):
            # 현재 샘플의 정답 이미지 리스트를 문자열로 변환
            gt_imgs = [str(x) for x in sample.images_list]

            # 현재 샘플에 대응하는 검색된 이미지 ID 리스트 가져오기
            ret_imgs = (
                per_sample_retrieved_image_ids[idx].image_ids
                if idx < len(per_sample_retrieved_image_ids)
                else []
            )

            # 현재 샘플의 Visual Recall 점수 계산
            score = compute_visual_recall(gt_imgs, ret_imgs)

            # 계산된 점수를 리스트에 추가
            visual_scores.append(score)

        # 평균 Visual Recall 점수 계산 (점수가 없으면 0.0 반환)
        avg_visual_recall = (
            float(sum(visual_scores) / len(visual_scores)) if visual_scores else 0.0
        )

        return avg_visual_recall

    def add_visual_recall_to_details(
        self,
        samples: List[Mqa],
        per_sample_retrieved_image_ids: List[RetrievedImageIds],
    ) -> float:
        """
        평가 상세 결과에 Visual Recall@K 점수를 추가합니다.

        Args:
            samples: 평가할 TestSample 객체들의 리스트
            per_sample_retrieved_image_ids: 각 샘플별로 검색된 이미지 ID를 담은 SampleImageIds 객체 리스트

        Returns:
            float: 평균 Visual Recall@K 점수
        """

        # Visual Recall@K 점수 계산
        avg_visual_recall = self.evaluate_visual_recall_at_k(
            samples, per_sample_retrieved_image_ids
        )

        # 상세 결과에 평균 Visual Recall@K 점수 추가
        return avg_visual_recall
