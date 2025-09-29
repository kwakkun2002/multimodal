"""
Context Recall 평가 로직을 클래스로 캡슐화
"""

# 타입 힌트를 위한 typing 임포트
# 환경 변수 접근을 위한 os 임포트
import os
from typing import Any, Dict, List, Tuple

# LLM 제공을 위한 LangChain OpenAI 임포트
from langchain_openai import ChatOpenAI

# RAGAS 관련 임포트
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRecall

# 내부 유틸 및 저장소 임포트
from mramg_proj.DocWitConfig import DocWitConfig
from mramg_proj.DocWitVectorStore import DocWitVectorStore
from ragas_eval.dataset_utils import load_test_samples


class ContextRecallEvaluator:
    """
    Context Recall@K 평가를 수행하는 클래스
    """

    # 생성자: 구성과 저장소 초기화
    def __init__(self) -> None:
        # DocWit 설정 불러오기
        self._config = DocWitConfig()
        # 벡터 저장소 인스턴스 생성
        self._store = DocWitVectorStore(self._config)
        # 인덱스 생성/로드(검색 성능 및 로드 보장)
        self._store.create_index()

    # 내부 메서드: 평가에 사용할 LLM 준비
    def _build_evaluator_llm(self, model_name: str | None = None):
        """
        평가에 사용할 LLM 래퍼 생성 함수
        """
        # 사용할 모델명 결정(환경변수 기본값)
        model = model_name or os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini")
        # OpenAI API 키 로드
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        # 키가 없으면 실행 불가이므로 오류 발생
        if not openai_api_key:
            # 사용자 안내 에러
            raise RuntimeError(
                "OPENAI_API_KEY 환경 변수가 설정되어야 합니다 (RAGAS LLM 평가용)."
            )
        # LangChain의 OpenAI LLM 인스턴스 생성
        langchain_llm = ChatOpenAI(model=model)
        # RAGAS에서 사용할 수 있도록 래핑
        evaluator_llm = LangchainLLMWrapper(langchain_llm)
        # LLM 래퍼 반환
        return evaluator_llm

    # 퍼블릭 메서드: Context Recall@K 평가 수행
    def evaluate_context_recall_at_k(
        self,
        jsonl_path: str,
        top_k: int = 10,
        sample_limit: int | None = None,
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]], List[List[str]]]:
        """
        Context Recall@K 평가를 수행
        """
        # 테스트 샘플 로드
        samples = load_test_samples(jsonl_path, limit=sample_limit)
        # RAGAS 입력 포맷 구성
        ragas_dataset: List[Dict[str, Any]] = []
        # 이미지 ID 누적(시각 리콜용)
        per_sample_retrieved_image_ids: List[List[str]] = []
        # 각 샘플에 대해 반복
        for sample in samples:
            # 검색 실행(텍스트+이미지ID 포함)
            results = self._store.search(sample["question"], top_k=top_k)
            # 컨텍스트 텍스트만 추출
            retrieved_contexts: List[str] = [r.get("text", "") for r in results]
            # 이미지 ID 수집(순서 보존 중복 제거)
            img_ids: List[str] = []
            for result in results:
                for iid in result.get("image_ids", []) or []:
                    img_ids.append(str(iid))
            per_sample_retrieved_image_ids.append(list(dict.fromkeys(img_ids)))
            # RAGAS 샘플 누적
            ragas_dataset.append(
                {
                    "user_input": sample["question"],
                    "retrieved_contexts": retrieved_contexts,
                    "reference": sample["ground_truth"],
                }
            )

        # 리스트를 EvaluationDataset으로 변환
        evaluation_dataset = EvaluationDataset.from_list(ragas_dataset)
        # 평가용 LLM 준비
        evaluator_llm = self._build_evaluator_llm()
        # Context Recall 메트릭 인스턴스 생성
        metric = ContextRecall(llm=evaluator_llm)
        # RAGAS 평가 실행
        results = evaluate(dataset=evaluation_dataset, metrics=[metric])
        # 판다스 데이터프레임으로 변환
        df = results.to_pandas()
        # 평균 점수 계산
        avg_recall = (
            float(df["context_recall"].mean())
            if "context_recall" in df.columns
            else float(results["context_recall"])
        )
        # 상세 결과 구성(평균 Visual Recall은 호출자에서 계산)
        details: Dict[str, Any] = {
            "average_context_recall@k": avg_recall,
            "k": top_k,
            "num_samples": len(ragas_dataset),
            # 참고: 평균 Visual Recall은 엔트리에서 별도 계산 가능
        }
        # 평균과 상세 결과, 그리고 시각 리콜 계산을 위한 보조 데이터 반환
        return avg_recall, details, samples, per_sample_retrieved_image_ids
