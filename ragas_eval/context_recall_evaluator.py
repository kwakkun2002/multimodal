import json
import os
from dataclasses import asdict, dataclass
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRecall
from ragas_eval.test_sample import TestSample
from ragas_eval.ragas_evaluation_data import RagasEvaluationData
from ragas_eval.sample_image_ids import SampleImageIds
from ragas_eval.retrieved_contexts_result import RetrievedContextsResult
from mramg_proj.doc_vector_store_config import DocVectorStoreConfig
from mramg_proj.doc_vector_store import DocVectorStore
from ragas_eval.context_recall_score_result import ContextRecallScoreResult

def load_test_data(jsonl_path: str, limit: int = None) -> List[TestSample]:
    """
    JSONL 테스트 샘플 로드 함수(테스트 데이터 읽기)

    Args:
        jsonl_path: JSONL 파일 경로
        limit: 로드할 최대 샘플 개수

    Returns:
        List[TestSample]: 로드된 테스트 샘플 리스트
    """
    # 테스트 샘플들을 저장할 리스트
    samples = []

    # JSONL 파일을 한 줄씩 읽기
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            # JSON 파싱
            data = json.loads(line)
            # TestSample 객체로 변환하여 추가
            samples.append(
                TestSample(
                    id=data.get("id", ""),
                    question=data.get("question", ""),
                    ground_truth=data.get("ground_truth", ""),
                    images_list=data.get("images_list", []),
                )
            )

    print("[load_test_data] 로드된 샘플 개수:", len(samples))
    print("[load_test_data] 로드된 샘플 첫 번째 샘플:")
    print(samples[0])

    # 샘플 개수 제한이 있으면 적용
    if limit:
        samples = samples[:limit]
    return samples


class ContextRecallEvaluator:
    def __init__(self):
        self._config = DocVectorStoreConfig()
        self._store = DocVectorStore(self._config)
        self._store.create_index()
        load_dotenv()

    def _build_evaluator_llm(self, model_name: str = "gpt-4o-mini"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경 변수가 설정되어야 합니다 (RAGAS LLM 평가용)."
            )
        langchain_llm = ChatOpenAI(model=model_name)
        evaluator_llm = LangchainLLMWrapper(langchain_llm)
        return evaluator_llm

    def retrieve_contexts_for_samples(
        self, jsonl_path: str, top_k: int = 10, limit: int = None
    ) -> RetrievedContextsResult:
        """
        테스트 샘플들에 대해 컨텍스트를 검색하고 데이터를 준비합니다.

        Args:
            jsonl_path: 테스트 데이터 JSONL 파일 경로
            top_k: 검색할 상위 k개의 결과
            limit: 처리할 샘플 개수 제한

        Returns:
            RetrievedContextsResult: 원본 샘플, RAGAS 평가 데이터, 검색된 이미지 ID를 담은 결과 객체
        """
        # JSONL 파일에서 테스트 데이터 로드 (TestSample 객체 리스트)
        test_samples = load_test_data(jsonl_path, limit=limit)

        # RAGAS 평가용 데이터셋을 저장할 리스트
        ragas_dataset: List[RagasEvaluationData] = []
        # 각 샘플별로 검색된 이미지 ID를 저장할 리스트
        all_image_ids: List[SampleImageIds] = []

        # 각 샘플에 대해 벡터 검색 수행
        for sample in test_samples:
            # 질문으로 벡터 스토어 검색
            search_results = self._search_contexts(sample.question, top_k)
            # 검색 결과에서 텍스트 컨텍스트와 이미지 ID 추출
            contexts, image_ids = self._extract_contexts_and_images(search_results)

            # RAGAS 평가 형식으로 데이터 객체 생성
            ragas_dataset.append(
                RagasEvaluationData(
                    user_input=sample.question,
                    retrieved_contexts=contexts,
                    reference=sample.ground_truth,
                )
            )
            # 현재 샘플의 검색된 이미지 ID를 객체로 저장
            all_image_ids.append(SampleImageIds(image_ids=image_ids))

        # 검색 결과를 데이터클래스로 반환
        return RetrievedContextsResult(
            original_test_samples=test_samples,
            ragas_evaluation_data=ragas_dataset,
            retrieved_image_ids_per_sample=all_image_ids,
        )

    def compute_context_recall_score(
        self, ragas_dataset: List[RagasEvaluationData], top_k: int = 10
    ):
        """
        RAGAS 데이터셋을 사용하여 Context Recall 점수를 계산합니다.

        Args:
            ragas_dataset: RAGAS 평가용 데이터셋 (RagasEvaluationData 객체 리스트)
            top_k: 검색한 상위 k개의 결과 (메타데이터용)

        Returns:
            ContextRecallScoreResult: 평균 recall 점수와 평가 상세 정보를 담은 결과 객체
        """
        # RAGAS를 사용하여 평균 Context Recall 점수 계산
        avg_recall = self._compute_average_recall(ragas_dataset)

        # 평가 결과를 데이터클래스로 반환
        return ContextRecallScoreResult(
            average_context_recall_at_k=avg_recall,
            k=top_k,
            num_samples=len(ragas_dataset),
        )

    def _search_contexts(self, query: str, top_k: int):
        """Search the vector store for relevant contexts."""
        return self._store.search(query, top_k=top_k)

    def _extract_contexts_and_images(self, search_results):
        """Extract texts and image IDs from search results."""
        contexts = [r.get("text", "") for r in search_results]

        image_ids = []
        for result in search_results:
            image_ids.extend(str(iid) for iid in (result.get("image_ids") or []))

        # Deduplicate while preserving order
        image_ids = list(dict.fromkeys(image_ids))
        return contexts, image_ids

    def _compute_average_recall(
        self, ragas_dataset: List[RagasEvaluationData]
    ) -> float:
        """
        RAGAS 평가를 실행하고 평균 recall을 계산합니다.

        Args:
            ragas_dataset: RagasEvaluationData 객체 리스트

        Returns:
            float: 평균 Context Recall 점수
        """
        # RagasEvaluationData 객체들을 RAGAS 라이브러리가 기대하는 딕셔너리 형식으로 변환
        ragas_dict_list = [asdict(data) for data in ragas_dataset]

        # RAGAS 평가 데이터셋 생성
        evaluation_dataset = EvaluationDataset.from_list(ragas_dict_list)
        # LLM 평가기 생성
        evaluator_llm = self._build_evaluator_llm()
        # Context Recall 메트릭 설정
        metric = ContextRecall(llm=evaluator_llm)
        # 평가 실행
        results = evaluate(dataset=evaluation_dataset, metrics=[metric])
        # 결과를 DataFrame으로 변환
        df = results.to_pandas()

        # Context Recall 점수 추출
        if "context_recall" in df.columns:
            return float(df["context_recall"].mean())
        return float(results["context_recall"])
