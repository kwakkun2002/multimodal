import os
from dataclasses import asdict
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRecall

from mramg_proj.doc_vector_store import DocVectorStore
from mramg_proj.doc_vector_store_config import DocVectorStoreConfig
from mramg_proj.models.milvus_search_result import MilvusSearchResult
from ragas_eval.models.context_recall_score_result import ContextRecallScoreResult
from ragas_eval.models.mqa import Mqa
from ragas_eval.models.ragas_evaluation_data import RagasEvaluationData
from ragas_eval.models.retrieved_contexts_result import RetrievedContextsResult
from ragas_eval.models.retrieved_image_ids import RetrievedImageIds
from ragas_eval.utiles.load_test_data import load_mqa


class ContextRecallEvaluator:
    def __init__(self):
        self._config: DocVectorStoreConfig = DocVectorStoreConfig(
            collection_name="doc_wit_documents"
        )
        self._store: DocVectorStore = DocVectorStore(self._config)
        self._store.create_index()
        load_dotenv()

    def _build_evaluator_llm(
        self, model_name: str = "gpt-4o-mini"
    ) -> LangchainLLMWrapper:
        openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경 변수가 설정되어야 합니다 (RAGAS LLM 평가용)."
            )
        langchain_llm: ChatOpenAI = ChatOpenAI(model=model_name)
        evaluator_llm: LangchainLLMWrapper = LangchainLLMWrapper(langchain_llm)
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
        mqa_list: List[Mqa] = load_mqa(jsonl_path, limit=limit)

        # RAGAS 평가용 데이터셋을 저장할 리스트
        ragas_dataset: List[RagasEvaluationData] = []
        # 각 샘플별로 검색된 이미지 ID를 저장할 리스트
        all_image_ids: List[RetrievedImageIds] = []

        # 각 샘플에 대해 벡터 검색 수행
        for mqa in mqa_list:
            # 질문으로 벡터 스토어 검색
            search_results: List[MilvusSearchResult] = self._search_contexts(
                mqa.question, top_k
            )
            # 검색 결과에서 텍스트 컨텍스트 추출
            contexts: list[str] = self._extract_contexts(search_results)
            # 검색 결과에서 이미지 ID 추출 및 중복 제거
            image_ids: list[str] = self._extract_image_ids(search_results)

            # RAGAS 평가 형식으로 데이터 객체 생성
            ragas_dataset.append(
                RagasEvaluationData(
                    user_input=mqa.question,
                    retrieved_contexts=contexts,
                    reference=mqa.ground_truth,
                )
            )
            # 현재 샘플의 검색된 이미지 ID를 객체로 저장
            all_image_ids.append(RetrievedImageIds(image_ids=image_ids))

        # 검색 결과를 데이터클래스로 반환
        return RetrievedContextsResult(
            original_test_samples=mqa_list,
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

    def _search_contexts(self, query: str, top_k: int) -> List[MilvusSearchResult]:
        """Search the vector store for relevant contexts."""
        return self._store.search(query, top_k=top_k)

    def _extract_contexts(self, search_results: List[MilvusSearchResult]) -> list[str]:
        """
        검색 결과에서 텍스트 컨텍스트들을 추출합니다.

        Args:
            search_results: Milvus 검색 결과 리스트

        Returns:
            list[str]: 추출된 텍스트 컨텍스트 리스트
        """
        # 각 검색 결과에서 텍스트만 추출하여 리스트로 반환
        return [result.text for result in search_results]

    def _extract_image_ids(self, search_results: List[MilvusSearchResult]) -> list[str]:
        """
        검색 결과에서 이미지 ID들을 추출하고 중복을 제거합니다.

        Args:
            search_results: Milvus 검색 결과 리스트

        Returns:
            list[str]: 중복이 제거된 이미지 ID 리스트 (순서 유지)
        """
        # 모든 검색 결과에서 이미지 ID를 수집할 리스트
        image_ids: list[str] = []

        # 각 검색 결과의 이미지 ID를 리스트에 추가
        for result in search_results:
            image_ids.extend(iid for iid in result.image_ids)

        # 순서를 유지하면서 중복 제거 (dict.fromkeys는 삽입 순서 유지)
        deduplicated_image_ids: list[str] = list(dict.fromkeys(image_ids))

        return deduplicated_image_ids

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
