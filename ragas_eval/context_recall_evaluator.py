import json
import os
from dataclasses import asdict, dataclass
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRecall

from mramg_proj.doc_wit_config import DocWitConfig
from mramg_proj.doc_wit_vector_store import DocWitVectorStore


# 테스트 샘플 데이터를 담는 소형 데이터클래스
@dataclass
class TestSample:
    """
    JSONL에서 로드한 단일 테스트 샘플을 담는 클래스

    Attributes:
        id: 샘플 고유 식별자
        question: 사용자 질문
        ground_truth: 정답 (참조 답변)
        images_list: 정답에 해당하는 이미지 ID 리스트
    """

    # 샘플의 고유 ID
    id: str
    # 사용자가 입력한 질문
    question: str
    # 모델이 생성해야 할 정답
    ground_truth: str
    # 정답과 관련된 이미지 ID들
    images_list: List[str]


# RAGAS 평가용 데이터를 담는 소형 데이터클래스
@dataclass
class RagasEvaluationData:
    """
    RAGAS 평가를 위한 단일 데이터 항목을 담는 클래스

    Attributes:
        user_input: 사용자 질문
        retrieved_contexts: 검색된 텍스트 컨텍스트 리스트
        reference: 정답 (참조 답변)
    """

    # 평가할 사용자 질문
    user_input: str
    # 벡터 스토어에서 검색된 컨텍스트들
    retrieved_contexts: List[str]
    # 정답 참조 텍스트
    reference: str


# 샘플별 검색된 이미지 ID를 담는 소형 데이터클래스
@dataclass
class SampleImageIds:
    """
    단일 샘플에 대해 검색된 이미지 ID들을 담는 클래스

    Attributes:
        image_ids: 검색된 이미지 ID 리스트
    """

    # 벡터 검색 결과로 얻은 이미지 ID들
    image_ids: List[str]


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


# Context Recall 평가 점수 결과를 담는 데이터클래스
@dataclass
class ContextRecallScoreResult:
    """
    Context Recall 평가 점수 결과를 담는 클래스

    Attributes:
        average_context_recall_at_k: 평균 Context Recall 점수 (0.0 ~ 1.0)
        k: 검색에 사용된 상위 k개의 결과 수
        num_samples: 평가에 사용된 샘플 개수
    """

    # 모든 샘플에 대한 평균 Context Recall 점수
    average_context_recall_at_k: float
    # 검색 시 사용한 top-k 값
    k: int
    # 평가에 포함된 샘플의 총 개수
    num_samples: int



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
        self._config = DocWitConfig()
        self._store = DocWitVectorStore(self._config)
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
    ) -> ContextRecallScoreResult:
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
