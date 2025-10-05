import os
import json

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRecall
from dotenv import load_dotenv
from mramg_proj.doc_wit_config import DocWitConfig
from mramg_proj.doc_wit_vector_store import DocWitVectorStore


def load_test_data(
    jsonl_path: str
):
    """
    JSONL 테스트 샘플 로드 함수(테스트 데이터 읽기)
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append(
                {
                    "id": data.get("id"),
                    "question": data.get("question"),
                    "ground_truth": data.get("ground_truth"),
                    "images_list": data.get("images_list", []),
                }
            )
    return samples


class ContextRecallEvaluator:
    def __init__(self):
        self._config = DocWitConfig()
        self._store = DocWitVectorStore(self._config)
        self._store.create_index()
        load_dotenv()

    def _build_evaluator_llm(self, model_name: str = "gpt-4o"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경 변수가 설정되어야 합니다 (RAGAS LLM 평가용)."
            )
        langchain_llm = ChatOpenAI(model=model_name)
        evaluator_llm = LangchainLLMWrapper(langchain_llm)
        return evaluator_llm

    def evaluate_context_recall_at_k(self, jsonl_path: str, top_k: int = 10):
        """Evaluate context recall@k for a test dataset."""
        test_dataset = load_test_data(jsonl_path)

        ragas_dataset = []
        all_image_ids = []

        for sample in test_dataset:
            search_results = self._search_contexts(sample["question"], top_k)
            contexts, image_ids = self._extract_contexts_and_images(search_results)

            ragas_dataset.append({
                "user_input": sample["question"],
                "retrieved_contexts": contexts,
                "reference": sample["ground_truth"],
            })
            all_image_ids.append(image_ids)

        avg_recall = self._compute_average_recall(ragas_dataset)
        details = {
            "average_context_recall@k": avg_recall,
            "k": top_k,
            "num_samples": len(ragas_dataset),
        }

        return avg_recall, details, ragas_dataset, all_image_ids


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


    def _compute_average_recall(self, ragas_dataset):
        """Run RAGAS evaluation and compute average recall."""
        evaluation_dataset = EvaluationDataset.from_list(ragas_dataset)
        evaluator_llm = self._build_evaluator_llm()
        metric = ContextRecall(llm=evaluator_llm)
        results = evaluate(dataset=evaluation_dataset, metrics=[metric])
        df = results.to_pandas()

        if "context_recall" in df.columns:
            return float(df["context_recall"].mean())
        return float(results["context_recall"])


