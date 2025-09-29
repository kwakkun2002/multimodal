"""
검색 유틸리티: 텍스트 컨텍스트 생성 함수만을 포함
"""

# 타입 힌트를 위한 typing 임포트
from typing import List

# DocWitVectorStore 타입 임포트를 위한 경로
from mramg_proj.DocWitVectorStore import DocWitVectorStore


def build_retrieved_contexts(
    store: DocWitVectorStore, question: str, top_k: int
) -> List[str]:
    """
    검색 결과에서 컨텍스트 리스트 생성 함수
    """
    # 벡터 검색 실행(top-k 결과 획득)
    results = store.search(question, top_k=top_k)
    # 각 결과의 텍스트만 추출하여 리스트화
    contexts: List[str] = [r.get("text", "") for r in results]
    # 컨텍스트 텍스트 리스트 반환
    return contexts
