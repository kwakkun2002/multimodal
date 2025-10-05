from typing import List

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


class BgeM3Embedder:
    """
    BGE-M3 임베더 클래스
    Args:
        device: 디바이스
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BgeM3Embedder] {self.device} 에서 모델 로드 중...")
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=self.device)
        print("[BgeM3Embedder] 로드 완료!")

    @torch.inference_mode()
    def embed(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, batch_size=32, max_length=512)

        # dict 형태일 경우 dense_vecs만 추출
        dense = (
            np.array(embeddings["dense_vecs"], dtype=np.float32)
            if isinstance(embeddings, dict)
            else np.array(embeddings, dtype=np.float32)
        )

        # L2 정규화
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = dense / norms
        return normalized
