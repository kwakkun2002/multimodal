import numpy as np
import torch
import torch.nn.functional as F


class TextEmbeddingModal:
    @torch.inference_mode()
    def embed_text(self, texts, batch_size=64) -> np.ndarray:
        """
        BGE-M3 모델을 사용하여 텍스트들의 임베딩 생성
        """

        # 단일 문자열인 경우 리스트로 변환
        if isinstance(texts, str):
            texts = [texts]
        outs = []

        # 배치 단위로 처리
        for i in range(0, len(texts), batch_size):
            sub = texts[i : i + batch_size]
            # 토크나이저 적용
            inputs = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=True
            )
            # 메모리 고정
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
            # 디바이스 이동
            inputs = {
                k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
            }
            # 모델 적용
            feats = self.model.get_text_features(**inputs)
            # L2 정규화
            outs.append(F.normalize(feats, dim=-1).to("cpu", non_blocking=True))
        # 디바이스 동기화
        torch.cuda.synchronize()  # host로의 비동기 복사 완료 대기
        return torch.cat(outs, dim=0).numpy().astype(np.float32)
