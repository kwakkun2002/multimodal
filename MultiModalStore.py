import io
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from minio import Minio
from PIL import Image
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

# -------- 설정 클래스 --------


@dataclass
class MinIOConfig:
    endpoint: str = "localhost:9000"  # docker-compose 내부면 "minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    bucket: str = "images"


@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "openclip_multimodal"
    dim: int = 768
    text_max_len: int = 2048
    path_max_len: int = 1024
    metric_type: str = "IP"
    index_type: str = "HNSW"
    index_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.index_params is None:
            self.index_params = {"M": 16, "efConstruction": 200}


# -------- 실제 서비스 클래스 --------
class MultiModalStore:
    def __init__(
        self,
        minio_cfg: MinIOConfig,
        milvus_cfg: MilvusConfig,
    ):
        self.minio_cfg = minio_cfg
        self.milvus_cfg = milvus_cfg

        # ---- MinIO ----
        self.minio = Minio(
            minio_cfg.endpoint,
            access_key=minio_cfg.access_key,
            secret_key=minio_cfg.secret_key,
            secure=minio_cfg.secure,
        )
        if not self.minio.bucket_exists(minio_cfg.bucket):  # 버킷 생성
            self.minio.make_bucket(minio_cfg.bucket)

        # ---- CLIP (transformers) ----
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "openai/clip-vit-large-patch14"  # 768-d common projection

        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id, use_fast=True
        )

        # ---- Milvus ----
        connections.connect(alias="default", host=milvus_cfg.host, port=milvus_cfg.port)
        self.coll = self._get_or_create_collection()
        # 단일 벡터 필드
        self.vector_field = "vector"

        self._ensure_index()  # 인덱스 생성
        self.coll.load()

    @torch.inference_mode()
    def embed_text(self, texts, batch_size=64) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        outs = []
        for i in range(0, len(texts), batch_size):
            sub = texts[i : i + batch_size]
            inputs = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
            inputs = {
                k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
            }
            feats = self.model.get_text_features(**inputs)
            outs.append(F.normalize(feats, dim=-1).to("cpu", non_blocking=True))
        torch.cuda.synchronize()  # host로의 비동기 복사 완료 대기
        return torch.cat(outs, dim=0).numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_images(self, pil_images, batch_size=32) -> np.ndarray:
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]
        outs = []
        for i in range(0, len(pil_images), batch_size):
            sub = pil_images[i : i + batch_size]
            inputs = self.processor(images=sub, return_tensors="pt")
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
            inputs = {
                k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
            }
            feats = self.model.get_image_features(**inputs)
            outs.append(F.normalize(feats, dim=-1).to("cpu", non_blocking=True))
        torch.cuda.synchronize()
        return torch.cat(outs, dim=0).numpy().astype(np.float32)

    # -------- MinIO (PIL만) --------
    def upload_image(
        self,
        image: Image.Image,
        object_name: Optional[str] = None,
        jpeg_quality: int = 95,
    ) -> str:
        """
        PIL 이미지를 JPEG로 인코딩하여 MinIO에 업로드.
        return: 's3://bucket/object_name'
        """
        if object_name is None:
            object_name = "img_" + os.urandom(6).hex() + ".jpg"

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
        data = buf.getvalue()
        self.minio.put_object(
            self.minio_cfg.bucket,
            object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type="image/jpeg",
        )
        return f"s3://{self.minio_cfg.bucket}/{object_name}"

    def presigned_url(self, s3_uri: str, expires_seconds: int = 3600) -> str:
        # s3://bucket/key -> presigned http url
        _, bucket, key = s3_uri.split("/", 2)
        return self.minio.presigned_get_object(
            bucket, key, expires=timedelta(seconds=expires_seconds)
        )

    # -------- Milvus --------
    def _get_or_create_collection(self) -> Collection:
        name = self.milvus_cfg.collection_name
        if utility.has_collection(name):  # 컬렉션 존재 여부 확인
            coll = Collection(name)
            # 필수 필드 검증 (단일 벡터 스키마)
            field_names = {f.name for f in coll.schema.fields}
            required = {"vector", "text", "image_path"}
            missing = required - field_names
            if missing:
                raise RuntimeError(
                    f"Existing collection '{name}' is missing required fields: {sorted(missing)}"
                )
            return coll

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.milvus_cfg.dim
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=self.milvus_cfg.text_max_len,
            ),
            FieldSchema(
                name="image_path",
                dtype=DataType.VARCHAR,
                max_length=self.milvus_cfg.path_max_len,
            ),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="CLIP(768d) vector + text + image_path",
        )
        return Collection(name=name, schema=schema)

    def _ensure_index(self):
        # 필드별 인덱스 존재 여부 확인 후 생성
        try:
            existing = {idx.field_name for idx in (self.coll.indexes or [])}
        except Exception:
            existing = set()

        def ensure(field_name: str):
            if field_name not in existing:
                self.coll.create_index(
                    field_name=field_name,
                    index_params={
                        "index_type": self.milvus_cfg.index_type,
                        "metric_type": self.milvus_cfg.metric_type,
                        "params": self.milvus_cfg.index_params,
                    },
                )

        ensure(self.vector_field)

    def recreate_index(
        self,
        index_type: str,
        index_params: Dict[str, Any],
        metric_type: Optional[str] = None,
    ):
        self.coll.release()
        # 벡터 인덱스 재생성
        try:
            self.coll.drop_index(self.vector_field)
        except Exception:
            pass
        self.coll.create_index(
            field_name=self.vector_field,
            index_params={
                "index_type": index_type,
                "metric_type": metric_type or self.milvus_cfg.metric_type,
                "params": index_params,
            },
        )
        self.coll.load()

    # -------- High-level API (PIL만) --------
    def add_images_with_captions(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
        object_names: Optional[Sequence[Optional[str]]] = None,
        batch_size: int = 32,
        do_flush: bool = True,
    ) -> List[int]:
        """
        1) 각 이미지 MinIO 업로드(JPEG) -> s3_uri 리스트
        2) 이미지 배치 임베딩 -> (N, D)
        3) Milvus insert(vector, text, image_path)  [컬럼 기반: 2D, 1D, 1D]
        4) 선택적으로 flush (대량 삽입 뒤 1회 권장)
        return: primary key 리스트
        """
        if len(images) != len(captions):
            raise ValueError(
                f"images({len(images)}) and captions({len(captions)}) must have the same length"
            )

        N = len(images)
        if object_names is not None and len(object_names) != N:
            raise ValueError("object_names length must match images length if provided")

        # 1) 업로드 (필요 시 ThreadPool로 병렬화 가능)
        s3_uris: List[str] = []
        for idx, img in enumerate(images):
            oname = None if object_names is None else object_names[idx]
            s3_uris.append(self.upload_image(img, object_name=oname))

        # 2) 배치 임베딩 (이미지)
        img_vecs = self.embed_images(images, batch_size=batch_size)  # (N, D)

        # 3) Milvus insert (컬럼 기반)
        vectors_col = [vec.tolist() for vec in img_vecs]
        texts_col = list(captions)
        paths_col = s3_uris

        mr = self.coll.insert(
            data=[vectors_col, texts_col, paths_col],
            fields=[self.vector_field, "text", "image_path"],
        )

        if do_flush:
            # 너무 자주 호출하지 말고, 배치 or 전체 ingest 후 1회 호출 권장
            self.coll.flush()

        # 4) PK 반환 (auto_id인 경우 MutatationResult.primary_keys에서 수집)
        pks = []
        if getattr(mr, "primary_keys", None) is not None:
            for pk in mr.primary_keys:
                try:
                    pks.append(int(pk))
                except (TypeError, ValueError):
                    # 문자열 PK 등인 경우 그대로 유지
                    pks.append(pk)
        return pks

    def search_by_text(
        self, query: str, top_k: int = 5, search_target: str = "images"
    ) -> List[Dict[str, Any]]:
        """
        텍스트 임베딩으로 단일 벡터 필드("vector")에서 검색합니다.
        """
        qvec = self.embed_text(query)[0].tolist()
        anns_field = self.vector_field

        res = self.coll.search(
            data=[qvec],
            anns_field=anns_field,
            param={"ef": 128} if self.milvus_cfg.index_type.upper() == "HNSW" else {},
            limit=top_k,
            output_fields=["text", "image_path"],
            consistency_level="Strong",
        )
        hits = res[0]
        return [
            {
                "id": int(h.id),
                "distance": float(h.distance),  # IP: 클수록 유사
                "text": h.entity.get("text"),
                "image_path": h.entity.get("image_path"),
            }
            for h in hits
        ]
