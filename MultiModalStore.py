import io
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import torch
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
        self._ensure_index()  # 인덱스 생성
        self.coll.load()

    # -------- Embedding (transformers/CLIP) --------
    @torch.no_grad()
    def embed_text(self, texts) -> np.ndarray:
        """
        texts: str 또는 List[str]
        return: (N, 768) float32 normalized
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_images(self, pil_images) -> np.ndarray:
        """
        pil_images: PIL.Image.Image 또는 List[PIL.Image.Image]
        return: (N, 768) float32 normalized
        """
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().numpy().astype(np.float32)

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
            return Collection(name)

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
            fields=fields, description="CLIP(768d) vector + text + image_path"
        )
        return Collection(name=name, schema=schema)

    def _ensure_index(self):
        try:
            if self.coll.indexes:
                return
        except Exception:
            pass
        self.coll.create_index(
            field_name="vector",
            index_params={
                "index_type": self.milvus_cfg.index_type,
                "metric_type": self.milvus_cfg.metric_type,
                "params": self.milvus_cfg.index_params,
            },
        )

    def recreate_index(
        self,
        index_type: str,
        index_params: Dict[str, Any],
        metric_type: Optional[str] = None,
    ):
        self.coll.release()
        try:
            self.coll.drop_index("vector")
        except Exception:
            pass
        self.coll.create_index(
            field_name="vector",
            index_params={
                "index_type": index_type,
                "metric_type": metric_type or self.milvus_cfg.metric_type,
                "params": index_params,
            },
        )
        self.coll.load()

    # -------- High-level API (PIL만) --------
    def add_image_with_caption(
        self, image: Image.Image, caption: str, object_name: Optional[str] = None
    ) -> int:
        """
        1) PIL → MinIO 업로드(JPEG)
        2) PIL → 임베딩
        3) Milvus insert(vector, text, image_path)
        """
        s3_uri = self.upload_image(image, object_name=object_name)
        vec = self.embed_images(image)[0]  # (768,)

        mr = self.coll.insert(
            data=[[vec.tolist()], [caption], [s3_uri]],
            fields=["vector", "text", "image_path"],
        )
        self.coll.flush()
        return int(mr.primary_keys[0]) if mr.primary_keys else -1

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = self.embed_text(query)[0].tolist()
        # HNSW: ef, IVF: nprobe 등 파라미터는 필요 시 여기서 분기해서 넣어도 됨
        res = self.coll.search(
            data=[qvec],
            anns_field="vector",
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
