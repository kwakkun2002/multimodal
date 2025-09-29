import io
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from minio import Minio  # MinIO 클라이언트 사용을 위한 임포트
from minio.error import (
    S3Error,  # presigned 및 객체 접근 시 오류 코드를 다루기 위한 예외 클래스 임포트
)
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

from repository.RepositoryConfig import VectorDatabaseConfig, WitImagesStorageConfig


# -------- 실제 서비스 클래스 --------
class MultiModalStore:
    def __init__(
        self,
        image_storage_config: WitImagesStorageConfig,
        vector_database_config: VectorDatabaseConfig,
    ):
        self.image_storage_config = image_storage_config
        self.vector_database_config = vector_database_config

        # MinIO
        self.image_storage = Minio(
            image_storage_config.endpoint,  # MinIO 서버 주소 (도커 컴포즈 내부면 "minio:9000")
            access_key=image_storage_config.access_key,  # MinIO 액세스 키
            secret_key=image_storage_config.secret_key,  # MinIO 시크릿 키
            secure=image_storage_config.secure,  # HTTP 사용 시 False, HTTPS 사용 시 True
        )

        # 버킷 존재 여부 확인 후 없으면 생성
        if not self.image_storage.bucket_exists(image_storage_config.bucket_name):
            self.image_storage.make_bucket(image_storage_config.bucket)

        # CLIP (transformers)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 768-d common projection
        self.model_id = "openai/clip-vit-large-patch14"

        # 모델 로드
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id, use_fast=True
        )

        # Milvus
        connections.connect(
            alias="default",
            host=vector_database_config.host,
            port=vector_database_config.port,
        )
        self.collection = self._get_or_create_collection()
        # 단일 벡터 필드
        self.vector_field = "vector"

        # 인덱스 생성
        self._ensure_index()

        # 컬렉션 로드
        self.collection.load()

    @torch.inference_mode()
    def embed_images(self, pil_images, batch_size=32) -> np.ndarray:
        """
        PIL 이미지를 임베딩하여 벡터 배열로 반환
        """

        # 단일 이미지인 경우 리스트로 변환
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]
        outs = []

        # 배치 단위로 처리
        for i in range(0, len(pil_images), batch_size):
            # 배치 데이터 준비
            sub = pil_images[i : i + batch_size]
            inputs = self.processor(images=sub, return_tensors="pt")
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
            # 디바이스 이동
            inputs = {
                k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
            }
            # 모델 적용
            feats = self.model.get_image_features(**inputs)
            # L2 정규화
            outs.append(F.normalize(feats, dim=-1).to("cpu", non_blocking=True))
        # 디바이스 동기화
        torch.cuda.synchronize()
        return torch.cat(outs, dim=0).numpy().astype(np.float32)

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
        self.image_storage.put_object(
            self.image_storage_config.bucket,
            object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type="image/jpeg",
        )
        return f"s3://{self.image_storage_config.bucket}/{object_name}"

    def presigned_url(self, s3_uri: str, expires_seconds: int = 3600) -> str:
        # s3://bucket/key -> presigned http url  # s3 스킴의 내부 경로를 사전서명된 HTTP URL로 변환
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid s3 uri: {s3_uri}")  # s3 접두어 누락 시 예외
        path = s3_uri[len("s3://") :]  # 접두어를 제거하여 'bucket/key'만 남김
        if "/" not in path:
            raise ValueError(
                f"Invalid s3 uri (missing key): {s3_uri}"
            )  # 키 구분자 누락 시 예외
        bucket, key = path.split("/", 1)  # 버킷과 키로 분리
        # 사전서명 전 실제 존재 여부를 확인하여 잘못된 키로 링크가 생성되지 않도록 방지
        try:
            self.image_storage.stat_object(bucket, key)  # 원본 키로 존재 확인
            final_key = key  # 존재하면 그대로 사용
        except S3Error as e:
            if e.code == "NoSuchKey" and not key.endswith(".jpg"):  # 확장자 누락 가능성
                alt_key = key + ".jpg"  # 대체 키 후보
                self.image_storage.stat_object(
                    bucket, alt_key
                )  # 대체 키 존재 확인(없으면 예외 전파)
                final_key = alt_key  # 대체 키로 확정
            else:
                raise  # 다른 오류는 상위로 전파
        return self.image_storage.presigned_get_object(  # 최종 키로 사전서명 URL 생성
            bucket, final_key, expires=timedelta(seconds=expires_seconds)
        )

    # Milvus
    def _get_or_create_collection(self) -> Collection:
        name = self.vector_database_config.collection_name
        if utility.has_collection(name):  # 컬렉션 존재 여부 확인
            coll = Collection(name)
            # 필수 필드 검증 (단일 벡터 스키마) -> vector, text, image_path
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
            existing = {idx.field_name for idx in (self.collection.indexes or [])}
        except Exception:
            existing = set()

        def ensure(field_name: str):
            if field_name not in existing:
                self.collection.create_index(
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
        self.collection.release()
        # 벡터 인덱스 재생성
        try:
            self.collection.drop_index(self.vector_field)
        except Exception:
            pass
        self.collection.create_index(
            field_name=self.vector_field,
            index_params={
                "index_type": index_type,
                "metric_type": metric_type or self.milvus_cfg.metric_type,
                "params": index_params,
            },
        )
        self.collection.load()

    # -------- High-level API (PIL만) --------
    def add_images_with_captions(
        self,
        images: Sequence[
            Image.Image
        ],  # PIL 이미지 객체들의 시퀀스 - MinIO에 업로드할 이미지들
        captions: Sequence[str],  # 각 이미지에 대응하는 캡션 문자열들의 시퀀스
        object_names: Optional[
            Sequence[Optional[str]]
        ] = None,  # MinIO에 저장될 객체 이름들 (None이면 자동 생성)
        batch_size: int = 32,  # 이미지 임베딩 시 한 번에 처리할 배치 크기
        do_flush: bool = True,  # Milvus 삽입 후 즉시 플러시 여부 (대량 삽입 시 False 권장)
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

        mr = self.collection.insert(
            data=[vectors_col, texts_col, paths_col],
            fields=[self.vector_field, "text", "image_path"],
        )

        if do_flush:
            # 너무 자주 호출하지 말고, 배치 or 전체 ingest 후 1회 호출 권장
            self.collection.flush()

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

        res = self.collection.search(
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
