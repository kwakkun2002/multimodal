import json
from typing import Any, Dict, List

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from mramg_proj.models.milvus_search_result import MilvusSearchResult


class MilvusManager:
    """
    Milvus 매니저 클래스
    Args:
        host: Milvus 서버 호스트
        port: Milvus 서버 포트
        collection_name: 컬렉션 이름
        embedding_dim: 임베딩 차원
        text_max_len: 텍스트 최대 길이
        image_ids_max_len: 이미지 ID 최대 길이
    """

    def __init__(
        self,
        host: str,
        port: str,
        collection_name: str,
        embedding_dim: int,
        text_max_len: int = 2000,
        image_ids_max_len: int = 1000,
    ):
        connections.connect(alias="default", host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.text_max_len = text_max_len
        self.image_ids_max_len = image_ids_max_len
        self.collection = self._get_or_create_collection()
        print(f"[MilvusManager] 컬렉션 '{collection_name}' 준비 완료")

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            print(f"[MilvusManager] 기존 컬렉션 '{self.collection_name}' 로드")
            return Collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="original_id", dtype=DataType.INT64),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR, max_length=self.text_max_len
            ),
            FieldSchema(
                name="image_ids",
                dtype=DataType.VARCHAR,
                max_length=self.image_ids_max_len,
            ),
            FieldSchema(
                name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim
            ),
        ]
        schema = CollectionSchema(fields=fields, description="")
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"[MilvusManager] 새 컬렉션 '{self.collection_name}' 생성")
        return collection

    def insert(self, docs: List[Dict[str, Any]], embeddings: np.ndarray):
        if not docs:
            return []
        texts = [d["text"] for d in docs]
        original_ids = [int(d["doc_id"]) for d in docs]
        image_ids_strs = [
            json.dumps(d["image_ids"]) if d["image_ids"] else "[]" for d in docs
        ]

        insert_data = [original_ids, texts, image_ids_strs, embeddings.tolist()]
        mr = self.collection.insert(insert_data)
        self.collection.flush()
        return [int(pk) for pk in mr.primary_keys]

    def create_index(self):
        if not self.collection.indexes:
            self.collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "IP",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            print("[MilvusManager] 인덱스 생성 완료")
        self.collection.load()

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[MilvusSearchResult]:
        """
        벡터 쿼리를 통해 유사한 문서를 검색

        Args:
            query_vec: 검색할 쿼리 벡터
            top_k: 반환할 최대 결과 개수

        Returns:
            검색된 결과 리스트 (MilvusSearchResult 객체들)
        """
        # Milvus 컬렉션에서 벡터 검색 수행
        results = self.collection.search(
            data=[query_vec.tolist()],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 128}},
            limit=top_k,
            output_fields=["text", "image_ids", "original_id"],
        )

        # 검색 결과를 MilvusSearchResult 객체로 변환
        formatted = []
        for hit in results[0]:
            # JSON 문자열로 저장된 이미지 ID 리스트를 파싱
            try:
                image_ids = json.loads(hit.entity.get("image_ids", "[]"))
            except Exception:
                image_ids = []

            # MilvusSearchResult 객체 생성 후 리스트에 추가
            formatted.append(
                MilvusSearchResult(
                    id=int(hit.id),
                    original_id=hit.entity.get("original_id", 0),
                    text=hit.entity.get("text", ""),
                    image_ids=image_ids,
                    score=float(hit.score),
                )
            )
        return formatted
