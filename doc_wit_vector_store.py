import json  # JSON 파일 처리를 위한 json 모듈 임포트
from dataclasses import dataclass  # 데이터 클래스 정의를 위한 dataclass 임포트
from typing import Any, Dict, List  # 타입 힌트를 위한 타이핑 모듈 임포트

import numpy as np  # 수치 연산을 위한 numpy 모듈 임포트
import torch  # 딥러닝을 위한 PyTorch 임포트
from FlagEmbedding import BGEM3FlagModel  # BGE-M3 임베딩 모델 임포트
from llama_index.core.node_parser import (
    SentenceSplitter,  # LlamaIndex 문장 분할기 임포트
)
from pymilvus import (  # Milvus 관련 클래스들 임포트
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from tqdm import tqdm  # 진행률 표시를 위한 tqdm 임포트

# -------- 설정 클래스 --------


@dataclass
class DocWitConfig:  # DocWit 벡터 저장소 설정을 담는 데이터 클래스
    """DocWit 벡터 저장소의 연결 및 컬렉션 설정을 담는 클래스"""

    milvus_host: str = "localhost"  # Milvus 서버 호스트 주소
    milvus_port: int = 19532  # Milvus 서버 포트 (도커 포트 매핑 19530->19532)
    collection_name: str = "doc_wit_documents"  # 사용할 컬렉션 이름
    embedding_dim: int = 1024  # BGE-M3 임베딩 차원
    chunk_size: int = 256  # 텍스트 청크 크기 (토큰 단위)
    chunk_overlap: int = 20  # 청크 간 중복 크기
    text_max_len: int = 8192  # 텍스트 최대 길이
    image_ids_max_len: int = 2048  # 이미지 ID 리스트 최대 길이


class DocWitVectorStore:  # DocWit JSONL 파일을 벡터DB에 저장하는 메인 클래스
    """DocWit JSONL 파일의 문서들을 BGE-M3 임베딩으로 벡터화하여 Milvus에 저장"""

    def __init__(self, config: DocWitConfig):  # 설정으로 초기화하는 생성자
        """DocWitVectorStore를 설정에 따라 초기화"""
        self.config = config  # 설정 저장

        # ---- BGE-M3 모델 ----
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # GPU 사용 가능 여부 확인
        print(f"BGE-M3 모델을 {self.device}에서 로드 중...")  # 모델 로드 시작 메시지
        self.embedding_model = BGEM3FlagModel(  # BGE-M3 모델 로드
            "BAAI/bge-m3",
            use_fp16=True,  # FP16 사용으로 메모리 절약
            device=self.device,  # 지정된 디바이스 사용
        )
        print("BGE-M3 모델 로드 완료!")  # 모델 로드 완료 메시지

        # ---- LlamaIndex SentenceSplitter ----
        self.splitter = SentenceSplitter(  # 문장 분할기 초기화
            chunk_size=config.chunk_size,  # 청크 크기 설정
            chunk_overlap=config.chunk_overlap,  # 청크 중복 설정
        )

        # ---- Milvus 연결 ----
        connections.connect(  # Milvus 서버 연결
            alias="default", host=config.milvus_host, port=config.milvus_port
        )
        self.collection = self._get_or_create_collection()  # 컬렉션 생성 또는 로드
        print(
            f"컬렉션 '{config.collection_name}' 준비 완료!"
        )  # 컬렉션 준비 완료 메시지

    def _get_or_create_collection(self) -> Collection:  # 컬렉션 생성 또는 로드 메서드
        """컬렉션을 생성하거나 기존 컬렉션을 반환"""
        name = self.config.collection_name  # 컬렉션 이름

        if utility.has_collection(name):  # 컬렉션 존재 여부 확인
            print(f"기존 컬렉션 '{name}'을(를) 로드합니다.")  # 기존 컬렉션 로드 메시지
            coll = Collection(name)  # 기존 컬렉션 로드
            # 필수 필드 검증
            field_names = {f.name for f in coll.schema.fields}  # 기존 필드 이름들
            required = {"id", "text", "image_ids", "vector"}  # 필요한 필드들
            missing = required - field_names  # 누락된 필드들
            if missing:  # 누락된 필드가 있는 경우
                raise RuntimeError(  # 런타임 에러 발생
                    f"기존 컬렉션 '{name}'에 필요한 필드가 없습니다: {sorted(missing)}"
                )
            return coll  # 기존 컬렉션 반환

        # 새 컬렉션 스키마 정의
        fields = [  # 필드 스키마 정의
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
            ),  # 기본키 필드 (자동 생성)
            FieldSchema(  # 원본 문서 ID 저장 필드
                name="original_id",
                dtype=DataType.INT64,
            ),
            FieldSchema(  # 텍스트 필드
                name="text",
                dtype=DataType.VARCHAR,
                max_length=self.config.text_max_len,
            ),
            FieldSchema(  # 이미지 ID 리스트 필드
                name="image_ids",
                dtype=DataType.VARCHAR,
                max_length=self.config.image_ids_max_len,
            ),
            FieldSchema(  # 벡터 필드
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.embedding_dim,
            ),
        ]
        schema = CollectionSchema(  # 컬렉션 스키마 생성
            fields=fields,
            description="DocWit 문서들의 BGE-M3 임베딩 저장소",  # 설명
        )
        collection = Collection(name=name, schema=schema)  # 컬렉션 생성
        print(f"새 컬렉션 '{name}'이(가) 생성되었습니다.")  # 새 컬렉션 생성 메시지
        return collection  # 새 컬렉션 반환

    def has_pic_tag(self, text: str) -> bool:  # <PIC> 태그 포함 여부 확인 메서드
        """텍스트에 <PIC> 태그가 포함되어 있는지 확인"""
        return "<PIC>" in text  # <PIC> 태그 포함 여부 반환

    @torch.inference_mode()  # 추론 모드로 설정 (그래디언트 계산 비활성화)
    def embed_text(self, texts: List[str]) -> np.ndarray:  # 텍스트 임베딩 생성 메서드
        """BGE-M3 모델을 사용하여 텍스트들의 임베딩 생성"""
        if isinstance(texts, str):  # 단일 문자열인 경우
            texts = [texts]  # 리스트로 변환

        embeddings = self.embedding_model.encode(  # BGE-M3로 임베딩 생성
            texts,
            batch_size=32,  # 배치 크기 설정
            max_length=512,  # 최대 길이 설정
        )

        # BGE-M3는 dict 형태로 반환하므로 dense_vecs 추출
        if isinstance(embeddings, dict):  # dict 형태인 경우
            dense_embeddings = np.array(
                embeddings["dense_vecs"], dtype=np.float32
            )  # dense_vecs 추출
        else:  # 이미 numpy 배열인 경우
            dense_embeddings = np.array(embeddings, dtype=np.float32)  # 그대로 사용

        # L2 정규화 수행 (단위 벡터로 만들기)
        norms = np.linalg.norm(
            dense_embeddings, axis=1, keepdims=True
        )  # 각 벡터의 L2 norm 계산
        norms = np.where(norms == 0, 1, norms)  # 0으로 나누는 것 방지
        normalized_embeddings = dense_embeddings / norms  # L2 정규화 수행

        # 정규화 확인 (디버깅용)
        final_norms = np.linalg.norm(normalized_embeddings, axis=1)
        print(
            f"벡터 정규화 확인 - Min norm: {final_norms.min():.6f}, Max norm: {final_norms.max():.6f}"
        )

        return normalized_embeddings  # 정규화된 벡터 반환

    def process_jsonl_file(
        self, file_path: str
    ) -> List[Dict[str, Any]]:  # JSONL 파일 처리 메서드
        """JSONL 파일을 읽어서 처리할 문서 데이터 생성"""
        processed_docs = []  # 처리된 문서들 저장 리스트

        print(f"JSONL 파일 '{file_path}'을(를) 처리 중...")  # 파일 처리 시작 메시지

        with open(file_path, "r", encoding="utf-8") as f:  # 파일 열기
            for line_num, line in enumerate(
                tqdm(f, desc="문서 처리 중"), 1
            ):  # 진행률 표시하며 순회
                try:
                    data = json.loads(line.strip())  # JSON 라인 파싱
                    doc_id = data[0]  # 문서 ID 추출
                    text = data[1]  # 텍스트 추출
                    image_ids = data[2]  # 이미지 ID 리스트 추출

                    # SentenceSplitter로 256 크기로 분할
                    if len(text) <= self.config.chunk_size:  # 짧은 텍스트는 그대로 사용
                        # <PIC> 태그가 포함되어 있는지 확인
                        chunk_image_ids = image_ids if self.has_pic_tag(text) else []
                        processed_docs.append(
                            {  # 처리된 문서 추가
                                "text": text,
                                "image_ids": chunk_image_ids,
                                "doc_id": doc_id,
                            }
                        )
                    else:  # 긴 텍스트는 SentenceSplitter로 분할
                        sub_chunks = self.splitter.split_text(text)
                        for sub_chunk in sub_chunks:  # 각 서브 청크 처리
                            # <PIC> 태그가 포함되어 있는지 확인
                            chunk_image_ids = (
                                image_ids if self.has_pic_tag(sub_chunk) else []
                            )
                            processed_docs.append(
                                {  # 처리된 문서 추가
                                    "text": sub_chunk,
                                    "image_ids": chunk_image_ids,
                                    "doc_id": doc_id,
                                }
                            )

                except json.JSONDecodeError as e:  # JSON 파싱 에러 처리
                    print(f"JSON 파싱 에러 (라인 {line_num}): {e}")  # 에러 메시지
                    continue  # 다음 라인으로 진행
                except Exception as e:  # 기타 예외 처리
                    print(f"처리 에러 (라인 {line_num}): {e}")  # 에러 메시지
                    continue  # 다음 라인으로 진행

        print(
            f"총 {len(processed_docs)}개의 문서 청크가 생성되었습니다."
        )  # 처리 완료 메시지
        return processed_docs  # 처리된 문서들 반환

    def insert_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 100
    ) -> List[int]:  # 문서 배치 삽입 메서드
        """처리된 문서들을 배치 단위로 Milvus에 삽입"""
        if not documents:  # 문서가 없는 경우
            return []  # 빈 리스트 반환

        print(
            f"총 {len(documents)}개의 문서를 Milvus에 삽입합니다..."
        )  # 삽입 시작 메시지

        all_pks = []  # 모든 기본키 저장 리스트

        # 배치 단위로 처리
        for i in tqdm(range(0, len(documents), batch_size), desc="배치 삽입 중"):
            batch_docs = documents[i : i + batch_size]  # 현재 배치 문서들

            # 배치 데이터 준비
            texts = [doc["text"] for doc in batch_docs]  # 텍스트 리스트
            image_ids_list = [
                doc["image_ids"] for doc in batch_docs
            ]  # 이미지 ID 리스트

            # 이미지 ID 리스트를 JSON 문자열로 변환
            image_ids_strs = []
            for img_ids in image_ids_list:
                if img_ids:  # 이미지 ID가 있는 경우
                    image_ids_strs.append(json.dumps(img_ids))  # JSON 문자열로 변환
                else:  # 이미지 ID가 없는 경우
                    image_ids_strs.append("[]")  # 빈 리스트 문자열

            # 텍스트 임베딩 생성
            embeddings = self.embed_text(texts)  # BGE-M3로 임베딩 생성

            # 원본 문서 ID 리스트 (JSONL의 첫 번째 컬럼을 정수로 사용)
            original_ids = [int(doc["doc_id"]) for doc in batch_docs]

            # Milvus 삽입 데이터 구성
            insert_data = [  # 삽입할 데이터
                original_ids,  # 원본 문서 ID 리스트
                texts,  # 텍스트 데이터 (리스트)
                image_ids_strs,  # 이미지 ID 문자열 데이터 (리스트)
                [vec.tolist() for vec in embeddings],  # 벡터 데이터 (리스트)
            ]

            # Milvus에 삽입
            mr = self.collection.insert(insert_data)  # 삽입 실행
            batch_pks = [int(pk) for pk in mr.primary_keys]  # 기본키 추출
            all_pks.extend(batch_pks)  # 전체 기본키에 추가

            # 주기적으로 플러시 (메모리 절약)
            if (i // batch_size + 1) % 10 == 0:  # 10배치마다 플러시
                self.collection.flush()  # 컬렉션 플러시
                print(f"  {i + len(batch_docs)}개 문서 처리됨 (플러시 완료)")

        # 최종 플러시
        self.collection.flush()  # 최종 플러시
        print(f"총 {len(all_pks)}개의 문서가 성공적으로 삽입되었습니다.")  # 완료 메시지

        return all_pks  # 기본키 리스트 반환

    def create_index(self):  # 인덱스 생성 메서드
        """벡터 필드에 인덱스 생성"""
        print("벡터 인덱스를 생성 중...")  # 인덱스 생성 시작 메시지

        # 기존 인덱스 확인
        try:
            existing_indexes = {
                idx.field_name for idx in (self.collection.indexes or [])
            }
        except Exception:
            existing_indexes = set()

        # 벡터 필드 인덱스 생성
        if "vector" not in existing_indexes:  # 인덱스가 없는 경우
            self.collection.create_index(  # 인덱스 생성
                field_name="vector",
                index_params={
                    "index_type": "HNSW",  # HNSW 인덱스 타입
                    "metric_type": "IP",  # 내적 메트릭
                    "params": {"M": 16, "efConstruction": 200},  # 인덱스 파라미터
                },
            )
            print("벡터 인덱스 생성 완료!")  # 완료 메시지
        else:
            print("벡터 인덱스가 이미 존재합니다.")  # 기존 인덱스 메시지

        # 컬렉션 로드
        self.collection.load()  # 컬렉션 로드
        print("컬렉션이 로드되었습니다.")  # 로드 완료 메시지

    def search(
        self, query: str, top_k: int = 5, include_image_docs: bool = True
    ) -> List[Dict[str, Any]]:  # 검색 메서드
        """쿼리 텍스트로 유사한 문서 검색"""
        # 쿼리 임베딩 생성
        query_embedding = self.embed_text([query])[0]  # 쿼리 임베딩 생성

        # 검색 파라미터
        search_params = {"metric_type": "IP", "params": {"ef": 128}}  # 검색 파라미터

        # 검색 실행
        results = self.collection.search(  # 검색 실행
            data=[query_embedding.tolist()],  # 쿼리 벡터
            anns_field="vector",  # 벡터 필드
            param=search_params,  # 검색 파라미터
            limit=top_k,  # 상위 K개
            output_fields=["text", "image_ids"],  # 출력 필드
        )

        # 결과 포맷팅
        formatted_results = []  # 포맷된 결과 저장 리스트
        for hit in results[0]:  # 첫 번째 결과 그룹
            image_ids_str = hit.entity.get("image_ids", "[]")  # 이미지 ID 문자열
            try:
                image_ids = json.loads(image_ids_str)  # JSON 파싱
            except:
                image_ids = []  # 파싱 실패 시 빈 리스트

            # original_id는 result 딕셔너리에서 직접 가져오기
            original_id = hit.get("original_id", 0)
            if original_id is None:
                original_id = 0

            formatted_results.append(
                {  # 포맷된 결과 추가
                    "id": int(hit.id),  # 자동 생성된 기본 키
                    "original_id": original_id,  # 원본 문서 ID (문자열)
                    "text": hit.entity.get("text", ""),  # 텍스트
                    "image_ids": image_ids,  # 이미지 ID 리스트
                    "score": float(hit.score),  # 유사도 점수
                }
            )

        return formatted_results  # 포맷된 결과 반환


def main():  # 메인 함수 - 스크립트 실행 시 호출
    """DocWit 벡터 저장소의 메인 실행 함수"""
    # 설정 생성
    config = DocWitConfig()  # 기본 설정 사용

    # DocWitVectorStore 인스턴스 생성
    store = DocWitVectorStore(config)  # 저장소 인스턴스 생성

    # JSONL 파일 경로
    jsonl_file_path = "/home/kun/Desktop/multimodal/data/MRAMG-Bench/doc_wit.jsonl"

    # 1단계: JSONL 파일 처리
    documents = store.process_jsonl_file(jsonl_file_path)

    # 2단계: 문서들 Milvus에 삽입
    primary_keys = store.insert_documents(documents)

    # 3단계: 인덱스 생성
    store.create_index()

    # 4단계: 검색 테스트
    print("\n" + "=" * 60)  # 구분선
    print("검색 테스트")  # 테스트 시작 메시지
    print("=" * 60)  # 구분선

    # 테스트 쿼리
    test_queries = [  # 테스트 쿼리 리스트
        "football player biography",
        "Italian municipality",
        "politician from Netherlands",
    ]

    for query in test_queries:  # 각 쿼리 테스트
        print(f"\n쿼리: '{query}'")  # 쿼리 출력
        results = store.search(query, top_k=3)  # 검색 실행
        for i, result in enumerate(results, 1):  # 결과 출력
            print(
                f"  {i}. [자동ID:{result['id']}, 원본ID:{result['original_id']}] {result['text'][:100]}..."
            )
            print(f"      이미지: {result['image_ids']}, 점수: {result['score']:.3f}")


if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()  # 메인 함수 호출
