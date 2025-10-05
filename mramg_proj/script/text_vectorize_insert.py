from mramg_proj.doc_vector_store import DocVectorStore
from mramg_proj.doc_vector_store_config import DocVectorStoreConfig


def main():  # 메인 함수 - 스크립트 실행 시 호출
    """Doc 벡터 저장소의 메인 실행 함수"""
    # 설정 생성 - 컬렉션 이름 지정
    config = DocVectorStoreConfig(
        collection_name="doc_manual_documents"
    )  # 기본 설정 사용

    # DocVectorStore 인스턴스 생성 - 임베더, 프로세서, DB 관리자가 자동으로 초기화됨
    store = DocVectorStore(config)

    # JSONL 파일 경로 지정
    jsonl_file_path = "/home/kun/Desktop/multimodal/data/MRAMG-Bench/doc_manual.jsonl"

    # 1단계: JSONL 파일 처리 및 Milvus에 삽입
    # DocVectorStore가 내부적으로 문서 처리와 임베딩을 수행함
    store.insert_documents(jsonl_file_path)

    # 2단계: 검색 성능 향상을 위한 인덱스 생성
    store.create_index()


if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()  # 메인 함수 호출
