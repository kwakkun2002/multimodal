from DocWitConfig import DocWitConfig
from DocWitVectorStore import DocWitVectorStore


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
    store.insert_documents(documents)

    # 3단계: 인덱스 생성
    store.create_index()


if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()  # 메인 함수 호출
