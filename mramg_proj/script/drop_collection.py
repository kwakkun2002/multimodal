from pymilvus import connections, utility


def drop_collection(collection_name: str):
    """
    Milvus 컬렉션을 삭제하는 함수

    Args:
        collection_name: 삭제할 컬렉션 이름
    """
    # Milvus 서버에 연결
    connections.connect(alias="default", host="localhost", port=19532)

    # 컬렉션 존재 여부 확인
    if utility.has_collection(collection_name):
        # 컬렉션 삭제
        utility.drop_collection(collection_name)
        print(f"[DROP] 컬렉션 '{collection_name}' 삭제 완료")
    else:
        print(f"[DROP] 컬렉션 '{collection_name}'이 존재하지 않습니다")

    # 연결 종료
    connections.disconnect(alias="default")


if __name__ == "__main__":
    # 삭제할 컬렉션 이름 지정
    drop_collection("doc_manual_documents")
