from DocWitConfig import DocWitConfig
from DocWitVectorStore import DocWitVectorStore


def main():
    # DocWitVectorStore 인스턴스 생성
    config = DocWitConfig()  # 기본 설정 사용
    store = DocWitVectorStore(config)  # 저장소 인스턴스 생성

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


if __name__ == "__main__":
    main()
