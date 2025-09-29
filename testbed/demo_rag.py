from typing import List  # 타입 힌트를 위한 List 임포트

from PIL import Image  # 로컬 이미지 로딩을 위한 PIL 임포트
from rag_langchain import (  # 방금 구현한 RAG 구성요소 임포트
    CLIPMilvusRetriever,  # 텍스트→이미지 검색을 수행하는 LangChain 검색기
    LlavaImageDescriber,  # LLaVA 기반 이미지 설명 생성기
    MultiModalRAG,  # 검색기+생성기 결합 파이프라인
)

from MultiModalStore import (  # 기존 MinIO/Milvus/CLIP 스토어 래퍼 임포트
    MinIOConfig,  # MinIO 설정 데이터클래스
    MultiModalStore,  # 업로드/임베딩/삽입/검색 기능 제공 클래스
    VectorStoreConfig,  # Milvus 설정 데이터클래스
)


def ingest_images(
    store: MultiModalStore, images: List[Image.Image], captions: List[str]
):  # 간단 삽입 유틸 함수(삽입_이미지)
    assert len(images) == len(captions), (
        "이미지와 캡션 길이가 달라요"
    )  # 입력 유효성 검증(이미지/캡션 수 일치 확인)
    pks = store.add_images_with_captions(  # 스토어를 통해 업로드/임베딩/밀버스 삽입을 일괄 처리
        images,
        captions,
        object_names=None,
        batch_size=len(images),
        do_flush=True,  # 한 번에 처리 후 flush로 보장
    )
    print(f"inserted {len(pks)} images")  # 실제 삽입 개수 출력


def main():  # 데모 진입점 함수(메인)
    query: str = "a cute cat"  # 사용자 질의를 직접 변수로 지정(쿼리)
    ingest: bool = False  # 예시 삽입 실행 여부를 변수로 제어(인제스트)

    # 스토어(검색 인프라) 준비
    minio_cfg = MinIOConfig()  # MinIO 설정 생성
    milvus_cfg = VectorStoreConfig()  # Milvus 설정 생성
    store = MultiModalStore(minio_cfg, milvus_cfg)  # 스토어 인스턴스 생성

    if ingest:  # 삽입 플래그가 True이면 예시 데이터 삽입 수행
        img1 = Image.new(
            "RGB", (256, 256), color=(255, 200, 200)
        )  # 단색 이미지 1 생성(핑크톤)
        img2 = Image.new(
            "RGB", (256, 256), color=(200, 200, 255)
        )  # 단색 이미지 2 생성(블루톤)
        ingest_images(
            store, [img1, img2], ["핑크색 배경의 정사각형", "파란색 배경의 정사각형"]
        )  # 간단 캡션과 함께 삽입

    # RAG 파이프라인 구성
    retriever = CLIPMilvusRetriever(
        store=store, top_k=1
    )  # 상위 3개 검색 결과를 반환하도록 검색기 구성
    generator = LlavaImageDescriber()  # LLaVA 생성기 초기화(기본 모델/옵션 사용)
    rag = MultiModalRAG(retriever, generator)  # 검색+생성을 결합한 파이프라인 생성

    # 실행
    result = rag.invoke(
        query, use_top_k=1
    )  # 질의를 수행하고 첫 번째 결과에 대해 설명 생성
    print("=== Answer ===")  # 구분선 출력
    print(result["answer"])  # 생성된 답변 출력
    if result.get("contexts"):  # 컨텍스트가 존재한다면
        first_ctx = result["contexts"][0]  # 첫 번째 컨텍스트 선택(선택된 이미지 정보)
        image_url = (first_ctx.get("metadata") or {}).get(
            "image_url"
        )  # 메타데이터에서 이미지 URL 추출
        if image_url:  # URL이 존재하면 출력
            print("=== Image URL ===")  # 이미지 URL 구분선 출력
            print(image_url)  # 사용된 이미지의 presigned URL 출력
    print("=== Contexts ===")  # 컨텍스트 구분선 출력
    for ctx in result["contexts"]:
        print(
            {k: ctx.get(k) for k in ("page_content", "metadata")}
        )  # 컨텍스트 요약 출력


if __name__ == "__main__":  # 스크립트 직접 실행 시 진입
    main()  # 메인 함수 호출
