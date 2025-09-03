# %%  # 셀 구분 주석(노트북 스타일 시각 구분)
from datasets import (
    load_from_disk,  # 디스크에 저장된 HuggingFace 데이터셋을 로드하는 함수 임포트
)
from PIL import Image  # 이미지 타입 판별과 크기 확인을 위한 PIL 이미지 클래스 임포트
from torch.utils.data import (
    DataLoader,  # 배치 단위 처리를 위한 PyTorch DataLoader 임포트
)
from tqdm import tqdm  # 진행 상황 표시를 위한 tqdm 진행바 임포트

from MultiModalStore import (  # Milvus/MinIO 설정 및 통합 스토리지 래퍼 임포트
    MilvusConfig,
    MinIOConfig,
    MultiModalStore,
)

# %%  # 셀 구분 주석(데이터셋 로딩 블록)

splits = load_from_disk(
    "data/flickr30k-split"
)  # 디스크에서 Flickr30k 분할 데이터셋 로드

train_ds = splits["train"]  # 학습용 데이터셋 분할 추출
val_ds = splits["validation"]  # 검증용 데이터셋 분할 추출
test_ds = splits["test"]  # 테스트용 데이터셋 분할 추출

# %%  # 셀 구분 주석(콜레이트/로더 구성 블록)


def collate_pil_batch(
    batch,
):  # 배치에서 PIL 이미지를 그대로 유지하고 캡션을 나란히 묶는 콜레이트 함수(콜레이트_배치)
    # PIL 이미지를 그대로 리스트로 유지하고, 캡션도 동일 길이 리스트로 묶음  # 기본 콜레이트의 텐서 스택을 방지하여 타입 오류 예방
    return {
        "image": [
            ex["image"] for ex in batch
        ],  # 각 샘플의 이미지만 추출하여 리스트로 구성
        "caption": [
            ex["caption"] for ex in batch
        ],  # 각 샘플의 캡션만 추출하여 리스트로 구성
    }


# DataLoader 구성 (PIL-safe collate)  # 커스텀 콜레이트를 지정해 PIL 이미지를 안전하게 배치 처리
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_pil_batch,  # 학습 분할 로더: 배치 32, 워커 4
)
val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_pil_batch,  # 검증 분할 로더: 배치 32, 워커 2
)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_pil_batch,  # 테스트 분할 로더: 배치 32, 기본 워커
)

# %%  # 셀 구분 주석(저장소 설정 및 드라이런)

minio_cfg = MinIOConfig()  # MinIO 연결/버킷 정보 등을 담는 설정 객체 생성(미니오_설정)
milvus_cfg = (
    MilvusConfig()
)  # Milvus 연결/컬렉션 정보 등을 담는 설정 객체 생성(밀버스_설정)

# 드라이런 모드: True이면 Milvus/MinIO 초기화 및 삽입을 건너뜀  # 안전 점검 시 외부 시스템 쓰기를 방지
DRY_RUN = False  # 실제 적재 실행 여부를 제어하는 스위치(드라이런)

if not DRY_RUN:  # 드라이런이 아닐 때만 외부 스토어 초기화
    store = MultiModalStore(
        minio_cfg, milvus_cfg
    )  # 멀티모달 스토어 인스턴스 생성(스토어)


def _join_caption(c):  # 캡션을 문자열로 일관화하는 헬퍼 함수(캡션_합치기)
    if isinstance(c, (list, tuple)):  # 캡션이 토큰 리스트/튜플 형태인 경우
        return " ".join(
            str(x) for x in c
        )  # 공백으로 이어붙여 하나의 문장 문자열로 변환
    return str(c)  # 이미 문자열이거나 기타 타입이면 문자열로 변환해 반환


def ingest_split(
    split_name, loader
):  # 지정한 데이터 분할을 배치 순회하며 스토어에 적재하는 함수(분할_적재)
    total = 0  # 누적 삽입(또는 시뮬레이션) 건수를 집계할 변수
    for batch in tqdm(
        loader, desc=f"Ingest {split_name}", leave=False
    ):  # 진행바로 적재 과정을 시각화하며 순회
        images = batch["image"]  # 현재 배치의 이미지 리스트 추출
        captions_raw = batch["caption"]  # 현재 배치의 원본 캡션 리스트 추출
        captions = [
            _join_caption(c) for c in captions_raw
        ]  # 캡션을 문자열로 정규화하여 리스트 생성
        if DRY_RUN:  # 드라이런 모드일 경우 실제 삽입 대신 집계만 수행
            # 드라이런 시에는 삽입하지 않고 건너뜀  # 외부 시스템 변경을 방지하여 안전 검증 수행
            total += len(images)  # 이번 배치의 샘플 수를 누적 집계
            continue  # 다음 배치로 진행
        pks = store.add_images_with_captions(  # 스토어에 이미지+캡션을 일괄 삽입하고 PK 목록 반환
            images,  # 삽입할 이미지 리스트
            captions,  # 삽입할 캡션 문자열 리스트
            batch_size=len(images),  # 현재 배치 크기만큼 처리하도록 명시
            do_flush=False,  # 매 배치마다 flush 하지 않고 마지막에 한 번에 flush
        )
        total += len(pks)  # 실제 삽입된 레코드 수를 누적 집계
    if not DRY_RUN:  # 드라이런이 아닐 때만 최종 flush 수행
        store.coll.flush()  # 컬렉션 버퍼를 디스크/인덱스에 반영하도록 플러시
        print(f"{split_name}: inserted {total}")  # 실제 삽입된 총 개수 출력
    else:  # 드라이런인 경우
        print(
            f"{split_name}: would insert {total} items (dry-run)"
        )  # 예상 삽입 개수(시뮬레이션 결과) 출력


def inspect_split(
    split_name, loader, max_batches: int = 2
):  # 분할의 앞쪽 배치를 샘플로 점검하는 함수(분할_점검)
    print(f"\n[Inspect] {split_name}")  # 어떤 분할을 점검 중인지 헤더 출력
    for bi, batch in enumerate(loader):  # 배치 인덱스와 함께 순회 시작
        if bi >= max_batches:  # 지정한 최대 배치 수만큼만 점검
            break  # 루프 종료
        images = batch["image"]  # 현재 배치의 이미지 리스트 추출
        captions_raw = batch["caption"]  # 현재 배치의 원본 캡션 리스트 추출
        batch_size = len(images)  # 배치 크기 계산
        same_len = batch_size == len(captions_raw)  # 이미지 수와 캡션 수가 같은지 확인
        imgs_are_pil = all(
            isinstance(img, Image.Image) for img in images
        )  # 모든 이미지가 PIL 타입인지 검사
        first_size = (  # 첫 번째 이미지의 크기 확인(데이터 일관성 확인용)
            images[0].size  # (폭, 높이)
            if batch_size > 0
            and isinstance(
                images[0], Image.Image
            )  # 배치가 비어있지 않고 PIL 타입일 때만 접근
            else None  # 조건을 만족하지 않으면 None
        )
        cap0 = (
            captions_raw[0] if batch_size > 0 else None
        )  # 첫 번째 캡션 샘플 추출(있을 때만)
        cap0_type = (
            type(cap0).__name__ if cap0 is not None else None
        )  # 첫 캡션의 타입명(str/list/tuple 등)
        cap0_len = (
            len(cap0) if isinstance(cap0, (list, tuple)) else None
        )  # 토큰 배열일 경우 길이
        print(  # 배치 요약 정보를 한 줄로 출력
            f" batch {bi}: size={batch_size}, same_len={same_len}, PIL={imgs_are_pil}, first_img_size={first_size}, caption0_type={cap0_type}, caption0_len={cap0_len}"
        )
        # 캡션 전처리 예시 확인  # 샘플 캡션의 앞부분을 미리보기로 제공하여 데이터 품질 확인
        if cap0 is not None:  # 캡션이 존재할 때만 미리보기 생성
            preview = (
                cap0 if isinstance(cap0, str) else " ".join(map(str, cap0[:2]))
            )  # 문자열이면 그대로, 토큰 배열이면 앞 2개 결합
            print(f"  caption0_preview: {preview!r}")  # repr 형식으로 출력(따옴표 포함)


# 필요 시 배치 구조 점검용  # 실제 적재 전에 데이터 형태를 빠르게 검증할 때 사용
# inspect_split("test", test_loader)  # 테스트 분할의 앞 배치 몇 개를 확인하는 예시 호출

if not DRY_RUN:  # 드라이런이 아닐 때만 실제 적재 실행
    ingest_split(
        "test", test_loader
    )  # 테스트 분할을 대상으로 적재 실행(처음엔 test로 검증 후 확장 권장)
