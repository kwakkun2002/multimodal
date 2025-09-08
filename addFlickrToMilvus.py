# %%  # 셀 구분 주석(노트북 스타일 시각 구분)
from datasets import (
    load_from_disk,  # 디스크에 저장된 HuggingFace 데이터셋을 로드하는 함수 임포트
)
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
    batch,  # DataLoader가 전달하는 배치 샘플 리스트, 각 샘플은 딕셔너리 형태로 'image', 'caption', 'img_id' 등의 키를 포함
):  # 배치에서 PIL 이미지를 그대로 유지하고 캡션과 이미지 ID를 나란히 묶는 콜레이트 함수(콜레이트_배치)
    # PIL 이미지를 그대로 리스트로 유지하고, 캡션과 이미지 ID도 동일 길이 리스트로 묶음  # 기본 콜레이트의 텐서 스택을 방지하여 타입 오류 예방
    return {
        "image": [
            ex["image"] for ex in batch
        ],  # 각 샘플의 이미지만 추출하여 리스트로 구성
        "caption": [
            ex["caption"] for ex in batch
        ],  # 각 샘플의 캡션만 추출하여 리스트로 구성
        "img_id": [
            ex["img_id"] for ex in batch
        ],  # 각 샘플의 이미지 ID만 추출하여 리스트로 구성
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

# %%  # 셀 구분 주석(저장소 설정)

minio_cfg = MinIOConfig()  # MinIO 연결/버킷 정보 등을 담는 설정 객체 생성(미니오_설정)
milvus_cfg = (
    MilvusConfig()
)  # Milvus 연결/컬렉션 정보 등을 담는 설정 객체 생성(밀버스_설정)

store = MultiModalStore(minio_cfg, milvus_cfg)  # 멀티모달 스토어 인스턴스 생성(스토어)


def _join_caption(c):  # 캡션을 문자열로 일관화하는 헬퍼 함수(캡션_합치기)
    # c: 캡션 데이터 - 문자열, 리스트, 튜플 등 다양한 형태로 들어올 수 있는 캡션
    if isinstance(c, (list, tuple)):  # 캡션이 토큰 리스트/튜플 형태인 경우
        return " ".join(
            str(x) for x in c
        )  # 공백으로 이어붙여 하나의 문장 문자열로 변환
    return str(c)  # 이미 문자열이거나 기타 타입이면 문자열로 변환해 반환


def ingest_split(
    split_name,  # 데이터 분할 이름 문자열 (예: 'train', 'validation', 'test') - 진행바 표시와 로그에 사용
    loader,  # PyTorch DataLoader 객체 - 배치 단위로 데이터를 제공하는 이터레이터
):  # 지정한 데이터 분할을 배치 순회하며 스토어에 적재하는 함수(분할_적재)
    total = 0  # 누적 삽입 건수를 집계할 변수
    for batch in tqdm(
        loader, desc=f"Ingest {split_name}", leave=False
    ):  # 진행바로 적재 과정을 시각화하며 순회
        images = batch["image"]  # 현재 배치의 이미지 리스트 추출
        captions_raw = batch["caption"]  # 현재 배치의 원본 캡션 리스트 추출
        image_ids = batch["img_id"]  # 현재 배치의 이미지 ID 리스트 추출
        captions = [
            _join_caption(c) for c in captions_raw
        ]  # 캡션을 문자열로 정규화하여 리스트 생성
        pks = store.add_images_with_captions(  # 스토어에 이미지+캡션을 일괄 삽입하고 PK 목록 반환
            images,  # 삽입할 이미지 리스트
            captions,  # 삽입할 캡션 문자열 리스트
            object_names=image_ids,  # 이미지 ID를 MinIO 객체 이름으로 사용
            batch_size=len(images),  # 현재 배치 크기만큼 처리하도록 명시
            do_flush=False,  # 매 배치마다 flush 하지 않고 마지막에 한 번에 flush
        )
        total += len(pks)  # 실제 삽입된 레코드 수를 누적 집계
    store.coll.flush()  # 컬렉션 버퍼를 디스크/인덱스에 반영하도록 플러시
    print(f"{split_name}: inserted {total}")  # 실제 삽입된 총 개수 출력


ingest_split(
    "test", test_loader
)  # 테스트 분할을 대상으로 적재 실행(처음엔 test로 검증 후 확장 권장)
