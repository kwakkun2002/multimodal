from datasets import (
    Dataset,
    load_from_disk,  # 디스크에 저장된 허깅페이스 데이터셋을 로드하는 함수 임포트
)
from torch.utils.data import DataLoader

from MultiModalStore import MilvusConfig, MinIOConfig

splits = load_from_disk("data/flickr30k-split")

train_dataset: Dataset = splits["train"]  # 학습용 데이터셋 분할 추출
validation_dataset: Dataset = splits["validation"]  # 검증용 데이터셋 분할 추출
test_dataset: Dataset = splits["test"]  # 테스트용 데이터셋 분할 추출


def collate_pil_with_caption(batch: Dataset):
    return {
        "image": [ex["image"] for ex in batch],
        "caption": [ex["caption"] for ex in batch],
    }


train_loader = DataLoader(
    dataset=train_dataset,  # 학습용 데이터셋 분할 로더
    batch_size=32,  # 배치 크기
    shuffle=False,  # 데이터 섞지 않음
    num_workers=4,  # 워커 수
    collate_fn=collate_pil_with_caption,  # 콜레이트 함수
)

val_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_pil_with_caption,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_pil_with_caption,
)

minio_cfg = MinIOConfig()
milvus_cfg = MilvusConfig()

