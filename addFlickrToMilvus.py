# %%
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from MultiModalStore import MilvusConfig, MinIOConfig, MultiModalStore

# %%

splits = load_from_disk("data/flickr30k-split")

train_ds = splits["train"]
val_ds = splits["validation"]
test_ds = splits["test"]

# %%


def collate_pil_batch(batch):
    # PIL 이미지를 그대로 리스트로 유지하고, 캡션도 동일 길이 리스트로 묶음
    return {
        "image": [ex["image"] for ex in batch],
        "caption": [ex["caption"] for ex in batch],
    }


# DataLoader 구성 (PIL-safe collate)
train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_pil_batch
)
val_loader = DataLoader(
    val_ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_pil_batch
)
test_loader = DataLoader(
    test_ds, batch_size=32, shuffle=False, collate_fn=collate_pil_batch
)

# %%

minio_cfg = MinIOConfig()
milvus_cfg = MilvusConfig()

# 드라이런 모드: True이면 Milvus/MinIO 초기화 및 삽입을 건너뜀
DRY_RUN = False

if not DRY_RUN:
    store = MultiModalStore(minio_cfg, milvus_cfg)


def _join_caption(c):
    if isinstance(c, (list, tuple)):
        return " ".join(str(x) for x in c)
    return str(c)


def ingest_split(split_name, loader):
    total = 0
    for batch in tqdm(loader, desc=f"Ingest {split_name}", leave=False):
        images = batch["image"]
        captions_raw = batch["caption"]
        captions = [_join_caption(c) for c in captions_raw]
        if DRY_RUN:
            # 드라이런 시에는 삽입하지 않고 건너뜀
            total += len(images)
            continue
        pks = store.add_images_with_captions(
            images,
            captions,
            batch_size=len(images),
            do_flush=False,
        )
        total += len(pks)
    if not DRY_RUN:
        store.coll.flush()
        print(f"{split_name}: inserted {total}")
    else:
        print(f"{split_name}: would insert {total} items (dry-run)")


def inspect_split(split_name, loader, max_batches: int = 2):
    print(f"\n[Inspect] {split_name}")
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        images = batch["image"]
        captions_raw = batch["caption"]
        batch_size = len(images)
        same_len = batch_size == len(captions_raw)
        imgs_are_pil = all(isinstance(img, Image.Image) for img in images)
        first_size = (
            images[0].size
            if batch_size > 0 and isinstance(images[0], Image.Image)
            else None
        )
        cap0 = captions_raw[0] if batch_size > 0 else None
        cap0_type = type(cap0).__name__ if cap0 is not None else None
        cap0_len = len(cap0) if isinstance(cap0, (list, tuple)) else None
        print(
            f" batch {bi}: size={batch_size}, same_len={same_len}, PIL={imgs_are_pil}, first_img_size={first_size}, caption0_type={cap0_type}, caption0_len={cap0_len}"
        )
        # 캡션 전처리 예시 확인
        if cap0 is not None:
            preview = cap0 if isinstance(cap0, str) else " ".join(map(str, cap0[:2]))
            print(f"  caption0_preview: {preview!r}")


# 필요 시 배치 구조 점검용
# inspect_split("test", test_loader)

if not DRY_RUN:
    ingest_split("test", test_loader)
