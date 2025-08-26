from datasets import load_from_disk
from tqdm import tqdm

from MultiModalStore import MilvusConfig, MinIOConfig, MultiModalStore

splits = load_from_disk("data/flickr30k-split")

train_ds = splits["train"]
val_ds = splits["validation"]
test_ds = splits["test"]

minio_cfg = MinIOConfig()
milvus_cfg = MilvusConfig()

store = MultiModalStore(minio_cfg, milvus_cfg)

for i in tqdm(range(len(train_ds))):
    cap = " ".join(train_ds[i]["caption"])
    pk = store.add_image_with_caption(train_ds[i]["image"], cap)

    print(f"Added train image {i} with pk {pk}")

for i in tqdm(range(len(val_ds))):
    cap = " ".join(val_ds[i]["caption"])
    pk = store.add_image_with_caption(val_ds[i]["image"], cap)

    print(f"Added val image {i} with pk {pk}")

for i in tqdm(range(len(test_ds))):
    cap = " ".join(test_ds[i]["caption"])
    pk = store.add_image_with_caption(test_ds[i]["image"], cap)

    print(f"Added test image {i} with pk {pk}")

print(store.search_by_text(test_ds[0]["caption"]))
