# %%
from datasets import load_dataset, DatasetDict
# %%
# Parquet 브랜치에서 바로 불러오기
ds = load_dataset(
    "nlphuji/flickr30k",
    revision="refs/convert/parquet",   # ← 중요
)  # DatasetDict 반환 예상
print(ds)  # train/val/test 등 스플릿 확인

# %%
d = ds["test"]
# %%
# 1) split 분포 확인
from collections import Counter
print(Counter(d["split"])) # Counter({'train': 29000, 'val': 1014, 'test': 1000})
# %%
# 2) HF 스타일로 train/val/test 분리
splits = DatasetDict({
    "train": d.filter(lambda x: x["split"] == "train"),
    "validation": d.filter(lambda x: x["split"] == "val"),  # 'val' 표기
    "test": d.filter(lambda x: x["split"] == "test"),
})
print(splits)

# %%
# 3) 데이터셋 저장
splits.save_to_disk("data/flickr30k-split")




# %%
