# %%
from datasets import load_dataset

# Parquet 브랜치에서 바로 불러오기
ds = load_dataset(
    "nlphuji/flickr30k",
    revision="refs/convert/parquet",   # ← 중요
)  # DatasetDict 반환 예상
print(ds)  # train/val/test 등 스플릿 확인

# %%
ds["train"][0]

# %%
from transformers import CLIPProcessor, CLIPModel
# %%
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"  # 768-d 공용 투영
model = CLIPModel.from_pretrained(model_id).to(device).eval()
proc  = CLIPProcessor.from_pretrained(model_id)


