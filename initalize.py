# %%
from datasets import load_dataset

# %%
# Parquet 브랜치에서 바로 불러오기
ds = load_dataset(
    "nlphuji/flickr30k",
    revision="refs/convert/parquet",  # ← 중요
)  # DatasetDict 반환 예상
print(ds)  # train/val/test 등 스플릿 확인


# %%
# 이 코드로는 데이터셋을 불러올 수 없다.
dataset = load_dataset("MRAMG/MRAMG-Bench", split="train", trust_remote_code=True)
print(dataset)

# 이런식으로 해야함
# 이미지 파일이 너무 커저 git lfs(large file storage) 설치 필요
# sudo apt update
# sudo apt install git-lfs
# git lfs install
# git lfs install
# git clone https://huggingface.co/datasets/MRAMG/MRAMG-Bench
# cd MRAMG-Bench

# %%
ds["train"][0]

# %%
# %%
import torch
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"  # 768-d 공용 투영
model = CLIPModel.from_pretrained(model_id).to(device).eval()
proc = CLIPProcessor.from_pretrained(model_id)
