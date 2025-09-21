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
import torch
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"  # 768-d 공용 투영
model = CLIPModel.from_pretrained(model_id).to(device).eval()
proc = CLIPProcessor.from_pretrained(model_id)


# %%

# jinja-clip-v2 불러오기
# from transformers import AutoModel, AutoProcessor
# model = AutoModel.from_pretrained("jinhaai/jinja-clip-v2", trust_remote_code=True)
# processor = AutoProcessor.from_pretrained("jinhaai/jinja-clip-v2")

# 모델을 상업용으로 쓸 수 없어서 다운로드가 안됨
# API를 사용하는 방식으로 가야할듯?



# %%

# BGE-M3 불러오기
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
# emb = model.encode(["문장1", "문장2"], max_length=8192, batch_size=...)['dense_vecs']


# %%
