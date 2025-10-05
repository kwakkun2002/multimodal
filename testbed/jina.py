# %%
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch

model = AutoModel.from_pretrained("jinhaai/jinja-clip-v2", trust_remote_code=True)
# %%