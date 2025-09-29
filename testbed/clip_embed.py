import torch
from transformers import CLIPModel, AutoTokenizer, AutoImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"  # 768-d common projection

model = CLIPModel.from_pretrained(model_id).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

print(device)

@torch.no_grad()
def embed_text(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy()

@torch.no_grad()
def embed_images(pil_images):
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy()
