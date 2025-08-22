# %%
from datasets import load_dataset, load_from_disk
# %%

splits = load_from_disk("data/flickr30k-split")


# %%
from IPython.display import display
import random

d = splits["train"]     # 또는 "validation", "test"
i = random.randrange(len(d))
ex = d[i]

print(ex["filename"], ex["img_id"], ex["split"])
display(ex["image"])            # PIL 이미지 표시
print("\n".join(ex["caption"])) # 5개 캡션 줄바꿈 출력

# %%
from clip_embed import embed_text, embed_images
# %%
texts = [c for c in ex["caption"]]
text_feats = embed_text(texts)
print(text_feats.shape)
print(text_feats)

# %%
not_relevant_text = "a photo of a cat"
not_relevant_text_feats = embed_text(not_relevant_text)
print(not_relevant_text_feats.shape)
print(not_relevant_text_feats)

# %%
image = ex["image"]
image_feats = embed_images(image)
print(image_feats.shape)
print(image_feats)

# %%
import numpy as np

print(np.dot(text_feats[0], image_feats[0]))

# %%
print(np.dot(not_relevant_text_feats[0], image_feats[0]))
# %%
