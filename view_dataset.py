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
from IPython.display import display
import random

d = splits["test"]     # 또는 "validation", "test"
i = random.randrange(len(d))
ex = d[i]

print(ex["filename"], ex["img_id"], ex["split"])
display(ex["image"])            # PIL 이미지 표시
print("\n".join(ex["caption"])) # 5개 캡션 줄바꿈 출력
# %%
print(ex)
# %%
