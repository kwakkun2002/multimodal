import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
).to("cuda")

processor = AutoProcessor.from_pretrained(model_id)

# 프롬프트 구성
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "explain this image."},
            {"type": "image"},
        ],
    }
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# 이미지 로드
raw_image = Image.open("kitten.png")

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
    "cuda", torch.float16
)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
