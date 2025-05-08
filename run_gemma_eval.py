import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
import yaml

with open('conf.yaml') as file:
    conf = yaml.safe_load(file)
    openai_key = conf['openai-key']

ckpt = "google/gemma-3-4b-it"
model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=openai_key,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-it",
    use_auth_token=openai_key,
)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a professional music recommendation expert that specializes in recommending local artists based on a user's specified, preferred artists."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "I really like Ariana Grande, Fleetwood Mac, BlackPink, Lady Gaga, Drake, and The Weeknd. What artists would you recommend from Ithaca, NY?"},]
        },
    ],
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)
