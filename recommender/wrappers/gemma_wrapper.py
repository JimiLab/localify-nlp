class GemmaWrapper:

    def __init__(self, model, tokenizer, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def get_messages(self, system_message, user_message):
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}, ]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}, ]
                },
            ],
        ]
        return messages

    def get_response(self, prompt):
        messages = self.get_messages(self.system_prompt, prompt)

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        generation = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        print(decoded)
        return decoded
