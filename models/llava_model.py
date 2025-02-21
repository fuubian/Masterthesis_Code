from models.text_image_model import TextImageModel
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

class LLaVaModel(TextImageModel):
    def __init__(self, model_name="llava-hf/llava-1.5-13b-hf"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=30):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "file_path": image},
                    {"type": "text", "text": question_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        generate_ids = self.model.generate(**inputs, max_new_tokens=max_answer_length)
        model_response = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        model_response = model_response.split("ASSISTANT:")[-1].strip()

        return model_response

    def _load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_name)