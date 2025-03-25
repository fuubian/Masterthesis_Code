# Following the instrctions on https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf

from models.text_image_model import TextImageModel
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class LLaVaModel(TextImageModel):
    def __init__(self, model_name="llava-hf/llava-v1.6-vicuna-13b-hf"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=500):
        image = Image.open(image)
    
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_prompt},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(image, prompt, return_tensors="pt").to("cuda:0")
        
        output = self.model.generate(**inputs, max_new_tokens=max_answer_length)
        model_response = self.processor.decode(output[0], skip_special_tokens=True)
        model_response = model_response.split("ASSISTANT:")[-1].strip()

        return model_response

    def _load_model(self):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model.to("cuda:0")
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)