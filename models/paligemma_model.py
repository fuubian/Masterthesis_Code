from models.text_image_model import TextImageModel
from utils.token_loader import TokenLoader
from huggingface_hub import login
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import torch

class PaligemmaModel(TextImageModel):
    def __init__(self, model_name="google/paligemma2-10b-pt-448"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image):
        image = load_image(image)

        model_inputs = self.processor(text=question_prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            model_response = self.processor.decode(generation, skip_special_tokens=True)

        return model_response

    def _load_model(self):
        access_token = TokenLoader.load_token_huggingface()
        login(access_token)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_name)