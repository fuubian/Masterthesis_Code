from models import text_image_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image

class LLaVa_model(text_image_model):
    def __init__(self, model_name="llava-hf/llava-1.5-13b"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image):
        pass

    def _load_model():
        pass