# Following the instructions on https://huggingface.co/openbmb/MiniCPM-V-2_6
# Note: This model does not run with transformers==4.49.0
# Check this discussion: https://huggingface.co/openbmb/MiniCPM-V-2_6/discussions/53

from models.text_image_model import TextImageModel
from utils.token_loader import TokenLoader
from huggingface_hub import login
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class MiniCPMModel(TextImageModel):
    def __init__(self, model_name='openbmb/MiniCPM-V-2_6'):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=1024):
        image = Image.open(image).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, question_prompt]}]

        model_response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=max_answer_length
        )

        return model_response

    def _load_model(self):
        access_token = TokenLoader.load_token_huggingface()
        login(access_token)

        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True,
                attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=True)