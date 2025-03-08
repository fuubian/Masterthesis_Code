from models.text_image_model import TextImageModel
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class MiniCPMModel(TextImageModel):
    def __init__(self, model_name='openbmb/MiniCPM-V-2_6'):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=2500):
        image = Image.open(image).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, question_prompt]}]

        model_response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        return model_response

    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)