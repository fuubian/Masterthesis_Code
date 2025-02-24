from models.text_image_model import TextImageModel
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class LLaVaModel(TextImageModel):
    def __init__(self, model_name="openbmb/MiniCPM-o-2_6"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=100):
        image = Image.open(image).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, question_prompt]}]
        
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=max_answer_length
        )
        
        return res

    def _load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False
        )

        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)