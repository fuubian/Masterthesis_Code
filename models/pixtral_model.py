from models.text_image_model import TextImageModel
from utils.token_loader import TokenLoader
from PIL import Image
import base64
from huggingface_hub import login
from vllm import LLM
from vllm.sampling_params import SamplingParams

class PixtralModel(TextImageModel):
    def __init__(self, model_name="mistralai/Pixtral-12B-2409"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=1000):
        image = self._encode_image_to_base64(image)
        sampling_params = SamplingParams(max_tokens=max_answer_length)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_prompt}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                ]
            },
        ]

        model_response = self.model.chat(messages, sampling_params=sampling_params)
        return model_response[0].outputs[0].text

    def _load_model(self):
        access_token = TokenLoader.load_token_huggingface()
        login(access_token)
        
         #TODO: Remove multiple GPUs before submission
        self.model = LLM(model=self.model_name,
                         tokenizer_mode="mistral",
                         tensor_parallel_size=2,
                         gpu_memory_utilization=0.95,
                         max_model_len=80000)
        
    def _encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")