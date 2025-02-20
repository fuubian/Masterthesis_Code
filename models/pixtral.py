from models.text_image_model import TextImageModel
from vllm import LLM
from vllm.sampling_params import SamplingParams

class PixtralModel(TextImageModel):
    def __init__(self, model_name="mistralai/Pixtral-12B-2409"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_prompt}, 
                    {"type": "image_url", "image_url": {"url": image}}
                ]
            },
        ]

        model_response = self.model.chat(messages, sampling_params=self.sampling_params)
        return model_response[0].outputs[0].text

    def _load_model(self):
        self.sampling_params = SamplingParams(max_tokens=8192)
        self.model = LLM(model=self.model_name, tokenizer_mode="mistral")