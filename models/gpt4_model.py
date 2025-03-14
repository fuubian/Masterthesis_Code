import base64
from models.text_image_model import TextImageModel
from utils.token_loader import TokenLoader
from openai import OpenAI

class GPTModel(TextImageModel):
    def __init__(self, model_name="gpt4"):
        super().__init__(model_name)
        self._load_model()

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_answer(self, question_prompt, image, max_answer_length=None):
        image = self.encode_image(image)

        model_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}",
                            },
                        },
                    ],
                }
            ],
        )

        return model_response.choices[0].message.content
    
    def _load_model(self):
        self.client = OpenAI(api_key=TokenLoader.load_api_key_openai())