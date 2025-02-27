from models.text_image_model import TextImageModel
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image

class InstructBlipModel(TextImageModel):
    def __init__(self, model_name="Salesforce/instructblip-vicuna-7b"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=500):
        image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, text=question_prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=max_answer_length,
            include_prompt_in_result=False,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )

        model_response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return model_response

    def _load_model(self):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(self.model_name, device_map="auto", quantization_config=quantization_config)
        self.processor = InstructBlipProcessor.from_pretrained(self.model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)