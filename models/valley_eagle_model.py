from models.text_image_model import TextImageModel
import torch
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModel

class ValleyEagleModel(TextImageModel):
    def __init__(self, model_name="bytedance-research/Valley-Eagle-7B"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=1024):
        image = Image.open(BytesIO(image)).convert("RGB")
        
        res = self.processor(
            {
                "conversations": 
                [
                    {"role": "system", "content": "You are Valley, developed by ByteDance. Your are a helpfull Assistant."},
                    {"role": "user", "content":  question_prompt},
                ], 
                "images": [image]
            }, 
            inference=True
        )

        with torch.inference_mode():
            self.model.to(dtype=torch.float16, device=self.device)
            output_ids = self.model.generate(
                input_ids=res["input_ids"].to(self.device),
                images=[[item.to(dtype=torch.float16, device=self.device) for item in img] for img in res["images"]],
                image_sizes=res["image_sizes"],
                pixel_values=res["pixel_values"].to(dtype=torch.float16, device=self.device),
                image_grid_thw=res["image_grid_thw"].to(self.device),
                do_sample=False,
                max_new_tokens=max_answer_length,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_token_len = res["input_ids"].shape[1]
        model_response = self.processor.batch_decode(output_ids.sequences[:, input_token_len:])[0]
        model_response = model_response.replace("<|im_end|>", "")

        return model_response[0]

    def _load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_name,  trust_remote_code=True)