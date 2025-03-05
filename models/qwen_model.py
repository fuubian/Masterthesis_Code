from models.text_image_model import TextImageModel
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenModel(TextImageModel):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=128):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_answer_length)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            model_response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return model_response[0]

    def _load_model(self):
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_name, min_pixels = min_pixels, max_pixels=max_pixels)