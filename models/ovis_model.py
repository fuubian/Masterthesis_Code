from models.text_image_model import TextImageModel
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

class OvisModel(TextImageModel):
    def __init__(self, model_name="AIDC-AI/Ovis2-8B"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=1024):
        # Set query
        images = [Image.open(image)]
        max_partition = 9
        query = f'<image>\n{question_prompt}'

        # Format conversation
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        pixel_values = [pixel_values]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=max_answer_length,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            model_response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

        return model_response

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                             torch_dtype=torch.bfloat16,
                                             llm_attn_implementation='eager',
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()