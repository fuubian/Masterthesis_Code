# Following the instructions on https://huggingface.co/THUDM/glm-4v-9b

from models.text_image_model import TextImageModel
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class GLMModel(TextImageModel):
    def __init__(self, model_name="THUDM/glm-4v-9b"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=2500):
        image = Image.open(image).convert('RGB')
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": question_prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)
                                       
        inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in inputs.items()}
    
        gen_kwargs = {"max_length": max_answer_length, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            model_response = self.tokenizer.decode(outputs[0])

        return model_response.replace("<|endoftext|>", "")

    def _load_model(self):
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map='auto'
        ).eval()