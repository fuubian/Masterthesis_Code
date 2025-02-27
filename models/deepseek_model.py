from models.text_image_model import TextImageModel
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images

class DeepSeekModel(TextImageModel):
    def __init__(self, model_name="deepseek-ai/deepseek-vl2-tiny"):
        super().__init__(model_name)
        self._load_model()

    def generate_answer(self, question_prompt, image, max_answer_length=512):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n {question_prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_answer_length,
            do_sample=False,
            use_cache=True
        )

        model_response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        return model_response

    def _load_model(self):
        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
        self.tokenizer = vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()