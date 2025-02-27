# This script follows and uses code from the tutorials of NielsRogge and FourthBrainAI.
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb
# https://colab.research.google.com/drive/1LFcri1CHxNWXG6W4DnTXorUvRT_xY0kd?usp=sharing#scrollTo=nGTCqz8QPO5u

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.data_loader import DataLoader
from datasets import load_dataset
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Constants
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_LENGTH = 384

# Load training dataset
dataset = load_dataset("json", data_files=config.QA_TRAIN_SPLIT_PATH)
split_datasets = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_data = split_datasets["train"]
val_data = split_datasets["test"]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID,
                                                      quantization_config=quantization_config,
                                                      torch_dtype=torch.float16)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer = tokenizer

class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            question = example["question"]
            answer = example["answer"]
            image_path = DataLoader.get_image_path(example["image_id"])
            image = Image.open(image_path)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ],
                }
            ]
            text = self.processor.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(image)

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

data_collator = LLavaDataCollator(processor)

training_args = TrainingArguments(
    output_dir="llava-1.5-7b-hf-ft-mix-vsft",
    report_to="tensorboard",
    learning_rate=1.4e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    logging_steps=5,
    num_train_epochs=5,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    fp16=True,
    bf16=False
)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    dataset_text_field="question",
    tokenizer=tokenizer,
    data_collator=data_collator,
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer.train()
trainer.save_model("/models")
tokenizer.save_pretrained("/models")