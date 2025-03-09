# This script follows the tutorial by Merve Noyan: https://github.com/merveenoyan/smol-vision/blob/main/Fine_tune_PaliGemma.ipynb

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.data_loader import DataLoader
from utils.token_loader import TokenLoader
from datasets import load_dataset
from PIL import Image
from huggingface_hub import login
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor, TrainingArguments, Trainer
from torchvision.io import read_image
from peft import get_peft_model, LoraConfig
import torch

# Login to HuggingFace
access_token = TokenLoader.load_token_huggingface()
login(access_token)

# Constants
MODEL_ID = "google/paligemma2-3b-pt-448"
device = "cuda"

# Load training dataset
dataset = load_dataset("json", data_files=config.QA_TRAIN_SPLIT_PATH)
split_datasets = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_data = split_datasets["train"]
val_data = split_datasets["test"]

# Load model and freeze layers
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

# Load model again for quantization and lora
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto", quantization_config=bnb_config, attn_implementation='eager')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
DTYPE = model.dtype

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
def collate_fn(examples):
    texts = ["<image> " + example["question"] for example in examples]
    labels= [example['answer'] for example in examples]
    images = [
        Image.open(DataLoader.get_image_path(example["image_id"]))
        .convert("RGB")
        .resize((448, 448))  # Reduce resolution
        for example in examples
    ]
    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
    
    tokens = tokens.to(DTYPE).to(device)
    return tokens

# Set training arguments
args=TrainingArguments(
            num_train_epochs=1,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=2,
            learning_rate=3e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            output_dir="paligemma_vqav2",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )

trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
        args=args
        )

# Start training
trainer.train()

# Push to HuggingFace
trainer.push_to_hub(repo_name="fuubian/trained_paligemma")

# Save locally
model.save_pretrained("training_scripts")
processor.save_pretrained("training_scripts")