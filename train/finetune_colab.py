"""
Om-Code Fine-Tuning Script
Run in Google Colab (T4 GPU) or Kaggle notebook.
Fine-tunes Qwen2.5-Coder-7B-Instruct with LoRA using Unsloth for 2-4x speedup.
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# !pip install datasets huggingface_hub

# ============================================================
# CELL 2: Imports and configuration
# ============================================================
import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
OUTPUT_DIR = "./om-code-lora"
HUB_MODEL_ID = "JohanKira/om-code-v1"

# ============================================================
# CELL 3: Login to HuggingFace (optional, for pushing model)
# ============================================================
# Uncomment and set your token to push to Hub
# login(token="hf_YOUR_TOKEN_HERE")

# ============================================================
# CELL 4: Load model with Unsloth
# ============================================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"Trainable parameters: {model.print_trainable_parameters()}")

# ============================================================
# CELL 5: Load and prepare dataset
# ============================================================
DATASET_PATH = "./training_data.jsonl"

# Upload training_data.jsonl to Colab first, or mount Google Drive
# from google.colab import files
# uploaded = files.upload()  # Upload training_data.jsonl

def load_dataset(path):
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example(example):
    """Convert messages to the model's chat template format."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


print("Loading dataset...")
raw_examples = load_dataset(DATASET_PATH)
print(f"Loaded {len(raw_examples)} examples")

dataset = Dataset.from_list(raw_examples)
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

print(f"Dataset ready: {len(dataset)} examples")
print(f"Sample (first 500 chars):\n{dataset[0]['text'][:500]}")

# ============================================================
# CELL 6: Training
# ============================================================
print("Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=42,
        report_to="none",
    ),
)

stats = trainer.train()
print(f"Training complete! Loss: {stats.training_loss:.4f}")

# ============================================================
# CELL 7: Save the LoRA adapter
# ============================================================
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved!")

# ============================================================
# CELL 8: Push to HuggingFace Hub (optional)
# ============================================================
# Uncomment to push:
# print(f"Pushing to {HUB_MODEL_ID}...")
# model.push_to_hub(HUB_MODEL_ID, token=os.environ.get("HF_TOKEN"))
# tokenizer.push_to_hub(HUB_MODEL_ID, token=os.environ.get("HF_TOKEN"))
# print("Pushed to Hub!")

# ============================================================
# CELL 9: Inference test
# ============================================================
print("\n" + "=" * 60)
print("INFERENCE TEST")
print("=" * 60)

FastLanguageModel.for_inference(model)

test_messages = [
    {
        "role": "system",
        "content": (
            "You are Om — The Divine Architect. A polymath AI that builds anything. "
            "You write complete, working, self-contained code. Your style: concise, no unnecessary comments, "
            "single-file HTML+CSS+JS when possible. You think like an architect, code like a craftsman, "
            "and design like an artist."
        ),
    },
    {
        "role": "user",
        "content": "Build a simple particle system with gravity that responds to mouse clicks",
    },
]

input_text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response[:2000])
print("\n... (truncated for display)")

# ============================================================
# CELL 10: Save as GGUF for local inference (optional)
# ============================================================
# Uncomment to export GGUF:
# model.save_pretrained_gguf(
#     "om-code-gguf",
#     tokenizer,
#     quantization_method="q4_k_m",
# )
# print("GGUF saved!")
