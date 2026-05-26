"""
Om-Think Fine-Tuning Script
Run in Google Colab (T4 GPU).
Fine-tunes Qwen2.5-3B-Instruct with LoRA to create Om-Think — the reasoning engine.

INSTRUCTIONS:
1. Open Google Colab → Runtime → Change runtime type → T4 GPU
2. Copy ALL of the cells below into Colab
3. Run them in order
4. Wait ~20-30 minutes for training
5. Model pushes to HuggingFace automatically
"""

# ============================================================
# CELL 1: Install dependencies + download data
# Copy this entire block into one Colab cell and run it.
# ============================================================
"""
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install datasets huggingface_hub
!wget -q https://raw.githubusercontent.com/KdarshUchiha/om-model/main/data/think/think_training_data.jsonl -O training_data.jsonl
!wc -l training_data.jsonl
"""

# ============================================================
# CELL 2: Load model, train, and push (ALL IN ONE CELL)
# Copy this entire block into one Colab cell and run it.
# After runtime restart, run ONLY this cell (not cell 1 again).
# ============================================================
"""
import json, torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

# --- CONFIG ---
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # Replace with your HuggingFace token
MODEL_NAME = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
OUTPUT_HUB = "JohanKira/om-think-v1"
MAX_SEQ = 1024
LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 5
LR = 2e-4
# --------------

login(token=HF_TOKEN)

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME, max_seq_length=MAX_SEQ, load_in_4bit=True)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(model, r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", use_gradient_checkpointing="unsloth")

print("Loading dataset...")
raw = [json.loads(l) for l in open("training_data.jsonl") if l.strip()]
print(f"Loaded {len(raw)} examples")

def fmt(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

dataset = Dataset.from_list(raw).map(fmt, remove_columns=["messages"])
print(f"Dataset ready: {len(dataset)} examples")

print("Starting training...")
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=MAX_SEQ, packing=False,
    args=TrainingArguments(
        output_dir="./om-think-lora",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        warmup_steps=5,
        optim="adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        report_to="none",
    ),
)

trainer.train()
print("✅ Training complete!")

print(f"Pushing to {OUTPUT_HUB}...")
model.push_to_hub(OUTPUT_HUB)
tokenizer.push_to_hub(OUTPUT_HUB)
print(f"✅ Model pushed to https://huggingface.co/{OUTPUT_HUB}")
"""

# ============================================================
# CELL 3 (OPTIONAL): Test the model
# Run after training to verify it works.
# ============================================================
"""
FastLanguageModel.for_inference(model)

test_messages = [
    {"role": "system", "content": "You are Om-Think — The Divine Reasoning Engine. Given a user's request, produce a comprehensive structured plan BEFORE any code is written. Think deeply about WHY, HOW, TRADEOFFS, EDGE CASES, and DECOMPOSITION."},
    {"role": "user", "content": "I want to build a real-time chat app. Walk me through your thinking."},
]

input_text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)

response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=" * 60)
print("OM-THINK RESPONSE:")
print("=" * 60)
print(response[:3000])
"""
