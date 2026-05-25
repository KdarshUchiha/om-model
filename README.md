# Om-Code Model

Fine-tuned code generation model with Om's personality — builds games, apps, and tools in a concise, confident, self-contained style.

## Architecture

```
Base Model: Qwen2.5-Coder-7B-Instruct (4-bit quantized)
Method:     LoRA (r=64, alpha=128, all linear layers)
Training:   3 epochs, ~2 hours on free Colab T4
Output:     LoRA adapter (~100MB) → push to HuggingFace
```

## Quick Start

### Step 1: Generate training data

```bash
cd data/
python generate_dataset.py YOUR_GEMINI_API_KEY
```

This takes the 26 seed examples in `seed_examples.jsonl` and generates 500+ variations using Gemini. Output: `training_data.jsonl`

### Step 2: Fine-tune on Google Colab (free)

1. Upload `training_data.jsonl` to Google Drive or Colab
2. Open `train/finetune_colab.py` in Colab
3. Click Runtime → Run All
4. Wait ~2 hours
5. Adapter saves to `./om-code-lora/` and optionally pushes to HF Hub

### Step 3: Deploy (free on HF Spaces)

```bash
# Create a new HF Space with Docker SDK + GPU (T4)
# Push the serve/ directory:
cd serve/
# Copy your adapter into serve/om-code-lora/
git add . && git commit -m "Deploy Om-Code" && git push
```

Your model will be live at: `https://YOUR_USERNAME-om-code.hf.space`

### Step 4: Connect to Om Agent

In `om-agent-backend/agents/base.py`, add Om-Code as a third provider option pointing to your HF Space URL.

## File Structure

```
om-model/
├── README.md                    ← You are here
├── data/
│   ├── seed_examples.jsonl     ← 26 hand-crafted examples (games, apps, architecture, debug)
│   └── generate_dataset.py     ← Expands seeds to 500+ examples via Gemini API
├── train/
│   ├── finetune_colab.py       ← Complete Unsloth fine-tuning script for Colab
│   └── config.yaml             ← Hyperparameters
├── serve/
│   ├── server.py               ← FastAPI inference server (OpenAI-compatible)
│   ├── Dockerfile              ← HF Spaces deployment (GPU T4)
│   └── requirements.txt
└── .gitignore
```

## Om's Code Style (what the model learns)

- Concise — no unnecessary comments
- Self-contained — single HTML file with embedded CSS/JS
- Complete — no placeholders, no "TODO", no truncation
- Confident — direct, no filler words
- Architectural — clean structure, proper game loops, event handling
- Visual — always has polished CSS, dark themes, smooth transitions

## Training Data Coverage

| Category | Examples | What it teaches |
|---|---|---|
| Games | 10 | Snake, Pong, Tetris, Memory, 2048, Simon, Runner, etc. |
| Web Apps | 8 | Todo, Timer, Editor, Calculator, Kanban, Tracker, etc. |
| Architecture | 4 | System design, API design, patterns with code |
| Debugging | 2 | Identify bugs, explain in 1 line, fix |
| Design | 2 | CSS components, responsive patterns |

## Model Card

- **Base**: Qwen2.5-Coder-7B-Instruct
- **Method**: QLoRA (4-bit + LoRA)
- **Training tokens**: ~500K
- **License**: Apache 2.0 (same as base model)
- **Use case**: Code generation, game building, web app creation
- **Limitations**: 7B model — good at single-file projects, less reliable for complex multi-file architectures
