"""
Om-Think Inference Server (CPU)
FastAPI server with OpenAI-compatible chat completions endpoint.
Loads Qwen2.5-Coder-3B-Instruct + Om-Think LoRA adapter on CPU.
"""

import os
import json
import time
import uuid
import asyncio
from typing import Optional, List
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-3B-Instruct")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "JohanKira/om-think-v1")
PORT = int(os.environ.get("PORT", "7860"))

app = FastAPI(title="Om-Think API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "om-think-v1"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


def load_model():
    global model, tokenizer

    print(f"Loading base model: {MODEL_ID} (CPU, float32)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    if ADAPTER_ID and ADAPTER_ID != "none":
        print(f"Loading LoRA adapter: {ADAPTER_ID}")
        try:
            model = PeftModel.from_pretrained(model, ADAPTER_ID)
            model = model.merge_and_unload()
            print("Adapter merged into base model.")
        except Exception as e:
            print(f"Warning: Could not load adapter ({e}). Using base model.")

    model.eval()
    print("Om-Think model ready for inference (CPU).")


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "adapter": ADAPTER_ID,
        "device": "cpu",
        "name": "Om-Think — The Divine Reasoning Engine",
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — still warming up")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": min(request.max_tokens, 2048),
        "temperature": max(request.temperature, 0.01),
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if request.stream:
        return StreamingResponse(
            stream_response(generation_kwargs, request.model),
            media_type="text/event-stream",
        )

    with torch.no_grad():
        output = model.generate(**generation_kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
    }


async def stream_response(generation_kwargs, model_name):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer

    thread = Thread(target=generate_with_torch, args=(generation_kwargs,))
    thread.start()

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    for text_chunk in streamer:
        if text_chunk:
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text_chunk},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0)

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def generate_with_torch(kwargs):
    with torch.no_grad():
        model.generate(**kwargs)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
