"""
Om-Code Inference Server
FastAPI server with OpenAI-compatible chat completions endpoint.
Loads base Qwen2.5-Coder + Om-Code LoRA adapter and streams responses via SSE.
"""

import os
import json
import time
import uuid
import asyncio
from typing import Optional, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
from threading import Thread

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Qwen2.5-Coder-7B-Instruct")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "JohanKira/om-code-v1")
PORT = int(os.environ.get("PORT", "7860"))

app = FastAPI(title="Om-Code API", version="1.0.0")

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
    model: Optional[str] = "om-code-v1"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False


def load_model():
    global model, tokenizer

    print(f"Loading base model: {MODEL_ID}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if ADAPTER_ID and ADAPTER_ID != "none":
        print(f"Loading LoRA adapter: {ADAPTER_ID}")
        try:
            model = PeftModel.from_pretrained(model, ADAPTER_ID)
            print("Adapter loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load adapter ({e}). Using base model.")

    model.eval()
    print("Model ready for inference.")


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "adapter": ADAPTER_ID}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if request.stream:
        return EventSourceResponse(stream_response(generation_kwargs, request.model))
    else:
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
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(new_tokens),
                "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens)
            }
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
                    "finish_reason": None
                }]
            }
            yield {"data": json.dumps(chunk_data)}
            await asyncio.sleep(0)

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield {"data": json.dumps(final_chunk)}
    yield {"data": "[DONE]"}


def generate_with_torch(kwargs):
    with torch.no_grad():
        model.generate(**kwargs)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
