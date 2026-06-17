---
title: Om-Think
emoji: 🧠
colorFrom: purple
colorTo: gold
sdk: docker
pinned: false
---

# Om-Think Inference API

Serves the Om-Think reasoning model (Qwen2.5-Coder-3B + LoRA adapter) as an OpenAI-compatible API.

Om-Think is the planning/reasoning engine used by the Om Agent orchestrator to produce structured task briefs before code generation begins.

- `GET /health` — status check
- `POST /v1/chat/completions` — generate reasoning output (streaming supported)
