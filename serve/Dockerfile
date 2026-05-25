FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV MODEL_ID="unsloth/Qwen2.5-Coder-7B-Instruct"
ENV ADAPTER_ID="JohanKira/om-code-v1"
ENV PORT=7860

EXPOSE 7860

CMD ["python", "server.py"]
