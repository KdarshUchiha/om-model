FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct
ENV ADAPTER_ID=JohanKira/om-think-v1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "server.py"]
