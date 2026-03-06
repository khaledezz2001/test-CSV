FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# System deps for OCR + PDFs
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /requirements.txt
RUN pip install flash-attn --no-build-isolation

# ===============================
# DOWNLOAD QWEN3-VL-30B-A3B-Instruct
# ===============================
RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download

print("Downloading Qwen3-VL-30B-A3B-Instruct...", flush=True)

snapshot_download(
    repo_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
    local_dir="/models/qwen",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Qwen3-VL-30B-A3B download complete", flush=True)
EOF

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
