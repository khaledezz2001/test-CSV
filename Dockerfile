FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# System dependencies for OCR + PDFs
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ===============================
# DOWNLOAD QWEN 2.5 14B (LOGGED)
# ===============================
RUN python3 -u <<'EOF'
import sys
from huggingface_hub import snapshot_download

print("====================================", flush=True)
print("STARTING QWEN 2.5 14B DOWNLOAD", flush=True)
print("THIS MAY TAKE A WHILE", flush=True)
print("====================================", flush=True)

try:
    snapshot_download(
        repo_id="Qwen/Qwen2.5-14B-Instruct",
        local_dir="/models/qwen",
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1   # reduce RAM spikes
    )
except Exception as e:
    print("DOWNLOAD FAILED:", e, flush=True)
    sys.exit(1)

print("====================================", flush=True)
print("QWEN 14B DOWNLOAD FINISHED", flush=True)
print("====================================", flush=True)
EOF

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
