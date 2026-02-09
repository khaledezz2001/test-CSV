import runpod
import torch
import re
import json
import base64
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM

def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ===============================
# CUDA / RTX 4090
# ===============================
log("Starting worker")
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ===============================
# LOAD QWEN 2.5 7B
# ===============================
tokenizer = AutoTokenizer.from_pretrained(
    "/models/qwen",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "/models/qwen",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
log("Qwen 7B loaded")

# ===============================
# OCR
# ===============================
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec=False)
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

def pdf_to_lines(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    lines = []
    for img in images:
        result = ocr.ocr(img, cls=True)
        for block in result:
            for line in block:
                txt = line[1][0].strip()
                if txt:
                    lines.append(txt)
    return lines

def build_rows(lines):
    rows, current = [], []
    for line in lines:
        if DATE_RE.search(line):
            if current:
                rows.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        rows.append(" ".join(current))
    return rows

SYSTEM_PROMPT = """
Return ONLY a JSON array of transactions.

Format:
{
  "date": "YYYY-MM-DD",
  "description": string,
  "debit": number | null,
  "credit": number | null,
  "balance": number | null
}

Rules:
- Negative → debit
- Positive → credit
- Absolute values
- No explanations
"""

def llm_extract(rows):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(rows)}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    pdf_b64 = event["input"]["pdf_base64"]
    pdf_bytes = base64.b64decode(pdf_b64)

    lines = pdf_to_lines(pdf_bytes)
    rows = build_rows(lines)

    raw = llm_extract(rows)

    try:
        return json.loads(raw)
    except Exception:
        return {"raw_output": raw}

runpod.serverless.start({"handler": handler})
