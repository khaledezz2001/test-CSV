import runpod
import torch
import re
import json
import base64
import numpy as np
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
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec=False, show_log=False)
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

def pdf_to_lines(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    lines = []

    for img in images:
        img_np = np.array(img)
        result = ocr.ocr(img_np, cls=True)

        if not result:
            continue

        for block in result:
            if not block:
                continue

            for line in block:
                if not line or len(line) < 2:
                    continue

                txt = line[1][0]
                if txt:
                    lines.append(txt.strip())

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

# ===============================
# DESCRIPTION NORMALIZATION
# ===============================
def normalize_description(desc):
    desc = re.sub(r"\bfor\s+[A-Za-z]+\s+\d{4}\b", "", desc, flags=re.I)
    desc = re.sub(r"\b[A-Za-z]+\s+\d{4}\b", "", desc)
    return " ".join(desc.split()).strip()

FEE_KEYWORDS = ["fee", "fees", "maintenance", "charge", "commission"]

SYSTEM_PROMPT = """
You are a financial transaction extraction engine.

Return ONLY a valid JSON array.
DO NOT use Markdown.
DO NOT wrap the output in code blocks.

Schema:
{
  "date": "YYYY-MM-DD",
  "description": string,
  "debit": number | null,
  "credit": number | null,
  "balance": number | null
}

STRICT RULES (NO EXCEPTIONS):

1. Preserve transaction order EXACTLY as it appears in the statement.
   NEVER reorder transactions.

2. Ignore rows that are not real transactions:
   - Balance brought forward
   - Opening / closing balance
   - Totals

3. Debit / credit determination:
   - Fees, charges, commissions, maintenance costs → ALWAYS debit
   - Do NOT infer credit unless explicitly incoming funds.

4. Use absolute numeric values only.

5. Description normalization:
   - Remove billing periods (months / years)
   - Keep only transaction name.

6. Use transaction date column ONLY.

7. Do NOT calculate balances.
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
        parsed = json.loads(raw)
    except Exception:
        return {"raw_output": raw}

    # ===============================
    # POST-PROCESSING (DETERMINISTIC)
    # ===============================

    for tx in parsed:
        # Normalize description
        tx["description"] = normalize_description(tx["description"])

        # Safety net: fees are always debit
        desc = tx["description"].lower()
        if any(k in desc for k in FEE_KEYWORDS):
            if tx["debit"] is None and tx["credit"] is not None:
                tx["debit"] = tx["credit"]
                tx["credit"] = None

    # Force chronological order
    parsed.sort(key=lambda x: x["date"])

    return parsed

runpod.serverless.start({"handler": handler})
