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
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec=False)
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

def pdf_to_lines(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    lines = []

    for page_idx, img in enumerate(images):
        img_np = np.array(img)

        result = ocr.ocr(img_np, cls=True)

        # 🔑 SAFETY: skip empty / failed OCR pages
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
   - If the description contains words like:
     "Fee", "Fees", "Charge", "Maintenance", "Commission"
     then debit MUST be used and credit MUST be null.
   - Do NOT infer credit unless the text explicitly indicates incoming funds.

4. Use absolute numeric values only.

5. Description normalization:
   - Remove any month or year references from descriptions
   - Remove billing periods like "For June 2025", "December 2022"
   - Keep only the transaction name.

6. Date:
   - Use the transaction date column ONLY.
   - Do not infer or override dates.

7. Balance:
   - Use the balance shown for that row.
   - Do not calculate balances.

Return ONLY the JSON array.
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
