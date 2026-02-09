import runpod
import torch
import re
import json
import base64
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================
# RTX 4090 OPTIMIZATION
# ======================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ======================================================
# OCR
# ======================================================
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec=False
)

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
    rows = []
    current = []

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

# ======================================================
# LLM (Qwen 2.5 14B)
# ======================================================
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

SYSTEM_PROMPT = """
You extract bank transactions from OCR text.

Return ONLY a JSON array.
Each object MUST be exactly:

{
  "date": "YYYY-MM-DD",
  "description": string,
  "debit": number | null,
  "credit": number | null,
  "balance": number | null
}

Rules:
- Negative amounts → debit
- Positive amounts → credit
- Use absolute values
- Convert commas to decimal points
- Do not invent data
- Do not explain anything
"""

def llm_extract(rows):
    text = "\n".join(rows)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
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
            max_new_tokens=2000,
            do_sample=False,
            use_cache=True
        )

    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return decoded

# ======================================================
# VALIDATION & NORMALIZATION
# ======================================================
def normalize_amount(x):
    if x is None:
        return None
    return abs(float(x))

def validate(tx):
    if not re.match(r"\d{4}-\d{2}-\d{2}", tx.get("date", "")):
        return False
    if tx.get("debit") is None and tx.get("credit") is None:
        return False
    return True

# ======================================================
# RUNPOD HANDLER
# ======================================================
def handler(event):
    pdf_b64 = event["input"]["pdf_base64"]
    pdf_bytes = base64.b64decode(pdf_b64)

    lines = pdf_to_lines(pdf_bytes)
    rows = build_rows(lines)

    raw = llm_extract(rows)

    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "error": "LLM output is not valid JSON",
            "raw_output": raw
        }

    clean = []
    for tx in parsed:
        tx["debit"] = normalize_amount(tx.get("debit"))
        tx["credit"] = normalize_amount(tx.get("credit"))
        if validate(tx):
            clean.append(tx)

    return clean

runpod.serverless.start({"handler": handler})
