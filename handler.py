import runpod
import torch
import json
import base64
import io
import gc
import re
from datetime import datetime
from pdf2image import convert_from_bytes
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ===============================
# CONFIG & MODEL LOADING
# ===============================
MODEL_PATH = "/models/qwen"
BATCH_SIZE = 1  # Process 1 page at a time to save VRAM

log("Loading Qwen2-VL-7B...")

try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    log("Qwen2-VL loaded successfully.")
except Exception as e:
    log(f"CRITICAL ERROR loading model: {str(e)}")
    raise e

# ===============================
# PROMPT
# ===============================
SYSTEM_PROMPT = """You are a helpful financial assistant.
Your task is to extract all transaction details from the provided bank statement images.
Return ONLY a valid JSON array of objects. Do not include any markdown formatting (like ```json).

Output Format:
[
  {
    "date": "YYYY-MM-DD",
    "description": "DIVIDEND",
    "reference": "VN 1628315",
    "currency": "USD",
    "debit": null,
    "credit": 1495.80,
    "balance": 514894.75
  }
]

Rules:
1. Extract every single transaction row.
2. If a value is missing, use null.
3. Ensure numbers are floats (no currency symbols or thousand separators). Use absolute values (always positive).
4. Date format: ALWAYS output dates as YYYY-MM-DD. IMPORTANT: Bank statements often use DD/MM/YYYY format (day first, then month). For example, 01/07/2025 means July 1st 2025 (2025-07-01), NOT January 7th. Pay attention to the date range shown at the top of the statement to determine the correct format.
5. CAREFULLY check the column headers to determine whether an amount is a debit or credit:
   - If there are separate "Debits" and "Credits" columns, look at which column the number appears under.
   - If there is a single "Amount" column, negative values (with a minus sign) are DEBITS and positive values are CREDITS.
   - Fees, charges, and withdrawals are always DEBITS.
6. "reference" is the structured identifier found in the transaction details. It includes:
   - Voucher numbers (e.g. "VN 1628315")
   - Transaction codes (e.g. "U5777201", "DK0", "D34")
   - Txn Code / Transaction Code fields
   - Cheque numbers, transfer references
   - Any alphanumeric code that identifies the transaction
   Extract the reference SEPARATELY from the description. If no reference is found, use null.
7. "description" should contain the transaction type/name and any meaningful details (e.g. "IBU-Low Activity Fees For June 2025", "DIVIDEND", "INTEREST"). Do NOT include reference codes, branch codes, or account numbers in the description.
8. "currency" is the currency of the account as shown on the statement header or transaction details (e.g. USD, EUR, GBP, SAR, AED, CHF). Detect it from the statement context.
"""

def process_batch(images):
    # Prepare Messages for VLM
    content_blocks = []
    
    for img in images:
        if max(img.size) > 2000:
             img.thumbnail((2000, 2000))
        
        content_blocks.append({
            "type": "image",
            "image": img,
        })
    
    content_blocks.append({
        "type": "text",
        "text": SYSTEM_PROMPT
    })

    messages = [
        {
            "role": "user",
            "content": content_blocks
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048, # Reduce max tokens per batch slightly
                do_sample=False, 
                temperature=0.1 
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Cleanup VRAM explicitly after each batch
        del inputs, generated_ids, generated_ids_trimmed, image_inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return output_text
        
    except Exception as e:
        log(f"Batch processing error: {e}")
        return json.dumps({"error": f"Batch failed: {str(e)}"})


def process_pdf(pdf_bytes):
    # 1. Convert PDF to Images
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        log(f"Converted PDF to {len(images)} images.")
    except Exception as e:
        log(f"Error converting PDF: {e}")
        return json.dumps({"error": f"Failed to convert PDF: {str(e)}"})

    if not images:
        return json.dumps({"error": "No images extracted from PDF"})

    all_transactions = []
    
    # Process in batches
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        log(f"Processing batch {i//BATCH_SIZE + 1}/{(len(images)+BATCH_SIZE-1)//BATCH_SIZE}...")
        
        raw_output = process_batch(batch)
        
        # Parse each batch result
        try:
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            batch_data = json.loads(cleaned)
            if isinstance(batch_data, list):
                all_transactions.extend(batch_data)
            else:
                 log(f"Warning: Batch returned non-list JSON: {batch_data}")
        except json.JSONDecodeError:
            log(f"Failed to parse JSON for batch {i}. Skipping.")
            log(f"Raw output: {raw_output[:200]}...")
            
    # Filter out ghost transactions (balance=0, credit=null, debit=null)
    final_transactions = []
    for t in all_transactions:
        balance = t.get("balance")
        credit = t.get("credit")
        debit = t.get("debit")
        
        # Check if it's a ghost record
        if ((balance == 0 or balance == 0.0) and credit is None and debit is None) or (credit is None and debit is None):
            continue
            
        final_transactions.append(t)
    
    # ---- Post-processing: normalize dates (fix DD/MM vs MM/DD ambiguity) ----
    for t in final_transactions:
        date_str = t.get("date", "")
        if date_str:
            # Try to detect and fix swapped month/day
            try:
                parts = date_str.split("-")
                if len(parts) == 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    # If month > 12, it's likely day and month are swapped
                    if month > 12 and day <= 12:
                        log(f"Fixing swapped date: {date_str} -> {year}-{day:02d}-{month:02d}")
                        t["date"] = f"{year}-{day:02d}-{month:02d}"
            except (ValueError, IndexError):
                pass
    
    # ---- Post-processing: sort by date (chronological) for balance validation ----
    try:
        final_transactions.sort(key=lambda t: t.get("date", "0000-00-00"))
    except Exception:
        pass
    
    # ---- Post-processing: validate debit/credit using balance changes ----
    for i in range(1, len(final_transactions)):
        prev_balance = final_transactions[i - 1].get("balance")
        curr_balance = final_transactions[i].get("balance")
        credit = final_transactions[i].get("credit")
        debit = final_transactions[i].get("debit")
        
        if prev_balance is None or curr_balance is None:
            continue
        
        balance_diff = curr_balance - prev_balance  # positive = increase, negative = decrease
        
        # If balance decreased, the transaction should be a debit
        if balance_diff < 0:
            if credit is not None and debit is None:
                log(f"Correcting transaction {i}: credit -> debit (balance decreased by {abs(balance_diff):.2f})")
                final_transactions[i]["debit"] = credit
                final_transactions[i]["credit"] = None
        
        # If balance increased, the transaction should be a credit
        elif balance_diff > 0:
            if debit is not None and credit is None:
                log(f"Correcting transaction {i}: debit -> credit (balance increased by {balance_diff:.2f})")
                final_transactions[i]["credit"] = debit
                final_transactions[i]["debit"] = None
            
    return final_transactions

# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    log(f"Received event: {event.keys()}")
    if "input" not in event:
        return {"error": "No input provided"}
        
    job_input = event["input"]
    
    if "pdf_base64" not in job_input:
        return {"error": "Missing pdf_base64 field"}

    pdf_b64 = job_input["pdf_base64"]
    
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
    except Exception as e:
        return {"error": f"Invalid base64: {str(e)}"}

    # Run Inference
    final_data = process_pdf(pdf_bytes)

    return final_data

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
