import runpod
import torch
import json
import base64
import io
import gc
import re
from datetime import datetime
from pdf2image import convert_from_bytes
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ===============================
# CONFIG & MODEL LOADING
# ===============================
MODEL_PATH = "/models/qwen"
BATCH_SIZE = 1  # Process 1 page at a time to save VRAM

log("Loading Qwen3-VL-8B-Instruct...")

try:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    log("Qwen3-VL loaded successfully.")
except Exception as e:
    log(f"CRITICAL ERROR loading model: {str(e)}")
    raise e

# ===============================
# PROMPT
# ===============================
SYSTEM_PROMPT = """You are a helpful financial assistant.
Your task is to extract all transaction details from the provided bank statement images.
Return ONLY a valid JSON array of objects. Do not include any markdown formatting (like ```json).

Output Format Examples:

Example 1 (separate Debits/Credits columns):
[
  {
    "date": "2014-05-15",
    "description": "DIVIDEND",
    "debit": null,
    "credit": 1495.80,
    "balance": 514894.75,
    "currency": "USD"
  }
]

Example 2 (single Amount column with negative values):
[
  {
    "date": "2025-07-01",
    "description": "IBU-Low Activity Fees For June 2025",
    "debit": 23.46,
    "credit": null,
    "balance": 5105.29,
    "currency": "USD"
  }
]

Rules:
1. Extract every single transaction row.
2. If a value is missing, use null.
3. Ensure numbers are floats (no currency symbols or thousand separators). Use absolute values (always positive).
4. Date format: ALWAYS output dates as YYYY-MM-DD. IMPORTANT: Bank statements almost always use DD/MM/YYYY format (day first, then month). For example, 02/06/2025 means June 2nd 2025 (2025-06-02), NOT February 6th. Even if both day and month values are 12 or below, assume DD/MM/YYYY. Use the statement's date range header and the description context (e.g. "Fees For May" posted in June) to confirm.
5. CAREFULLY check the column headers to determine whether an amount is a debit or credit:
   - If there are separate "Debits" and "Credits" columns, look at which column the number appears under.
   - If there is a single "Amount" column, negative values (with a minus sign) are DEBITS and positive values are CREDITS.
   - Fees, charges, and withdrawals are always DEBITS.
6. "description" should contain the transaction type/name and any meaningful details (including any reference codes, voucher numbers, or transaction IDs found in the row).
7. "currency" is the currency of the account as shown on the statement header or transaction details (e.g. USD, EUR, GBP, SAR, AED, CHF). Detect it from the statement context.
8. Output ONLY these 6 fields per transaction: date, description, debit, credit, balance, currency. Do NOT include any other fields.
9. CRITICAL WARNING: Do NOT extract bank account numbers, IBANs, BIKs, or long IDs (e.g. 40702810...) as a `debit` or `credit` amount. `debit` and `credit` must ONLY be the actual transaction or transfer amounts found in the amount columns (e.g., "Сумма по дебету", "Сумма по кредиту", "Debit", "Credit", "Amount" - typically values like 100.50, 4700.08). Long strings of digits are NEVER amounts.
10. This statement may have TWO amount columns: "Money out" (debits) and "Money in" (credits).
    A value appearing under "Money out" is ALWAYS a debit. Under "Money in" is ALWAYS a credit.
    Never swap them.
"""

def repair_truncated_json(text):
    """Attempt to repair truncated JSON arrays by finding the last complete object."""
    start = text.find('[')
    if start == -1:
        return None
    
    last_brace = text.rfind('}')
    if last_brace == -1:
        return None
    
    truncated = text[start:last_brace + 1].rstrip().rstrip(',')
    repaired = truncated + '\n]'
    
    try:
        data = json.loads(repaired)
        if isinstance(data, list):
            log(f"Repaired truncated JSON: recovered {len(data)} transactions.")
            return data
    except json.JSONDecodeError:
        pass
    
    return None


def process_batch(images):
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
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        del inputs, generated_ids, generated_ids_trimmed
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
        
        try:
            cleaned = raw_output

            # Strip Qwen3 <think>...</think> block if present
            think_match = re.search(r'</think>\s*', cleaned)
            if think_match:
                cleaned = cleaned[think_match.end():]

            # Strip markdown code fences
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            batch_data = None
            try:
                batch_data = json.loads(cleaned)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                if json_match:
                    try:
                        batch_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                if batch_data is None:
                    log("Direct parse failed. Attempting truncated JSON repair...")
                    batch_data = repair_truncated_json(cleaned)

            if batch_data is not None and isinstance(batch_data, list):
                all_transactions.extend(batch_data)
                log(f"Batch {i} parsed successfully: {len(batch_data)} transactions.")
            elif batch_data is not None:
                log(f"Warning: Batch returned non-list JSON: {batch_data}")
            else:
                log(f"Failed to parse JSON for batch {i}. Skipping.")
                log(f"Raw output (first 500 chars): {raw_output[:500]}")
                log(f"Cleaned text (first 300 chars): {cleaned[:300]}")
        except Exception as e:
            log(f"Failed to parse JSON for batch {i}: {e}. Skipping.")
            log(f"Raw output (first 500 chars): {raw_output[:500]}")
            
    # Filter out ghost transactions
    final_transactions = []
    for t in all_transactions:
        balance = t.get("balance")
        credit = t.get("credit")
        debit = t.get("debit")
        
        if ((balance == 0 or balance == 0.0) and credit is None and debit is None) or (credit is None and debit is None) or (credit == 0 and debit == 0):
            continue
        
        cleaned_t = {
            "date": t.get("date", ""),
            "description": t.get("description", ""),
            "debit": t.get("debit"),
            "credit": t.get("credit"),
            "balance": t.get("balance"),
            "currency": t.get("currency", "")
        }
        
        if cleaned_t["debit"] is not None and cleaned_t["debit"] > 1e12:
            log(f"Warning: Abnormally large debit ({cleaned_t['debit']}). Likely an account number. Nullifying.")
            cleaned_t["debit"] = None
        if cleaned_t["credit"] is not None and cleaned_t["credit"] > 1e12:
            log(f"Warning: Abnormally large credit ({cleaned_t['credit']}). Likely an account number. Nullifying.")
            cleaned_t["credit"] = None
            
        final_transactions.append(cleaned_t)
    
    # ---- Post-processing: normalize dates (fix DD/MM vs MM/DD ambiguity) ----
    for t in final_transactions:
        date_str = t.get("date", "")
        if date_str:
            try:
                parts = date_str.split("-")
                if len(parts) == 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    if month > 12 and day <= 12:
                        log(f"Fixing swapped date: {date_str} -> {year}-{day:02d}-{month:02d}")
                        t["date"] = f"{year}-{day:02d}-{month:02d}"
            except (ValueError, IndexError):
                pass
    
    # ---- Post-processing: sort chronologically, preserving PDF order as tiebreaker ----
    # The PDF is printed newest-first, so the raw extracted order is newest-first.
    # We tag each transaction with its original index, then use -index as tiebreaker
    # so that same-date transactions are reversed back into correct chronological order.
    for idx, t in enumerate(final_transactions):
        t["_original_index"] = idx

    try:
        final_transactions.sort(
            key=lambda t: (t.get("date", "0000-00-00"), -t["_original_index"])
        )
    except Exception:
        pass

    for t in final_transactions:
        t.pop("_original_index", None)

    # ---- Post-processing: validate debit/credit using balance changes ----
    for i in range(1, len(final_transactions)):
        prev_balance = final_transactions[i - 1].get("balance")
        curr_balance = final_transactions[i].get("balance")
        credit = final_transactions[i].get("credit")
        debit = final_transactions[i].get("debit")
        
        if prev_balance is None or curr_balance is None:
            continue
        
        balance_diff = round(curr_balance - prev_balance, 2)
        amount = debit if debit is not None else credit

        if balance_diff < 0:
            # Balance decreased → must be a debit
            expected_debit = round(abs(balance_diff), 2)
            if credit is not None and debit is None:
                # Amount is in the wrong field (credit), move it to debit
                if amount is not None and round(abs(amount - expected_debit), 2) < 0.02:
                    log(f"Correcting tx {i} [{final_transactions[i]['date']}]: moving credit {credit} → debit (balance decreased by {expected_debit})")
                    final_transactions[i]["debit"] = expected_debit
                    final_transactions[i]["credit"] = None
                elif amount is None:
                    log(f"Correcting tx {i} [{final_transactions[i]['date']}]: setting missing debit to {expected_debit}")
                    final_transactions[i]["debit"] = expected_debit
                    final_transactions[i]["credit"] = None
            elif debit is not None and round(abs(debit - expected_debit), 2) >= 0.02:
                # Debit value is wrong, correct it
                log(f"Correcting tx {i} [{final_transactions[i]['date']}]: debit {debit} → {expected_debit}")
                final_transactions[i]["debit"] = expected_debit

        elif balance_diff > 0:
            # Balance increased → must be a credit
            expected_credit = round(balance_diff, 2)
            if debit is not None and credit is None:
                # Amount is in the wrong field (debit), move it to credit
                if amount is not None and round(abs(amount - expected_credit), 2) < 0.02:
                    log(f"Correcting tx {i} [{final_transactions[i]['date']}]: moving debit {debit} → credit (balance increased by {expected_credit})")
                    final_transactions[i]["credit"] = expected_credit
                    final_transactions[i]["debit"] = None
                elif amount is None:
                    log(f"Correcting tx {i} [{final_transactions[i]['date']}]: setting missing credit to {expected_credit}")
                    final_transactions[i]["credit"] = expected_credit
                    final_transactions[i]["debit"] = None
            elif credit is not None and round(abs(credit - expected_credit), 2) >= 0.02:
                # Credit value is wrong, correct it
                log(f"Correcting tx {i} [{final_transactions[i]['date']}]: credit {credit} → {expected_credit}")
                final_transactions[i]["credit"] = expected_credit

        # balance_diff == 0: reversal pair or zero-amount entry, don't touch it

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

    final_data = process_pdf(pdf_bytes)

    return final_data

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
