import runpod
import json
import base64
import re
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datetime import datetime
from pdf2image import convert_from_bytes
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ===============================
# CONFIG & MODEL LOADING
# ===============================
MODEL_PATH = "/models/gemma4"
MAX_PAGES_PER_BATCH = 4  # Number of pages to process in a single prompt (multi-image)
MAX_NEW_TOKENS = 65536  # Large enough for 50+ transactions per page

log("Loading Gemma-4-26B-A4B-it with Transformers...")

processor = AutoProcessor.from_pretrained(MODEL_PATH)

try:
    # Auto-detect best experts implementation based on compute capability (H100=90, A100=80)
    # batched_mm duplicates weights per token and causes CUDA OOM with large image inputs.
    capability = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 8
    experts_impl = "grouped_mm" if capability >= 9 else "eager"

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        experts_implementation=experts_impl,
    )
    model.eval()
    log("Gemma-4-26B-A4B-it loaded successfully with Transformers.")
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
"""

def repair_truncated_json(text):
    """Attempt to repair truncated JSON arrays by finding the last complete object.
    
    Strategy: walk backwards through all '}' positions to find the last
    closing brace that, when followed by ']', yields a valid JSON array.
    This handles cases where the truncation happens mid-object.
    """
    start = text.find('[')
    if start == -1:
        return None
    
    # Find all '}' positions and try each from the end
    body = text[start:]
    positions = [i for i, ch in enumerate(body) if ch == '}']
    
    for pos in reversed(positions):
        candidate = body[:pos + 1].rstrip().rstrip(',') + '\n]'
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and len(data) > 0:
                log(f"Repaired truncated JSON: recovered {len(data)} transactions.")
                return data
        except json.JSONDecodeError:
            continue
    
    return None


def process_pages(images):
    """
    Process multiple pages using Gemma 4's native multi-image support
    with Hugging Face Transformers (official API from model card).
    """
    # Resize images for consistency
    processed_images = []
    for img in images:
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000))
        processed_images.append(img)

    # Build multi-image prompt — use image placeholders in content,
    # then pass the actual PIL images separately to the processor.
    content = []
    for idx in range(len(processed_images)):
        # Each image placeholder tells the template where to insert an image token
        content.append({"type": "image"})
    content.append({"type": "text", "text": SYSTEM_PROMPT})

    messages = [
        {"role": "user", "content": content}
    ]

    try:
        # Step 1: Apply chat template to get the formatted text prompt
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Step 2: Process text + images together into model inputs
        inputs = processor(
            text=text,
            images=processed_images,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Step 3: Generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # Step 4: Decode only the generated tokens (skip the input)
        generated_ids = output_ids[:, input_len:]
        raw_response = processor.decode(generated_ids[0], skip_special_tokens=True)

        log(f"Raw model output (first 500 chars): {raw_response[:500]}")
        return [raw_response]

    except Exception as e:
        log(f"Inference error: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return [json.dumps({"error": f"Batch failed: {str(e)}"})]


def parse_raw_output(raw_output, batch_idx):
    """Parse raw model output into transaction list.
    
    Returns:
        tuple: (transactions_list, was_truncated)
    """
    was_truncated = False
    try:
        cleaned = raw_output

        # Strip markdown code fences
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Try direct JSON parse first
        batch_data = None
        try:
            batch_data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback 1: extract JSON array using regex
            json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if json_match:
                try:
                    batch_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback 2: repair truncated JSON (model ran out of tokens)
            if batch_data is None:
                log("Direct parse failed. Attempting truncated JSON repair...")
                batch_data = repair_truncated_json(cleaned)
                if batch_data is not None:
                    was_truncated = True

        if batch_data is not None and isinstance(batch_data, list):
            log(f"Batch {batch_idx} parsed successfully: {len(batch_data)} transactions.")
            return batch_data, was_truncated
        elif batch_data is not None:
            log(f"Warning: Batch {batch_idx} returned non-list JSON: {batch_data}")
        else:
            log(f"Failed to parse JSON for batch {batch_idx}. Skipping.")
            log(f"Raw output (first 500 chars): {raw_output[:500]}")
            log(f"Cleaned text (first 300 chars): {cleaned[:300]}")
    except Exception as e:
        log(f"Failed to parse JSON for batch {batch_idx}: {e}. Skipping.")
        log(f"Raw output (first 500 chars): {raw_output[:500]}")
    
    return [], was_truncated


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
    
    # Process in batches — Gemma 4 handles multi-image natively
    for i in range(0, len(images), MAX_PAGES_PER_BATCH):
        batch = images[i:i + MAX_PAGES_PER_BATCH]
        batch_num = i // MAX_PAGES_PER_BATCH + 1
        total_batches = (len(images) + MAX_PAGES_PER_BATCH - 1) // MAX_PAGES_PER_BATCH
        log(f"Processing batch {batch_num}/{total_batches} ({len(batch)} pages as multi-image prompt)...")
        
        # Process all pages in the batch as a single multi-image prompt
        raw_outputs = process_pages(batch)
        
        # Parse the combined output
        batch_truncated = False
        for j, raw_output in enumerate(raw_outputs):
            batch_transactions, was_truncated = parse_raw_output(raw_output, batch_num)
            if was_truncated:
                batch_truncated = True
            all_transactions.extend(batch_transactions)
        
        # If output was truncated and batch had multiple pages, retry each page individually
        if batch_truncated and len(batch) > 1:
            log(f"Batch {batch_num} was truncated with {len(batch)} pages. Retrying each page individually...")
            all_transactions = all_transactions[:-len(batch_transactions)]  # remove partial results
            for page_idx, single_page in enumerate(batch):
                log(f"  Re-processing page {i + page_idx + 1} individually...")
                page_outputs = process_pages([single_page])
                for raw_output in page_outputs:
                    page_txns, _ = parse_raw_output(raw_output, f"{batch_num}-page{page_idx+1}")
                    all_transactions.extend(page_txns)
            
    # Filter out ghost transactions — only remove truly empty records
    final_transactions = []
    for t in all_transactions:
        balance = t.get("balance")
        credit = t.get("credit")
        debit = t.get("debit")
        description = t.get("description", "").strip()
        
        # Only remove if ALL value fields are empty/zero AND no meaningful description
        is_completely_empty = (
            (balance is None or balance == 0 or balance == 0.0)
            and credit is None
            and debit is None
        )
        # Also remove if both debit and credit are explicitly zero
        both_zero = (credit == 0 and debit == 0)
        
        if (is_completely_empty and not description) or both_zero:
            log(f"Filtered ghost transaction: {t}")
            continue
        
        # Keep only the 6 required fields
        cleaned_t = {
            "date": t.get("date", ""),
            "description": t.get("description", ""),
            "debit": t.get("debit"),
            "credit": t.get("credit"),
            "balance": t.get("balance"),
            "currency": t.get("currency", "")
        }
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
        
        balance_diff = curr_balance - prev_balance
        
        if balance_diff < 0:
            if credit is not None and debit is None:
                log(f"Correcting transaction {i}: credit -> debit (balance decreased by {abs(balance_diff):.2f})")
                final_transactions[i]["debit"] = credit
                final_transactions[i]["credit"] = None
        
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
        log("ERROR: No 'input' key in event")
        return {"error": "No input provided"}
        
    job_input = event["input"]
    log(f"Input keys: {job_input.keys() if isinstance(job_input, dict) else type(job_input)}")
    
    # Accept either 'pdf_base64' or 'file' as the input key
    pdf_b64 = job_input.get("pdf_base64") or job_input.get("file")
    
    if not pdf_b64:
        log("ERROR: Missing 'pdf_base64' or 'file' in input")
        return {"error": "Missing pdf_base64 or file field"}

    log(f"Received PDF data of length: {len(pdf_b64)}")
    
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        log(f"Decoded PDF: {len(pdf_bytes)} bytes")
    except Exception as e:
        log(f"ERROR: Invalid base64: {str(e)}")
        return {"error": f"Invalid base64: {str(e)}"}

    # Run Inference
    try:
        final_data = process_pdf(pdf_bytes)
        log(f"Processing complete. Transactions found: {len(final_data) if isinstance(final_data, list) else 'N/A'}")
        return final_data
    except Exception as e:
        log(f"ERROR during process_pdf: {str(e)}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Processing failed: {str(e)}"}

if __name__ == "__main__":
    # Log GPU status at startup
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        log("WARNING: CUDA is NOT available! Model will run on CPU (very slow).")
    runpod.serverless.start({"handler": handler})
