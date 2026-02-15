import runpod
import torch
import json
import base64
import io
import gc
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
    "description": "Transaction description",
    "reference": "TXN-123456",
    "currency": "USD",  # Detected from the statement (e.g. USD, EUR, GBP, etc.)
    "debit": 100.00,   # Use number for debit/withdrawal (positive value), or null
    "credit": null,    # Use number for credit/deposit, or null
    "balance": 5000.00 # Running balance if available
  }
]

Rules:
1. Extract every single transaction row.
2. If a value is missing, use null.
3. Keep descriptions exactly as they appear.
4. Ensure numbers are floats (no currency symbols).
5. Detect the year from the statement context if not explicitly in the row.
6. "reference" is the structured identifier assigned by the bank or the counterparty (e.g. cheque number, transfer reference, transaction ID). If not available, use null.
7. "currency" is the currency of the transaction as shown on the statement (e.g. USD, EUR, GBP, SAR, AED, etc.). Detect it from the statement context.
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
