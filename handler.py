import runpod
import torch
import json
import base64
import io
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

log("Loading Qwen2-VL-7B...")

try:
    # Load Model
    # Note: We remove explicit flash_attn to support a wider range of GPUs (T4, V100, etc.)
    # If the user has Ampere+, it will still be fast.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load Processor
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
"""

def process_pdf(pdf_bytes):
    # 1. Convert PDF to Images
    try:
        # Convert first 5 pages to avoid OOM if PDF is huge
        images = convert_from_bytes(pdf_bytes, dpi=200) # 200 DPI is sufficient for Qwen2-VL
        log(f"Converted PDF to {len(images)} images.")
    except Exception as e:
        log(f"Error converting PDF: {e}")
        return json.dumps({"error": f"Failed to convert PDF: {str(e)}"})

    if not images:
        return json.dumps({"error": "No images extracted from PDF"})

    # 2. Prepare Messages for VLM
    content_blocks = []
    
    # Add all images
    for i, img in enumerate(images):
        # Resize if image is massive (>2000px) to save tokens/VRAM
        # Qwen2-VL handles variable resolution, but let's be safe
        if max(img.size) > 2000:
             img.thumbnail((2000, 2000))
        
        content_blocks.append({
            "type": "image",
            "image": img,
        })
    
    # Add the text prompt
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

    # 3. Preprocess Inputs
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
    except Exception as e:
        log(f"Error processing inputs: {e}")
        return json.dumps({"error": f"Input processing failed: {str(e)}"})

    # 4. Generate
    log("Running inference...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False, 
                temperature=0.1 
            )
    except Exception as e:
         log(f"Inference error: {e}")
         return json.dumps({"error": f"Inference failed: {str(e)}"})

    # 5. Decode
    try:
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    except Exception as e:
        log(f"Decoding error: {e}")
        return json.dumps({"error": f"Decoding failed: {str(e)}"})

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
    raw_output = process_pdf(pdf_bytes)
    log(f"Raw Output: {raw_output[:100]}...")

    # Attempt to parse JSON
    try:
        # Clean up any potential markdown fences
        cleaned_output = raw_output.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_output)
        return data
    except json.JSONDecodeError:
        log("Failed to parse JSON directly. Returning raw text.")
        return {"raw_output": raw_output}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
