import os
import io
from pathlib import Path
from dotenv import load_dotenv

# 1. SETTING UP ENVIRONMENT VARIABLES (Must happen before heavy ML imports)
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = BASE_DIR / "hf_cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR)

# 2. IMPORTING CORE MACHINE LEARNING & SERVER LIBRARIES
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL FROM HUGGING FACE ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "Trijoy/task2-document-classifier")

model = ViTForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

def get_vit_prediction(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Squeeze batch dimension to avoid indexing errors
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)

    predicted_class_idx = probs.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    confidence = probs[predicted_class_idx].item()

    return label, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        vit_label, vit_confidence = get_vit_prediction(image)

        # Enforce threshold fallback logic
        if vit_confidence < 0.85:
            final_label = "Invalid Document"
        else:
            final_label = vit_label

        # Clean payload completely removed of all OCR metrics
        return {
            "label": final_label,
            "confidence": f"{vit_confidence:.2%}"
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)