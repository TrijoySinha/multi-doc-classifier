import os
import io
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

LOCAL_CACHE_DIR = BASE_DIR / "hf_cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR)

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI(title="Document Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = BASE_DIR / "models" / "final_model"

print(f"--- Loading local model from: {MODEL_PATH} ---")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModelForImageClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

print("\n===== MODEL LABELS =====")
print(model.config.id2label)
print("========================\n")

model.to(device)
model.eval()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

DISPLAY_LABELS = {
    "Pan": "PAN Card",
    "pan": "PAN Card",
    "pan_data": "PAN Card",
    "aadhaar": "Aadhaar Card",
    "aadhaar_data": "Aadhaar Card",
    "dl": "Driving License",
    "dl_data": "Driving License",
    "passport": "Passport",
    "passport_data": "Passport",
    "voter": "Voter ID",
    "voter_data": "Voter ID",
}


def clean_label(raw_label: str) -> str:
    return DISPLAY_LABELS.get(raw_label, raw_label)


def predict_document(image: Image.Image):
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())

    raw_label = model.config.id2label[predicted_idx]
    label = clean_label(raw_label)

    top_probs, top_indices = torch.topk(probs, k=min(3, probs.shape[0]))

    top_predictions = []
    for prob, idx in zip(top_probs, top_indices):
        idx = int(idx.item())
        raw_top_label = model.config.id2label[idx]

        top_predictions.append({
            "label": clean_label(raw_top_label),
            "raw_label": raw_top_label,
            "confidence": round(float(prob.item()), 4),
        })

    return label, raw_label, confidence, top_predictions


@app.get("/")
def home():
    return {
        "message": "Document Classifier API is running",
        "model_path": str(MODEL_PATH),
        "device": str(device),
        "threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if file.content_type is None or not file.content_type.startswith("image/"):
            return {
                "error": "Only image files are supported. Upload JPG, PNG, WEBP, etc."
            }

        image_data = await file.read()

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError:
            return {
                "error": "Invalid image file. Please upload a valid JPG, PNG, or WEBP image."
            }

        label, raw_label, confidence, top_predictions = predict_document(image)

        if confidence < CONFIDENCE_THRESHOLD:
            final_label = "Invalid / Uncertain Document"
        else:
            final_label = label

        return {
            "label": final_label,
            "raw_label": raw_label,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence:.2%}",
            "threshold": CONFIDENCE_THRESHOLD,
            "top_predictions": top_predictions,
        }

    except Exception as e:
        return {
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )