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
from fastapi import FastAPI, UploadFile, File, HTTPException, status
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

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")

if not HF_MODEL_REPO:
    raise ValueError("HF_MODEL_REPO is missing. Add it to your .env file.")

print(f"--- Loading model from Hugging Face Hub: {HF_MODEL_REPO} ---")
print(f"--- Hugging Face cache: {LOCAL_CACHE_DIR} ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(
    HF_MODEL_REPO,
    cache_dir=str(LOCAL_CACHE_DIR)
)

model = AutoModelForImageClassification.from_pretrained(
    HF_MODEL_REPO,
    cache_dir=str(LOCAL_CACHE_DIR)
)

print("\n===== MODEL LABELS =====")
print(model.config.id2label)
print("========================\n")

model.to(device)
model.eval()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))


def predict_document(image: Image.Image):
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())

    label = model.config.id2label[predicted_idx]

    top_probs, top_indices = torch.topk(probs, k=min(3, probs.shape[0]))

    top_predictions = []
    for prob, idx in zip(top_probs, top_indices):
        idx = int(idx.item())
        top_predictions.append({
            "label": model.config.id2label[idx],
            "confidence": round(float(prob.item()), 4),
        })

    return label, confidence, top_predictions


@app.get("/")
def home():
    return {
        "message": "Document Classifier API is running",
        "model_repo": HF_MODEL_REPO,
        "device": str(device),
        "threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        if file.content_type is None or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files are supported. Upload JPG, PNG, WEBP, etc."
            )

        image_data = file.file.read()

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid image file. Please upload a valid image."
            )
        finally:
            file.file.close()

        label, confidence, top_predictions = predict_document(image)

        if confidence < CONFIDENCE_THRESHOLD:
            final_label = "Invalid / Uncertain Document"
        else:
            final_label = label

        return {
            "label": final_label,
            "raw_label": label,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence:.2%}",
            "threshold": CONFIDENCE_THRESHOLD,
            "top_predictions": top_predictions,
        }

    except HTTPException as he:
        return {"error": he.detail}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    script_name = Path(__file__).stem
    uvicorn.run(f"{script_name}:app", host="0.0.0.0", port=8000, reload=True)