import os
import io
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = BASE_DIR / "hf_cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR)

import torch
import easyocr
import numpy as np
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

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", "0.15"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(
    HF_MODEL_REPO,
    cache_dir=str(LOCAL_CACHE_DIR)
)

model = AutoModelForImageClassification.from_pretrained(
    HF_MODEL_REPO,
    cache_dir=str(LOCAL_CACHE_DIR)
)

model.to(device)
model.eval()

ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

print("\n===== MODEL LABELS =====")
print(model.config.id2label)
print("========================\n")


OCR_PATTERNS = {
    "Aadhaar Card": [
        "aadhaar", "aadhar", "uidai", "unique identification", "government of india",
        "vid", "dob", "year of birth"
    ],
    "PAN Card": [
        "income tax department", "permanent account number", "pan", "father's name",
        "signature"
    ],
    "Driving License": [
        "driving licence", "driving license", "dl no", "licence no", "license no",
        "transport", "non transport", "valid till", "rto"
    ],
    "Passport": [
        "passport", "republic of india", "passport no", "nationality", "place of birth",
        "date of expiry"
    ],
    "Voter ID": [
        "election commission", "elector", "voter", "epic", "identity card"
    ],
    "Bank Statement": [
        "bank statement", "account statement", "transaction", "balance", "debit",
        "credit", "ifsc", "account number", "closing balance"
    ],
    "Salary Slip": [
        "salary slip", "payslip", "pay slip", "earnings", "deductions",
        "net pay", "gross salary", "basic salary", "hra"
    ],
    "Court Paper": [
        "court", "petition", "affidavit", "plaintiff", "defendant", "hon'ble",
        "judgement", "case no"
    ],
    "Application Form": [
        "application form", "applicant", "form no", "declaration", "signature of applicant"
    ],
    "Experience Letter": [
        "experience letter", "worked from", "employment", "tenure", "relieved",
        "designation", "served as", "experience certificate"
    ],
    "Mark Sheet": [
        "marksheet", "mark sheet", "statement of marks", "grade", "semester",
        "examination", "cgpa", "sgpa", "roll no"
    ],
}

UNSEEN_DOC_PATTERNS = {
    "Offer Letter": [
        "offer letter", "we are pleased to offer", "ctc", "compensation",
        "joining date", "position offered", "annual package"
    ],
    "Appointment Letter": [
        "appointment letter", "appointed as", "date of joining", "terms of appointment"
    ],
    "Relieving Letter": [
        "relieving letter", "relieved from", "last working day", "notice period"
    ],
    "Resignation Letter": [
        "resignation", "resign", "notice period", "stepping down"
    ],
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_ocr_text(image: Image.Image) -> str:
    image_np = np.array(image.convert("RGB"))
    result = ocr_reader.readtext(image_np, detail=0, paragraph=True)
    return " ".join(result)


def keyword_score(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword.lower() in text)


def analyze_ocr_text(ocr_text: str):
    text = normalize_text(ocr_text)

    known_scores = {
        label: keyword_score(text, keywords)
        for label, keywords in OCR_PATTERNS.items()
    }

    unseen_scores = {
        label: keyword_score(text, keywords)
        for label, keywords in UNSEEN_DOC_PATTERNS.items()
    }

    best_known = max(known_scores, key=known_scores.get)
    best_known_score = known_scores[best_known]

    best_unseen = max(unseen_scores, key=unseen_scores.get)
    best_unseen_score = unseen_scores[best_unseen]

    return {
        "best_known_from_ocr": best_known if best_known_score > 0 else None,
        "known_score": best_known_score,
        "possible_unseen_document": best_unseen if best_unseen_score >= 2 else None,
        "unseen_score": best_unseen_score,
        "ocr_text_preview": ocr_text[:500],
    }


def predict_with_model(image: Image.Image):
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(3, probs.shape[0]))

    top_predictions = []
    for prob, idx in zip(top_probs, top_indices):
        idx = int(idx.item())
        top_predictions.append({
            "label": model.config.id2label[idx],
            "confidence": round(float(prob.item()), 4),
        })

    top1 = top_predictions[0]
    top2 = top_predictions[1] if len(top_predictions) > 1 else None

    margin = top1["confidence"] - top2["confidence"] if top2 else top1["confidence"]

    return top1["label"], top1["confidence"], margin, top_predictions


@app.get("/")
def home():
    return {
        "message": "Document Classifier API is running",
        "model_repo": HF_MODEL_REPO,
        "device": str(device),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "ocr": "EasyOCR",
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

        model_label, confidence, margin, top_predictions = predict_with_model(image)

        ocr_text = extract_ocr_text(image)
        ocr_analysis = analyze_ocr_text(ocr_text)

        final_label = model_label
        warning = None
        possible_actual_document = None

        if ocr_analysis["possible_unseen_document"]:
            possible_actual_document = ocr_analysis["possible_unseen_document"]
            warning = (
                f"This document may be an unseen type: {possible_actual_document}. "
                f"The closest trained class predicted by the model is {model_label}."
            )

        elif confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            warning = (
                "Low confidence or close prediction margin. "
                "Document may belong to an unseen or visually similar category."
            )

            if ocr_analysis["best_known_from_ocr"]:
                final_label = ocr_analysis["best_known_from_ocr"]

        return {
            "label": final_label,
            "raw_model_label": model_label,
            "closest_known_class": model_label,
            "possible_actual_document": possible_actual_document,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence:.2%}",
            "margin": round(margin, 4),
            "warning": warning,
            "top_predictions": top_predictions,
            "ocr_best_known_class": ocr_analysis["best_known_from_ocr"],
            "ocr_known_score": ocr_analysis["known_score"],
            "ocr_unseen_score": ocr_analysis["unseen_score"],
            "ocr_text_preview": ocr_analysis["ocr_text_preview"],
        }

    except HTTPException as he:
        return {"error": he.detail}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    script_name = Path(__file__).stem
    uvicorn.run(f"{script_name}:app", host="0.0.0.0", port=8000, reload=True)