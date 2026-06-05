import os
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import MODEL_SAVE_PATH, DATASET_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Loading model from: {MODEL_SAVE_PATH}")

    processor = AutoImageProcessor.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_SAVE_PATH)

    model.to(device)
    model.eval()

    return processor, model


def predict_single_image(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    predicted_label = model.config.id2label[pred_idx]

    top_probs, top_indices = torch.topk(probs, k=min(3, len(probs)))

    top_3 = [
        f"{model.config.id2label[int(idx.item())]}: {float(prob.item()):.2%}"
        for prob, idx in zip(top_probs, top_indices)
    ]

    return predicted_label, confidence, " | ".join(top_3)


def batch_test_to_csv(limit=15, output_file="test_results.csv"):
    processor, model = load_model()

    dataset_root = Path(DATASET_PATH)

    if (dataset_root / "test").exists():
        search_path = dataset_root / "test"
    elif (dataset_root / "validation").exists():
        search_path = dataset_root / "validation"
    elif (dataset_root / "val").exists():
        search_path = dataset_root / "val"
    elif (dataset_root / "train").exists():
        search_path = dataset_root / "train"
    else:
        search_path = dataset_root

    categories = [d for d in search_path.iterdir() if d.is_dir()]
    print(f"Scanning {len(categories)} categories from: {search_path}")

    results = []

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    for category_folder in categories:
        images = []

        for ext in image_extensions:
            images.extend(list(category_folder.glob(ext)))

        images = images[:limit]

        for img_path in tqdm(images, desc=f"Testing {category_folder.name}"):
            try:
                predicted_label, confidence, top_3 = predict_single_image(
                    img_path,
                    processor,
                    model,
                )

                results.append({
                    "Original_Folder": category_folder.name,
                    "File_Name": img_path.name,
                    "Predicted_Label": predicted_label,
                    "Confidence": f"{confidence:.2%}",
                    "Top_3": top_3,
                    "Correct": category_folder.name.lower() in predicted_label.lower().replace(" ", "_"),
                })

            except Exception as e:
                print(f"Skipping {img_path.name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"\nCSV saved at: {os.path.abspath(output_file)}")

    if len(df) > 0:
        print("\nPrediction count:")
        print(df["Predicted_Label"].value_counts())


if __name__ == "__main__":
    batch_test_to_csv(limit=15)