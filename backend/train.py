import os
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    EarlyStoppingCallback,
)

# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = Path("/home/devadmin/BGV_CLASSIFICATION_TRAINING_DATA")
MODEL_SAVE_PATH = BASE_DIR / "models" / "final_model"
CACHE_DIR = BASE_DIR / "hf_cache"

MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)

MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"

# =========================
# LABEL MAPPING
# =========================

custom_label_map = {
    "aadhaar": "Aadhaar Card",
    "application_form": "Application Form",
    "bank_statement": "Bank Statement",
    "court_paper": "Court Paper",
    "dl": "Driving License",
    "experience_letter": "Experience Letter",
    "marksheets": "Mark Sheet",
    "Pan": "PAN Card",
    "pan": "PAN Card",
    "passport": "Passport",
    "payslips": "Salary Slip",
    "voter": "Voter ID",
}

# =========================
# TRANSFORMS
# =========================

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
    transforms.RandomRotation(degrees=7, fill=(255, 255, 255)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.04, 0.04),
        scale=(0.90, 1.10),
        shear=3,
        fill=(255, 255, 255),
    ),
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.08,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


def train_transform(batch):
    batch["pixel_values"] = [
        train_transforms(img.convert("RGB")) for img in batch["image"]
    ]

    batch.pop("image", None)
    return batch


def val_transform(batch):
    batch["pixel_values"] = [
        val_transforms(img.convert("RGB")) for img in batch["image"]
    ]

    batch.pop("image", None)
    return batch


# =========================
# METRICS
# =========================

accuracy_metric = load("accuracy")
f1_metric = load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_metric.compute(
        predictions=preds,
        references=labels,
    )["accuracy"]

    f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="macro",
    )["f1"]

    return {
        "accuracy": acc,
        "macro_f1": f1,
    }


# =========================
# TRAINING
# =========================

def train_model():
    print(f"Loading dataset from: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

    dataset = load_dataset(
        "imagefolder",
        data_dir=str(DATASET_PATH),
        cache_dir=str(CACHE_DIR),
    )

    if "validation" in dataset:
        train_ds = dataset["train"]
        val_ds = dataset["validation"]
    else:
        print("No validation folder found. Splitting training data 80/20...")

        split = dataset["train"].train_test_split(
            test_size=0.2,
            seed=42,
            stratify_by_column="label",
        )

        train_ds = split["train"]
        val_ds = split["test"]

    folder_names = train_ds.features["label"].names

    id2label = {
        i: custom_label_map.get(name, name)
        for i, name in enumerate(folder_names)
    }

    label2id = {
        label: i
        for i, label in id2label.items()
    }

    print("\n===== FINAL CLASSES =====")
    for i, label in id2label.items():
        print(f"{i}: {label}")
    print("=========================\n")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        cache_dir=str(CACHE_DIR),
    )

    train_ds.set_transform(train_transform)
    val_ds.set_transform(val_transform)

    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_PATH),
        remove_unused_columns=False,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,

        num_train_epochs=12,
        learning_rate=2e-5,
        weight_decay=0.05,
        warmup_ratio=0.08,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,

        fp16=torch.cuda.is_available(),

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",

        label_smoothing_factor=0.05,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    print("Starting training...")
    trainer.train()

    print("\nEvaluating best model...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"\nSaving final model to: {MODEL_SAVE_PATH}")
    trainer.save_model(str(MODEL_SAVE_PATH))
    processor.save_pretrained(str(MODEL_SAVE_PATH))

    print("\nReload check...")
    reloaded_model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_SAVE_PATH),
        local_files_only=True,
    )

    print("\n===== SAVED MODEL LABELS =====")
    print(reloaded_model.config.id2label)
    print("==============================\n")

    print("Training Complete")


if __name__ == "__main__":
    train_model()