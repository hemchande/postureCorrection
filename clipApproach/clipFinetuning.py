from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image
import os
from icrawler.builtin import GoogleImageCrawler
from tqdm import tqdm

# Step 1: Define posture error descriptions
posture_errors = {
    "slouching_shoulders": "A human with rounded shoulders leaning slightly forward.",
    "shoulder_imbalance": "A human with uneven shoulders, one side visibly higher.",
    "arching_back": "A human with a pronounced curve in the lower back.",
    "bowing_legs": "A human standing with legs curved outward like a bow.",
    "hyperextended_knees": "A human with knees pushed backward beyond normal alignment."
}

# Step 2: Download 100 images per posture error (optional)
def download_images():
    for label in posture_errors:
        folder = f"posture_images/{label}"
        os.makedirs(folder, exist_ok=True)
        crawler = GoogleImageCrawler(storage={"root_dir": folder})
        crawler.crawl(keyword=f"{label} posture error", max_num=500, file_idx_offset=0)

# Uncomment to run download
download_images()

# Step 3: Collect image paths and descriptions
image_paths, texts = [], []
for label, desc in posture_errors.items():
    folder = f"posture_images/{label}"
    if not os.path.exists(folder):
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(folder, fname)
            try:
                _ = Image.open(path).convert("RGB")
                image_paths.append(path)
                texts.append(desc)
            except:
                continue

raw_dataset = Dataset.from_dict({"image_path": image_paths, "text": texts})

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Preprocessing
# def transform(example):
#     image = Image.open(example["image_path"]).convert("RGB")
#     inputs = processor(text=example["text"], images=image, return_tensors="pt", padding=True)
#     return {
#         "input_ids": inputs["input_ids"][0],
#         "attention_mask": inputs["attention_mask"][0],
#         "pixel_values": inputs["pixel_values"][0],
#         "text": example["text"],
#         "image_path": example["image_path"]
#     }


# def transform(example):
#     image = Image.open(example["image_path"]).convert("RGB")
#     inputs = processor(
#         text=example["text"],
#         images=image,
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True
#     )
#     return {
#         "input_ids": inputs["input_ids"].squeeze(0),
#         "attention_mask": inputs["attention_mask"].squeeze(0),
#         "pixel_values": inputs["pixel_values"].squeeze(0),
#     }


def transform(example):
    image = Image.open(example["image_path"]).convert("RGB")
    inputs = processor(
        text=example["text"],
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    return {
        "input_ids": inputs["input_ids"].squeeze(0).numpy(),
        "attention_mask": inputs["attention_mask"].squeeze(0).numpy(),
        "pixel_values": inputs["pixel_values"].squeeze(0).numpy(),
    }


dataset = raw_dataset.map(transform, remove_columns=raw_dataset.column_names)
dataset.set_format(type="python")  # For direct iteration compatibility


if len(dataset) == 0:
    raise ValueError("❌ No data found after preprocessing. Check your image paths and descriptions.")




# @dataclass
# class CLIPDataCollator:
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         return {
#             "input_ids": torch.stack([f["input_ids"] for f in features]),
#             "attention_mask": torch.stack([f["attention_mask"] for f in features]),
#             "pixel_values": torch.stack([f["pixel_values"] for f in features]),
#         }


@dataclass
class CLIPDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "pixel_values": torch.tensor([f["pixel_values"] for f in features]),
        }

from transformers import Trainer
import torch.nn.functional as F

class CLIPContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        # Move inputs to model device
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=False
        )

        # Contrastive loss (InfoNCE-style)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(len(logits_per_image), device=model.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return (loss, outputs) if return_outputs else loss



# Training arguments
training_args = TrainingArguments(
    output_dir="./clip-posture-error",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    remove_unused_columns=False,
    fp16=torch.cuda.is_available()
)

# Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     tokenizer=processor,
#     data_collator=CLIPDataCollator()
# )

trainer = CLIPContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=CLIPDataCollator()
)



# Start fine-tuning
trainer.train()

# ==============================
# Evaluation: Top-1 Accuracy
# ==============================

def evaluate_clip(model, processor, dataset):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_embeddings = []
    text_embeddings = []
    labels = []

    print("Encoding all images and texts...")
    for sample in tqdm(dataset):
        inputs = processor(
            text=sample["text"],
            images=Image.open(sample["image_path"]).convert("RGB"),
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}


        with torch.no_grad():
            img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_feat = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            image_embeddings.append(img_feat)
            text_embeddings.append(txt_feat)
            labels.append(sample["text"])

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    # Normalize
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarity = image_embeddings @ text_embeddings.T  # (N, N)
    top1 = similarity.argmax(dim=1)
    correct = sum(labels[i] == labels[j] for i, j in enumerate(top1.tolist()))
    accuracy = correct / len(labels)

    print(f"🔍 CLIP Top-1 Matching Accuracy: {accuracy:.4f}")

# Run evaluation
evaluate_clip(model, processor, dataset)


