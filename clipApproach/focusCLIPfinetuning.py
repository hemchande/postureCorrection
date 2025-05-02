# ==========================
# FocusCLIP-Inspired Fine-Tuning Script
# Using MPII + ROI-based image masking
# ==========================

from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image
import os
import numpy as np
import io
from tqdm import tqdm
import torchvision.transforms as T

# Load MPII Pose dataset (already parsed to have 'image_path', 'keypoints', 'text')
# The text should already be generated using LLM as done in FocusCLIP
import json
with open("pose_feedback_dataset_mpii_hf.json") as f:
    pose_data = json.load(f)

# Focused heatmap masking function
from matplotlib.patches import Ellipse
from PIL import ImageDraw

def generate_roi_mask(image_size, keypoints):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for (x, y) in keypoints:
        if x == 0 or y == 0:
            continue
        draw.ellipse([x-20, y-20, x+20, y+20], fill=255)
    return mask

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Combine image and ROI-masked image for training
@dataclass
class FocusCLIPDataset:
    data: List[Dict[str, Any]]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry['image_path']).convert("RGB")
        keypoints = np.array(entry['keypoints'])

        # Generate ROI mask
        roi_mask = generate_roi_mask(image.size, keypoints)
        roi_image = Image.composite(image, Image.new("RGB", image.size), roi_mask)

        inputs = processor(
            text=entry['text'],
            images=[image, roi_image],  # Two branches: full + ROI
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"],  # shape (2, 3, H, W)
            "text": entry["text"]
        }

# Prepare dataset
wrapped_dataset = FocusCLIPDataset(pose_data)

@dataclass
class FocusCLIPDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        }

# Trainer with dual-branch contrastive loss
class FocusCLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)  # (B, 2, 3, H, W)

        # Split into full image and ROI views
        full_view = pixel_values[:, 0]
        roi_view = pixel_values[:, 1]

        labels = torch.arange(full_view.size(0), device=model.device)

        outputs1 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=full_view)
        outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=roi_view)

        logits1 = outputs1.logits_per_image
        logits2 = outputs2.logits_per_image

        loss1 = torch.nn.functional.cross_entropy(logits1, labels)
        loss2 = torch.nn.functional.cross_entropy(logits2, labels)
        loss = (loss1 + loss2) / 2

        return (loss, outputs1) if return_outputs else loss

# Training config
training_args = TrainingArguments(
    output_dir="./focusclip-mpii",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_dir="./logs",
    remove_unused_columns=False,
    fp16=torch.cuda.is_available()
)

# Fine-tune model
trainer = FocusCLIPTrainer(
    model=model,
    args=training_args,
    train_dataset=wrapped_dataset,
    data_collator=FocusCLIPDataCollator()
)

trainer.train()
