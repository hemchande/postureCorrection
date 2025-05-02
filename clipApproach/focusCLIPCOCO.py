from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from torch.utils.data import Dataset
from dataclasses import dataclass
from PIL import Image, ImageDraw
from typing import List, Dict, Any
import torch
import os
import json

# === 1. Load cleaned dataset with COCO image paths ===
with open("pose_coco_feedback_dataset_4.json", "r") as f:
    pose_data = json.load(f)

# === 2. Load CLIP processor and model ===
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# === 3. Define dataset class with ROI image masking ===
def generate_roi_mask(image_size, keypoints):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for i in range(0, len(keypoints), 2):
        x, y = keypoints[i], keypoints[i + 1]
        if x > 0 and y > 0:
            draw.ellipse([x - 20, y - 20, x + 20, y + 20], fill=255)
    return mask

@dataclass
class FocusCLIPDataset(Dataset):
    data: List[Dict[str, Any]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        keypoints = entry["pose"]

        roi_mask = generate_roi_mask(image.size, keypoints)
        roi_image = Image.composite(image, Image.new("RGB", image.size), roi_mask)

        text_input = entry["label"] + ", severity: " + entry["severity_label"]

        inputs = processor(
            text=text_input,
            images=[image, roi_image],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "full_image": inputs["pixel_values"][0],
            "roi_image": inputs["pixel_values"][1],
        }

@dataclass
class FocusCLIPDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "full_image": torch.stack([f["full_image"] for f in features]),
            "roi_image": torch.stack([f["roi_image"] for f in features]),
        }


class FocusCLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        full_view = inputs["full_image"].to(model.device)
        roi_view = inputs["roi_image"].to(model.device)

        labels = torch.arange(full_view.size(0), device=model.device)

        out1 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=full_view)
        out2 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=roi_view)

        loss1 = torch.nn.functional.cross_entropy(out1.logits_per_image, labels)
        loss2 = torch.nn.functional.cross_entropy(out2.logits_per_image, labels)
        loss = (loss1 + loss2) / 2

        return (loss, out1) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Run the same as compute_loss but without labels/targets
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        full_view = inputs["full_image"].to(model.device)
        roi_view = inputs["roi_image"].to(model.device)

        labels = torch.arange(full_view.size(0), device=model.device)

        with torch.no_grad():
            out1 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=full_view)
            out2 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=roi_view)

            loss1 = torch.nn.functional.cross_entropy(out1.logits_per_image, labels)
            loss2 = torch.nn.functional.cross_entropy(out2.logits_per_image, labels)
            loss = (loss1 + loss2) / 2

        return (loss, None, None)  # We don’t need logits/labels for now





# # === 4. Define the trainer with dual-view loss ===
# class FocusCLIPTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
#         input_ids = inputs["input_ids"].to(model.device)
#         attention_mask = inputs["attention_mask"].to(model.device)
#         full_view = inputs["full_image"].to(model.device)
#         roi_view = inputs["roi_image"].to(model.device)

#         labels = torch.arange(full_view.size(0), device=model.device)

#         out1 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=full_view)
#         out2 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=roi_view)

#         loss1 = torch.nn.functional.cross_entropy(out1.logits_per_image, labels)
#         loss2 = torch.nn.functional.cross_entropy(out2.logits_per_image, labels)
#         loss = (loss1 + loss2) / 2

#         return (loss, out1) if return_outputs else loss

# === 5. Create train/val splits ===
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(pose_data, test_size=0.15, random_state=42)

train_dataset = FocusCLIPDataset(train_data)
val_dataset = FocusCLIPDataset(val_data)

# === 6. Training configuration ===
training_args = TrainingArguments(
    output_dir="./focusclip-coco",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

trainer = FocusCLIPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=FocusCLIPDataCollator()
)

trainer.train()