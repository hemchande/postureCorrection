# ==========================
# FocusCLIP-Inspired Fine-Tuning Script
# Using Google Images + MediaPipe ROI-based image masking
# ==========================

from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer

print(TrainingArguments.__module__)
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image, ImageDraw
import os
import numpy as np
import json
from tqdm import tqdm
import mediapipe as mp

# Define posture error descriptions for Google image crawling
posture_errors = {
    "slouching_shoulders": "A human with rounded shoulders leaning slightly forward.",
    "shoulder_imbalance": "A human with uneven shoulders, one side visibly higher.",
    "arching_back": "A human with a pronounced curve in the lower back.",
    "bowing_legs": "A human standing with legs curved outward like a bow.",
    "hyperextended_knees": "A human with knees pushed backward beyond normal alignment.",
    "rounded upper back":"A human with forward curve of the upper back, often with the head projecting forward.,
    "bowed leg": "A human with leg curved outward at the knee, away from a straight alignment.",
    "hyperextended knees": "A human with knees locked backward beyond its natural range.",
    "head tilt":"The head tilts to one side, misaligning the cervical spine.",
    "forward head posture":"The head juts forward relative to the shoulders and spine."
}

# Load image data (assuming you downloaded with icrawler)
data_dir = "google_posture_images"
pose_data = []
mp_pose = mp.solutions.pose.Pose(static_image_mode=True)

for label, text in posture_errors.items():
    folder = os.path.join(data_dir, label)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(("jpg", "jpeg", "png")):
            continue
        image_path = os.path.join(folder, fname)
        print("image path", image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            results = mp_pose.process(image_np)

            if results.pose_landmarks:
                keypoints = [
                    (lmk.x * image.width, lmk.y * image.height)
                    for lmk in results.pose_landmarks.landmark
                ]
                if keypoints and isinstance(keypoints[0], (list, tuple)) and len(keypoints[0]) == 2:
                    pose_data.append({
                        "image_path": image_path,
                        "keypoints": keypoints,
                        "text": text
                    })
                    # Filter out any broken samples before training


                else:
                    print(f"⚠️ Invalid keypoints for {image_path}, skipping...")

        except Exception as e:
            print(f"Skipping {image_path}: {e}")


required_keys = {"image_path", "keypoints", "text"}
pose_data = [d for d in pose_data if isinstance(d, dict) and required_keys.issubset(d)]

print(f"✅ Cleaned dataset: {len(pose_data)} samples remaining")
# # Save intermediate JSON if needed
# with open("pose_feedback_google_mediapipe.json", "w") as f:
#     json.dump(pose_data, f, indent=2)

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ROI heatmap masking
def generate_roi_mask(image_size, keypoints):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for (x, y) in keypoints:
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
        image = Image.open(entry['image_path']).convert("RGB")
        keypoints = entry['keypoints']

        roi_mask = generate_roi_mask(image.size, keypoints)
        roi_image = Image.composite(image, Image.new("RGB", image.size), roi_mask)

        inputs = processor(
            text=entry['text'],
            images=[image, roi_image],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        # return {
        #     "input_ids": inputs["input_ids"].squeeze(0),
        #     "attention_mask": inputs["attention_mask"].squeeze(0),
        #     "pixel_values": inputs["pixel_values"]
        # }

        return {
    "input_ids": inputs["input_ids"].squeeze(0),
    "attention_mask": inputs["attention_mask"].squeeze(0),
    "full_image": inputs["pixel_values"][0],  # (3, 224, 224)
    "roi_image": inputs["pixel_values"][1],   # (3, 224, 224)
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


# @dataclass
# class FocusCLIPDataCollator:
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         return {
#             "input_ids": torch.stack([f["input_ids"] for f in features]),
#             "attention_mask": torch.stack([f["attention_mask"] for f in features]),
#             "pixel_values": torch.stack([f["pixel_values"] for f in features]),
#         }

# class FocusCLIPTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         input_ids = inputs["input_ids"].to(model.device)
#         attention_mask = inputs["attention_mask"].to(model.device)
#         pixel_values = inputs["pixel_values"].to(model.device)

#         full_view = pixel_values[:, 0]
#         roi_view = pixel_values[:, 1]
#         labels = torch.arange(full_view.size(0), device=model.device)

#         out1 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=full_view)
#         out2 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=roi_view)

#         loss1 = torch.nn.functional.cross_entropy(out1.logits_per_image, labels)
#         loss2 = torch.nn.functional.cross_entropy(out2.logits_per_image, labels)
#         loss = (loss1 + loss2) / 2

#         return (loss, out1) if return_outputs else loss


# class FocusCLIPTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         input_ids = inputs["input_ids"].to(model.device)
#         attention_mask = inputs["attention_mask"].to(model.device)
#         pixel_values = inputs["pixel_values"].to(model.device)

#         full_view = pixel_values[:, 0]
#         roi_view = pixel_values[:, 1]
#         labels = torch.arange(full_view.size(0), device=model.device)

#         out1 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=full_view)
#         out2 = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=roi_view)

#         loss1 = torch.nn.functional.cross_entropy(out1.logits_per_image, labels)
#         loss2 = torch.nn.functional.cross_entropy(out2.logits_per_image, labels)
#         loss = (loss1 + loss2) / 2

#         return (loss, out1) if return_outputs else loss


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
        # ✅ This ensures evaluation works without crashing on unknown keys
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=True)[0]
        return (loss, None, None)



from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(pose_data, test_size=0.15, random_state=42)

def compute_clip_recall(eval_preds, k=1):
    logits_per_image = eval_preds.predictions.logits_per_image  # shape: [B, B]
    labels = torch.arange(logits_per_image.shape[0])
    topk = logits_per_image.topk(k, dim=-1).indices
    recall = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"recall@{}".format(k): recall}


# Dataset and training
train_dataset = FocusCLIPDataset(train_data)
eval_dataset = FocusCLIPDataset(val_data)
training_args = TrainingArguments(
    output_dir="./focusclip-google",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    eval_strategy="epoch",  # <== ADD THIS
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
    eval_dataset=eval_dataset,
    data_collator=FocusCLIPDataCollator(),
    compute_metrics=lambda p: compute_clip_recall(p, k=1)
)


# Dataset and training
dataset = FocusCLIPDataset(pose_data)
# training_args = TrainingArguments(
#     output_dir="./focusclip-google",
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     save_strategy="epoch",
#     logging_dir="./logs",
#     remove_unused_columns=False,
#     fp16=torch.cuda.is_available()
# )

# trainer = FocusCLIPTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=FocusCLIPDataCollator()
# )

trainer.train()
