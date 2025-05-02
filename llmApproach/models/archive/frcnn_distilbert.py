# pose_feedback_pipeline_frcnn.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import os
from datasets import load_dataset
import cv2
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torch.nn.functional as Fnn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================
# 1. Dataset using Faster R-CNN Keypoint Model + Hugging Face COCO
# ============================
class FasterRCNNPoseDataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = load_dataset("coco", split=split)
        self.model = keypointrcnn_resnet50_fpn(pretrained=True).eval()
        self.samples = []
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        for example in self.dataset:
            image_path = example['file_name']
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_tensor = self.transform(image)
            with torch.no_grad():
                outputs = self.model([image_tensor])[0]

            if len(outputs['keypoints']) > 0:
                keypoints = outputs['keypoints'][0][:17, :2]  # (17, 2)
                keypoints = keypoints.flatten()
                self.samples.append((keypoints, "label", "engage your back more"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pose, label_text, feedback_text = self.samples[idx]
        return pose, label_text, feedback_text


# ============================
# 2. Simple Pose Adapter
# ============================
class SimplePoseAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================
# 3. Training Loop
# ============================
def train_loop(dataloader, pose_adapter, bert_model, optimizer):
    pose_adapter.train()

    total_loss = 0
    loss_fn = nn.MSELoss()
    for pose, label_text, _ in dataloader:
        optimizer.zero_grad()

        pose_emb = pose_adapter(pose)

        with torch.no_grad():
            label_tokens = bert_tokenizer(list(label_text), return_tensors="pt", padding=True, truncation=True)
            label_emb = bert_model(**label_tokens).last_hidden_state.mean(dim=1)

        loss = loss_fn(pose_emb, label_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================
# 4. Evaluation
# ============================
def evaluate_model(dataloader, pose_adapter, bert_model):
    pose_adapter.eval()
    all_pose_embs, all_label_embs, all_labels = [], [], []

    with torch.no_grad():
        for pose, label_text, _ in dataloader:
            pose_emb = pose_adapter(pose)
            label_tokens = bert_tokenizer(list(label_text), return_tensors="pt", padding=True, truncation=True)
            label_emb = bert_model(**label_tokens).last_hidden_state.mean(dim=1)

            all_pose_embs.append(pose_emb)
            all_label_embs.append(label_emb)
            all_labels.extend(label_text)

    pose_matrix = torch.cat(all_pose_embs)
    label_matrix = torch.cat(all_label_embs)

    cos_sim = Fnn.cosine_similarity(pose_matrix, label_matrix).mean().item()
    print(f"Average Cosine Similarity: {cos_sim:.4f}")

    label_indices = {label: i for i, label in enumerate(set(all_labels))}
    y = [label_indices[l] for l in all_labels]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(pose_matrix.cpu().numpy(), y)
    preds = clf.predict(pose_matrix.cpu().numpy())
    acc = accuracy_score(y, preds)
    print(f"Classification Accuracy (LogReg): {acc:.4f}")


# ============================
# 5. Main Execution
# ============================
if __name__ == "__main__":
    bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    for param in bert_model.parameters():
        param.requires_grad = False

    pose_adapter = SimplePoseAdapter(input_dim=34, output_dim=768)

    optimizer = torch.optim.AdamW(
        pose_adapter.parameters(), lr=1e-4
    )

    dataset = FasterRCNNPoseDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(5):
        avg_loss = train_loop(dataloader, pose_adapter, bert_model, optimizer)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    evaluate_model(dataloader, pose_adapter, bert_model)

    torch.save({
        "pose_adapter": pose_adapter.state_dict(),
    }, "frcnn_pose_feedback_model.pt")
    print("Saved model to frcnn_pose_feedback_model.pt")

# 🔗 Pretrained model URL:
# torchvision model link for pretrained FasterRCNN with keypoints:
# https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html
