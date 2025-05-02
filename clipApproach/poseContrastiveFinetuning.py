import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import random

# ---------------------------
# Dataset Class
# ---------------------------
class PoseContrastiveDataset(Dataset):
    def __init__(self, json_path="pose_feedback_dataset.json", num_negatives=4):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.labels = list({d["label"] for d in self.data})
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pose = torch.tensor(sample["pose"], dtype=torch.float32)
        label = sample["label"]
        negative_labels = [l for l in self.labels if l != label]
        negative_labels = random.sample(negative_labels, min(self.num_negatives, len(negative_labels)))
        return pose, label, negative_labels

# ---------------------------
# Model: Pose Adapter
# ---------------------------
class PoseAdapter(nn.Module):
    def __init__(self, input_dim=34, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Text Encoder Wrapper
# ---------------------------
def encode_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # shape (1, 768)

# ---------------------------
# Contrastive Loss
# ---------------------------
def cosine_contrastive_loss(pose_emb, pos_emb, neg_embs):
    pos_sim = F.cosine_similarity(pose_emb, pos_emb)
    neg_sims = [F.cosine_similarity(pose_emb, neg) for neg in neg_embs]
    logits = torch.cat([pos_sim] + neg_sims).unsqueeze(0)
    labels = torch.tensor([0]).long().to(pose_emb.device)
    return F.cross_entropy(logits, labels)

# ---------------------------
# Training Loop
# ---------------------------
def train_pose_adapter(
    dataset_path="pose_feedback_dataset.json",
    epochs=5,
    batch_size=1,
    lr=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoseContrastiveDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pose_adapter = PoseAdapter().to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()
    bert.requires_grad_(False)

    optimizer = torch.optim.AdamW(pose_adapter.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for pose, label, neg_labels in dataloader:
            pose = pose.to(device)
            pose_emb = pose_adapter(pose)

            pos_emb = encode_text(label[0], tokenizer, bert, device)
            neg_embs = [encode_text(lbl, tokenizer, bert, device) for lbl in neg_labels[0]]

            loss = cosine_contrastive_loss(pose_emb, pos_emb, neg_embs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} / {epochs} - Avg Loss: {total_loss / len(dataloader):.4f}")

    torch.save(pose_adapter.state_dict(), "pose_adapter_contrastive.pth")
    print("✅ Training complete. Model saved to 'pose_adapter_contrastive.pth'.")

# Run it
if __name__ == "__main__":
    train_pose_adapter()
