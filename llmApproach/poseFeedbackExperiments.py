# pose_feedback_pipeline.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel
import numpy as np
import json
import os

# ============================
# 0. Load Config
# ============================
with open("config.json") as f:
    config = json.load(f)

# ============================
# 1. Dataset
# ============================
class PoseFeedbackDataset(Dataset):
    def __init__(self, pose_data, label_texts, feedback_texts, tokenizer):
        self.pose_data = pose_data                # shape: (N, 51)
        self.label_texts = label_texts            # list of N strings
        self.feedback_texts = feedback_texts      # list of N strings
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        pose = torch.tensor(self.pose_data[idx], dtype=torch.float32)
        label = self.label_texts[idx]
        feedback = self.feedback_texts[idx]
        return pose, label, feedback


# ============================
# 2. Complex Pose Adapter
# ============================
class ComplexPoseAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, transformer_depth=2, n_heads=4):
        super().__init__()
        self.linear_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear_proj(x).unsqueeze(1)      # (B, 1, H)
        x = x.permute(1, 0, 2)                    # (1, B, H)
        x = self.transformer(x)                   # (1, B, H)
        x = x.permute(1, 0, 2).squeeze(1)         # (B, H)
        return self.output_proj(x)                # (B, 768)


# ============================
# 3. Training Loop
# ============================
def train_loop(dataloader, pose_adapter, bert_model, t5_model, t5_tokenizer, optimizer):
    pose_adapter.train()
    t5_model.train()

    total_loss = 0
    for pose, label_text, feedback_text in dataloader:
        optimizer.zero_grad()

        # Encode pose
        pose_emb = pose_adapter(pose)             # (B, 768)

        # Encode label
        with torch.no_grad():
            label_tokens = bert_tokenizer(list(label_text), return_tensors="pt", padding=True, truncation=True)
            label_emb = bert_model(**label_tokens).last_hidden_state.mean(dim=1)  # (B, 768)

        # Fuse and project to T5 hidden dim
        fused = torch.cat([pose_emb, label_emb], dim=-1)         # (B, 1536)
        projector = nn.Linear(1536, 512).to(pose.device)
        projected = projector(fused).unsqueeze(1)                # (B, 1, 512)

        # Decode feedback
        decoder_inputs = t5_tokenizer(["Feedback:"] * pose.shape[0], return_tensors="pt", padding=True).input_ids
        labels = t5_tokenizer(list(feedback_text), return_tensors="pt", padding=True, truncation=True).input_ids

        outputs = t5_model(encoder_outputs=(projected,), decoder_input_ids=decoder_inputs, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================
# 4. Entry Point
# ============================
if __name__ == "__main__":
    # Load dataset
    data = np.load(config["dataset"]["path"], allow_pickle=True)["data"]
    pose_data, label_texts, feedback_texts = zip(*data)

    # Tokenizers and Models
    bert_tokenizer = BertTokenizer.from_pretrained(config["label_encoder"]["tokenizer_name"])
    bert_model = BertModel.from_pretrained(config["label_encoder"]["model_name"])
    if config["label_encoder"]["freeze"]:
        for param in bert_model.parameters():
            param.requires_grad = False

    t5_tokenizer = T5Tokenizer.from_pretrained(config["text_decoder"]["tokenizer_name"])
    t5_model = T5ForConditionalGeneration.from_pretrained(config["text_decoder"]["model_name"])
    if config["text_decoder"]["freeze"]:
        for param in t5_model.parameters():
            param.requires_grad = False

    # Adapter and Optimizer
    adapter_cfg = config["adapter"]
    pose_adapter = ComplexPoseAdapter(
        input_dim=adapter_cfg["input_dim"],
        hidden_dim=adapter_cfg["hidden_dim"],
        output_dim=adapter_cfg["output_dim"],
        transformer_depth=adapter_cfg["transformer_depth"],
        n_heads=adapter_cfg["n_heads"]
    )

    optimizer = torch.optim.AdamW(
        list(pose_adapter.parameters()) + list(t5_model.parameters()),
        lr=config["training"]["learning_rate"]
    )

    # Dataset and Dataloader
    dataset = PoseFeedbackDataset(pose_data, label_texts, feedback_texts, t5_tokenizer)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=config["training"]["shuffle"])

    # Training
    for epoch in range(config["training"]["epochs"]):
        avg_loss = train_loop(dataloader, pose_adapter, bert_model, t5_model, t5_tokenizer, optimizer)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(config["training"]["save_path"]), exist_ok=True)
    torch.save({
        "pose_adapter": pose_adapter.state_dict(),
        "t5_model": t5_model.state_dict(),
    }, config["training"]["save_path"])
    print(f"Saved model to {config['training']['save_path']}")
