import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import random

# ============================
# 1. Load from JSON Pose Feedback Dataset
# ============================
class PoseFeedbackDataset(Dataset):
    def __init__(self, json_path="pose_feedback_dataset.json"):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pose = torch.tensor(sample["pose"], dtype=torch.float32)
        label = sample["label"]
        feedback = sample["feedback"]
        return pose, label, feedback

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
# 3. Convert Pose to Readable String
# ============================
def pose_to_text(pose_tensor):
    coords = pose_tensor.tolist()
    return " ".join(
        [f"x{i//2+1}={coords[i]:.2f}" if i % 2 == 0 else f"y{i//2+1}={coords[i]:.2f}" for i in range(len(coords))]
    )

POSTURE_ERROR_PROMPT_TEMPLATES = [
    "Posture data: {pose_str}. Detected issue: {label}. Provide ergonomic feedback:",
    "Given this body pose: {pose_str}, and posture error: {label}, generate corrective suggestions:",
    "The following pose: {pose_str} shows the problem '{label}'. What is the best ergonomic correction?",
    "Analyze this posture: {pose_str}. Identified issue: {label}. Recommend adjustments to improve alignment:",
    "Here’s the pose: {pose_str}. The user has: {label}. Suggest how to correct their posture ergonomically:"
]

# ============================
# 4. Training Loop using GPT-2
# ============================
def train_loop_gpt2_prompt(dataloader, pose_adapter, gpt_model, gpt_tokenizer, optimizer):
    gpt_model.train()
    pose_adapter.train()

    total_loss = 0

    for pose, label_text, feedback_text in dataloader:
        pose = pose.to(next(pose_adapter.parameters()).device)
        optimizer.zero_grad()

        batch_texts = []
        for i in range(pose.size(0)):
            pose_str = pose_to_text(pose[i])
            label = label_text[i]
            feedback = feedback_text[i]
            prompt = random.choice(POSTURE_ERROR_PROMPT_TEMPLATES).format(pose_str=pose_str, label=label)
            full_text = prompt + " " + feedback
            batch_texts.append(full_text)

        encodings = gpt_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = encodings.input_ids.to(pose.device)
        attention_mask = encodings.attention_mask.to(pose.device)

        labels = input_ids.clone()
        labels[input_ids == gpt_tokenizer.pad_token_id] = -100  # mask padding tokens

        outputs = gpt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# ============================
# 5. Main Execution
# ============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPT-2 setup
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # required for batching
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    pose_adapter = SimplePoseAdapter(input_dim=34, output_dim=768).to(device)

    optimizer = torch.optim.AdamW(
        list(pose_adapter.parameters()) + list(gpt_model.parameters()), lr=1e-4
    )

    dataset = PoseFeedbackDataset("pose_feedback_dataset.json")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(5):
        avg_loss = train_loop_gpt2_prompt(dataloader, pose_adapter, gpt_model, gpt_tokenizer, optimizer)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save({
        "pose_adapter": pose_adapter.state_dict(),
        "gpt2_model": gpt_model.state_dict(),
    }, "pose_feedback_gpt2.pt")
    print("✅ Saved model to pose_feedback_gpt2.pt")
