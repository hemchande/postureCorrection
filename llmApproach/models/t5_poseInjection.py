import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

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



class PoseFeedbackDataset(Dataset):
    def __init__(self, json_path="pose_feedback_dataset.json", expected_dim=34):
        with open(json_path, "r") as f:
            self.data = json.load(f)
            print(self.data)
        self.expected_dim = expected_dim

        # (Optional) Filter data inline
        self.data = [d for d in self.data if len(d["pose"]) == self.expected_dim]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pose = torch.tensor(sample["pose"], dtype=torch.float32)

        # Safety check
        if pose.shape[0] != self.expected_dim:
            raise ValueError(f"Inconsistent pose dimension at idx {idx}: got {pose.shape[0]} instead of {self.expected_dim}")

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
    "Given the 2D body joint coordinates where each joint is named and given as x/y position: {pose_str}. "
    "The user has the following posture issue: {label}. Please provide a correction strategy in clear, ergonomic terms.",

    "Analyze the following human pose based on named joint coordinates: {pose_str}. "
    "Detected problem: {label}. Suggest how the user can improve posture effectively.",

    "Each joint in this format represents (x, y) pixel locations. Use the pose to diagnose the issue '{label}' "
    "and return a specific corrective action for posture improvement: {pose_str}.",

    "With these 2D coordinates labeled per joint: {pose_str}, and the issue being '{label}', "
    "generate ergonomic advice or posture correction tips.",

    "This is the user's skeletal pose (joint-wise): {pose_str}. They've been identified with the following flaw: {label}. "
    "What movement or adjustment should they make to correct this?"
]



JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def pose_to_text(pose_tensor):
    coords = pose_tensor.tolist()
    joint_descriptions = []
    for i in range(len(coords) // 2):
        x = coords[2 * i]
        y = coords[2 * i + 1]
        joint_name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint{i+1}"
        joint_descriptions.append(f"{joint_name}: x={x:.2f}, y={y:.2f}")
    return "; ".join(joint_descriptions)



# ============================
# 4. Training Loop with Natural Language Prompt
# ============================
import random

def train_loop_posture_error_prompt(dataloader, pose_adapter, t5_model, t5_tokenizer, optimizer):
    t5_model.train()
    pose_adapter.train()

    total_loss = 0

    for pose, label_text, feedback_text in dataloader:
        pose = pose.to(next(pose_adapter.parameters()).device)
        optimizer.zero_grad()

        batch_prompts = []
        for i in range(pose.size(0)):
            pose_str = pose_to_text(pose[i])
            label = label_text[i]  # now treated as the error
            prompt = random.choice(POSTURE_ERROR_PROMPT_TEMPLATES).format(pose_str=pose_str, label=label)
            batch_prompts.append(prompt)

        inputs = t5_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(pose.device)
        targets = t5_tokenizer(
            list(feedback_text),
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=64
        ).input_ids.to(pose.device)
        targets[targets == t5_tokenizer.pad_token_id] = -100

        outputs = t5_model(**inputs, labels=targets)
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

    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

    pose_adapter = SimplePoseAdapter(input_dim=32, output_dim=768).to(device)
    optimizer = torch.optim.AdamW(
        list(pose_adapter.parameters()) + list(t5_model.parameters()), lr=1e-4
    )

    dataset = PoseFeedbackDataset("COCO-data-processed/pose_coco_feedback_dataset.json")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(5):
        avg_loss = train_loop_posture_error_prompt(dataloader, pose_adapter, t5_model, t5_tokenizer, optimizer)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save({
        "pose_adapter": pose_adapter.state_dict(),
        "t5_model": t5_model.state_dict(),
    }, "pose_feedback_model.pt")
    print("✅ Saved model to pose_feedback_model.pt")
