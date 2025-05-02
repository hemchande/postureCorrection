import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# # --- Pose Adapter to Project to Text Embedding Space ---
class PoseAdapter(nn.Module):
    def __init__(self, input_dim=51, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

pose_adapter = PoseAdapter()

# --- Load BERT as fixed text encoder ---
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()
bert_model.requires_grad_(False)

def encode_text(text):
    inputs = bert_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # (1, 768)

# --- Contrastive Loss Function ---
def cosine_contrastive_loss(pose_emb, text_emb, negatives):
    """
    pose_emb: (1, D)
    text_emb: (1, D)
    negatives: list of (1, D)
    """
    positive_sim = F.cosine_similarity(pose_emb, text_emb)  # scalar
    neg_sims = [F.cosine_similarity(pose_emb, neg) for neg in negatives]  # list of scalars

    logits = torch.cat([positive_sim] + neg_sims, dim=0).unsqueeze(0)  # (1, N+1)
    labels = torch.tensor([0]).long()  # only first is correct

    return F.cross_entropy(logits, labels)

# --- Create a Batch (Pose + Phrases) ---
preset_errors = [
    "slouching shoulders",
    "shoulder imbalance",
    "arched back",
    "bowed legs",
    "hyperextended knee"
]

# Positive label and pose
positive_text = "Straighten your back"
positive_emb = encode_text(positive_text)

# Generate pose embedding (simulated pose vector here)
pose_tensor = torch.rand(1, 51)
pose_emb = pose_adapter(pose_tensor)

# Create negatives (random other feedbacks)
negative_texts = [t for t in preset_errors if t != positive_text]
negative_embs = [encode_text(t) for t in negative_texts]

# --- Compute Loss ---
loss = cosine_contrastive_loss(pose_emb, positive_emb, negative_embs)
print(f"Contrastive Loss: {loss.item():.4f}")
