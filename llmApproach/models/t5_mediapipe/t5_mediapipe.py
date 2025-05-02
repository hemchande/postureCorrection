import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel
import numpy as np
import json
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutput

# Properly wrap encoder_input in a BaseModelOutput



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
# Pose Adapters
# ============================
class SimplePoseAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.net(x)





class ComplexPoseAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, transformer_depth=2, n_heads=4):
        super().__init__()
        self.linear_proj = nn.Linear(input_dim, hidden_dim)
        #self.linear_proj = nn.Linear(joint_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear_proj(x).unsqueeze(1)      # (B, 1, H)
        x = x.permute(1, 0, 2)                    # (1, B, H)
        x = self.transformer(x)                   # (1, B, H)
        x = x.permute(1, 0, 2).squeeze(1)         # (B, H)
        return self.output_proj(x)    



class ComplexPoseAdapter2(nn.Module):
    def __init__(self, joint_dim, num_joints, hidden_dim, output_dim, transformer_depth=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.joint_dim = joint_dim
        self.num_joints = num_joints
        self.linear_proj = nn.Linear(joint_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.output_proj = nn.Linear(hidden_dim * num_joints, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: (B, num_joints * joint_dim)
        print(x.shape)
        B = x.shape[0]
        x = x.view(B, self.num_joints, self.joint_dim)        # (B, num_joints, joint_dim)
        x = self.linear_proj(x)                               # (B, num_joints, hidden_dim)
        x = self.input_norm(x + self.pos_embedding)           # add positional encoding
        x = self.transformer(x)                               # (B, num_joints, hidden_dim)
        x = x.flatten(start_dim=1)                            # (B, num_joints * hidden_dim)
        return self.output_norm(self.output_proj(x))          # (B, output_dim)
 


class GraphAttentionPoseAdapter(nn.Module):
    def __init__(self, input_dim, num_joints, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim // num_joints, 64)
        self.input_norm = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(64 * num_joints, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1, x.shape[1] // x.shape[-1])  # (B, num_joints, joint_dim)
        x = self.input_norm(self.input_proj(x))
        x, _ = self.attn(x, x, x)
        x = x.flatten(start_dim=1)
        return self.output_norm(self.output_proj(x))


class GraphAttentionPoseAdapter2(nn.Module):
    def __init__(self, joint_dim, num_joints, output_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.joint_dim = joint_dim
        self.num_joints = num_joints
        self.input_proj = nn.Linear(joint_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim * num_joints, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.num_joints, self.joint_dim)              # (B, num_joints, joint_dim)
        x = self.input_proj(x)                                      # (B, num_joints, hidden_dim)
        x = self.input_norm(x + self.pos_embedding)                 # add positional encoding
        x, _ = self.attn(x, x, x)                                   # self-attention over joints
        x = x.flatten(start_dim=1)                                  # (B, num_joints * hidden_dim)
        return self.output_norm(self.output_proj(x))               # (B, output_dim)



class MultiViewPoseAdapter(nn.Module):
    def __init__(self, input_dim, angle_dim, dist_dim, output_dim):
        super().__init__()
        self.pose_branch = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256)
        )
        self.angle_branch = nn.Sequential(
            nn.Linear(angle_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.dist_branch = nn.Sequential(
            nn.Linear(dist_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(self, pose_vec, angle_vec, dist_vec):
        pose_feat = self.pose_branch(pose_vec)
        angle_feat = self.angle_branch(angle_vec)
        dist_feat = self.dist_branch(dist_vec)
        fused = torch.cat([pose_feat, angle_feat, dist_feat], dim=-1)
        return self.fusion(fused)


# ============================
# 3. Training Loop
# ============================



def train_loop(dataloader, pose_adapter, t5_model, t5_tokenizer, optimizer):
    pose_adapter.train()
    t5_model.train()

    pose_to_prefix = nn.Linear(768, 512 * 4).to(next(pose_adapter.parameters()).device)  # 4 prefix tokens
    total_loss = 0

    for pose, label_text, feedback_text in dataloader:
        pose = pose.to(next(pose_adapter.parameters()).device)
        B = pose.shape[0]

        optimizer.zero_grad()

        # 1. Pose embedding → prefix tokens
        pose_emb = pose_adapter(pose)  # (B, 768)
        prefix_emb = pose_to_prefix(pose_emb).view(B, 4, 512)  # (B, 4, 512)

        # 2. Construct natural language prompt
        prompts = [f"Observed this posture flaw: {lbl}. Feedback:" for lbl in label_text]
        prompt_tokens = t5_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(pose.device)
        input_ids = prompt_tokens.input_ids  # (B, T)
        attention_mask = prompt_tokens.attention_mask  # (B, T)

        # 3. Get token embeddings from T5
        text_emb = t5_model.encoder.embed_tokens(input_ids)  # (B, T, 512)

        # 4. Concatenate pose prefix + prompt embeddings
        encoder_input = torch.cat([prefix_emb, text_emb], dim=1)  # (B, 4+T, 512)

        # 5. Labels for supervision
        tokenized = t5_tokenizer(
            list(feedback_text),
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=64
        )
        labels = tokenized.input_ids.to(pose.device)
        labels[labels == t5_tokenizer.pad_token_id] = -100

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_input)

        # 6. Forward pass
        outputs = t5_model(
            encoder_outputs=encoder_outputs,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)



#OLD

# def train_loop(dataloader, pose_adapter, bert_model, t5_model, t5_tokenizer, optimizer):
#     pose_adapter.train()
#     t5_model.train()

#     projector = nn.Linear(1536, 512).to(next(pose_adapter.parameters()).device)
#     total_loss = 0

#     for pose, label_text, feedback_text in dataloader:
#         pose = pose.to(next(pose_adapter.parameters()).device)

#         optimizer.zero_grad()

#         pose_emb = pose_adapter(pose)  # (B, 768)

#         with torch.no_grad():
#             label_tokens = bert_tokenizer(
#                 list(label_text), return_tensors="pt", padding=True, truncation=True
#             ).to(pose.device)
#             label_emb = bert_model(**label_tokens).last_hidden_state.mean(dim=1)  # (B, 768)

#         fused = torch.cat([pose_emb, label_emb], dim=-1)
#         projected = projector(fused)  # (B, 512)

#         # Repeat each projected embedding across a fixed sequence length (e.g., 16)
#         seq_len = 16
#         encoder_inputs = {
#             "inputs_embeds": projected.unsqueeze(1).repeat(1, seq_len, 1)  # shape: (B, seq_len, 512)
#         }

#         # Labels — for training loss only; decoder inputs are handled internally
#         tokenized = t5_tokenizer(
#             list(feedback_text),
#             padding="longest",
#             truncation=True,
#             return_tensors="pt",
#             max_length=64
#         )
#         labels = tokenized.input_ids.to(pose.device)
#         labels[labels == t5_tokenizer.pad_token_id] = -100  # mask pad tokens for loss

#         outputs = t5_model(**encoder_inputs, labels=labels)

#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(dataloader)


# def train_loop(dataloader, pose_adapter, bert_model, t5_model, t5_tokenizer, optimizer):
#     pose_adapter.train()
#     t5_model.train()

#     projector = nn.Linear(1536, 512).to(next(pose_adapter.parameters()).device)
#     total_loss = 0

#     for pose, label_text, feedback_text in dataloader:
#         pose = pose.to(next(pose_adapter.parameters()).device)

#         optimizer.zero_grad()

#         pose_emb = pose_adapter(pose)  # (B, 768)

#         with torch.no_grad():
#             label_tokens = bert_tokenizer(
#                 list(label_text), return_tensors="pt", padding=True, truncation=True
#             ).to(pose.device)
#             label_emb = bert_model(**label_tokens).last_hidden_state.mean(dim=1)  # (B, 768)

#         fused = torch.cat([pose_emb, label_emb], dim=-1)
#         projected = projector(fused)

#         # Pass into encoder as embedding
#         encoder_inputs = {"inputs_embeds": projected.unsqueeze(1)}  # shape (B, 1, 512)

#         # Decoder inputs
#         decoder_inputs = t5_tokenizer(
#             ["Feedback:"] * pose.shape[0],
#             return_tensors="pt", padding=True, truncation=True
#         ).input_ids.to(pose.device)

#         # Labels —> shift and pad
#         labels = t5_tokenizer(
#             list(feedback_text),
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=64  # set a reasonable limit
#         ).input_ids.to(pose.device)

#         # Replace pad token ID with -100 to ignore in loss
#         labels[labels == t5_tokenizer.pad_token_id] = -100

#         outputs = t5_model(**encoder_inputs, decoder_input_ids=decoder_inputs, labels=labels)

#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# def train_loop(dataloader, pose_adapter, bert_model, t5_model, t5_tokenizer, optimizer):
#     pose_adapter.train()
#     t5_model.train()

#     projector = nn.Linear(1536, 512).to(next(pose_adapter.parameters()).device)
#     total_loss = 0

#     for pose, label_text, feedback_text in dataloader:
#         pose = pose.to(next(pose_adapter.parameters()).device)

#         optimizer.zero_grad()

#         pose_emb = pose_adapter(pose)  # (B, 768)

#         with torch.no_grad():
#             label_tokens = bert_tokenizer(list(label_text), return_tensors="pt", padding=True, truncation=True).to(pose.device)
#             label_emb = bert_model(**label_tokens).last_hidden_state.mean(dim=1)  # (B, 768)

#         fused = torch.cat([pose_emb, label_emb], dim=-1)
#         projected = projector(fused)

#         encoder_inputs = {"inputs_embeds": projected.unsqueeze(1)}
#         decoder_inputs = t5_tokenizer(["Feedback:"] * pose.shape[0], return_tensors="pt", padding=True).input_ids.to(pose.device)
#         labels = t5_tokenizer(list(feedback_text), return_tensors="pt", padding=True, truncation=True).input_ids.to(pose.device)

#         outputs = t5_model(**encoder_inputs, decoder_input_ids=decoder_inputs, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# ============================
# 4. Main Execution
# ============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    for param in bert_model.parameters():
        param.requires_grad = False

    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)


    # Configuration for structured pose input (e.g., COCO-style with 17 joints and 3D pose: x, y, v)
    num_joints = 17
    joint_dim = 2  # typically (x, y, visibility)
    hidden_dim = 64
    output_dim = 768

    # pose_adapter = ComplexPoseAdapter2(
    #         joint_dim=joint_dim,
    #         num_joints=num_joints,
    #         hidden_dim=hidden_dim,
    #         output_dim=output_dim
    # ).to(device)



    pose_adapter = GraphAttentionPoseAdapter2(joint_dim=joint_dim,
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            output_dim=output_dim)


    # pose_adapter = SimplePoseAdapter(input_dim=34, output_dim=768).to(device)

    optimizer = torch.optim.AdamW(
        list(pose_adapter.parameters()) + list(t5_model.parameters()), lr=1e-4
    )

    dataset = PoseFeedbackDataset("pose_feedback_dataset.json")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(5):
        avg_loss = train_loop(dataloader, pose_adapter,t5_model, t5_tokenizer, optimizer)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save({
        "pose_adapter": pose_adapter.state_dict(),
        "t5_model": t5_model.state_dict(),
    }, "pose_feedback_model.pt")
    print("✅ Saved model to pose_feedback_model.pt")
