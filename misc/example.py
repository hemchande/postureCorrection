import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from ultralytics import YOLO
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration


#need to automate labels for pose corrections for each image 

# ---------------------------
# 1. Pose Adapter MLP
# ---------------------------
class PoseAdapter(nn.Module):
    def __init__(self, input_dim=51, output_dim=768):  # COCO 17 keypoints x 3
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

pose_adapter = PoseAdapter()


#train llm to take pose and text embedding to generate meaningfulke feedback 


t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")


# ---------------------------
# 2. Load Pretrained Models
# ---------------------------
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Step 1: Load Pose Estimation Model (YOLOv8 Keypoints)
pose_model = YOLO()  # or yolov8s-pose.pt for better accuracy

# Step 2: Load LLM (DistilBERT as vision-language target space)
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
# text_model.eval()

# Step 3: Adapter Layer to Project Pose Embedding -> BERT Space
class PoseToTextAdapter(nn.Module):
    def __init__(self, input_dim=34, output_dim=768):  # 17 keypoints x 2 (x,y)
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, pose_embedding):
        return self.adapter(pose_embedding)

adapter = PoseToTextAdapter()

# Step 4: Run Pose Model and Process Outputs
def extract_pose_embedding(image_path):
    results = pose_model(image_path)
    keypoints = results[0].keypoints.xy.cpu()  # (1, 17, 2)
    flat_pose = keypoints.view(-1)  # Flatten to shape (34,)
    return flat_pose

# # Step 5: Encode Reference Feedback Phrase
# def encode_text(text):
#     inputs = tokenizer(text, return_tensors="pt")
#     outputs = text_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings

# # Step 6: Match Pose to Feedback
# def match_pose_to_feedback(pose_tensor, adapter, reference_texts):
#     pose_emb = adapter(pose_tensor.unsqueeze(0))  # (1, 768)
#     scores = []
#     for text in reference_texts:
#         text_emb = encode_text(text)
#         sim = torch.nn.functional.cosine_similarity(pose_emb, text_emb)
#         scores.append((text, sim.item()))
#     return sorted(scores, key=lambda x: -x[1])



def generate_feedback_direct(t5_model, pose_emb, label_emb, target_text):
    """
    Inputs:
      pose_emb: Tensor (1, 768)
      label_emb: Tensor (1, 768)
      target_text: ground truth feedback string

    Returns:
      loss, decoded output
    """

    # Project fused embedding to 512-dim encoder hidden state (T5-small expects d_model=512)
    projector = nn.Linear(768 * 2, 512)
    fused = torch.cat([pose_emb, label_emb], dim=-1)  # (1, 1536)
    projected = projector(fused).unsqueeze(1)         # (1, 1, 512) = (batch, seq_len, hidden_dim)

    # Decoder input: "Feedback:"
    decoder_input_ids = t5_tokenizer("Feedback:", return_tensors="pt").input_ids

    # Labels (feedback supervision)
    labels = t5_tokenizer(target_text, return_tensors="pt").input_ids

    # Call T5 forward with encoder_outputs
    outputs = t5_model(
        encoder_outputs=(projected,),  # inject the embedding directly
        decoder_input_ids=decoder_input_ids,
        labels=labels
    )

    # Decode output (optional for inspection)
    decoded = t5_tokenizer.decode(outputs.logits.argmax(-1)[0], skip_special_tokens=True)
    return outputs.loss, decoded


# === EXAMPLE USAGE ===
# pose_tensor = extract_pose_embedding("example.jpg")
# feedbacks = ["Straighten your back", "Relax your shoulders", "Do not hyperextend"]
# results = match_pose_to_feedback(pose_tensor, adapter, feedbacks)
# print("Best Feedback:", results[0])
# Pose and label embeddings
pose_emb = pose_adapter(torch.rand(1, 51))  # (1, 768)
label_emb = bert_model(**bert_tokenizer("arched back", return_tensors="pt"))[0].mean(dim=1)  # (1, 768)

# Target text
target_feedback = "Tuck your pelvis and engage your core to support your lower back."

# Get loss + prediction
loss, prediction = generate_feedback_direct(t5_model, pose_emb, label_emb, target_feedback)

print("Loss:", loss.item())
print("Generated:", prediction)

