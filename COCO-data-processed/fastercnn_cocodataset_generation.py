import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import json
import cv2
import os
from tqdm import tqdm
from PIL import Image
import base64
import torch.nn.functional as Fnn
import math
import openai
from openai import OpenAI


client = OpenAI(api_key="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")

openai.api_key = "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"

# ============================
# 1. Load COCO Dataset Locally
# ============================
coco_image_dir = "../Downloads/train2014"
coco_ann_file = "../Downloads/annotations/person_keypoints_train2014.json"

with open(coco_ann_file, 'r') as f:
    coco = json.load(f)

annotations = [ann for ann in coco['annotations'] if ann['num_keypoints'] > 0]
id_to_file = {img['id']: img['file_name'] for img in coco['images']}

# ============================
# 2. Load Pretrained Faster R-CNN
# ============================
# fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True)
# fasterrcnn.eval()

# Step 2: Define inference functions
fasterrcnn =keypointrcnn_resnet50_fpn(pretrained=True)
fasterrcnn.eval()

# ============================
# 3. Helper Functions
# ============================
# def angle_between(p1, p2, p3):
#     a = torch.tensor(p1)
#     b = torch.tensor(p2)
#     c = torch.tensor(p3)
#     ba = a - b
#     bc = c - b
#     cos_angle = Fnn.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
#     return torch.acos(cos_angle).item() * (180 / math.pi)

# def infer_posture_label(keypoints):
#     kp = {i: keypoints[i] for i in range(len(keypoints)) if keypoints[i] is not None}
#     labels = []
#     if 5 in kp and 7 in kp and kp[7][1] - kp[5][1] < 10:
#         labels.append("slouching shoulders")
#     if 6 in kp and 8 in kp and kp[8][1] - kp[6][1] < 10:
#         labels.append("drooping right shoulder")
#     if 5 in kp and 6 in kp and abs(kp[5][1] - kp[6][1]) > 20:
#         labels.append("shoulder imbalance")
#     if 5 in kp and 11 in kp and abs(kp[5][1] - kp[11][1]) < 30:
#         labels.append("arched back")
#     if 6 in kp and 12 in kp and abs(kp[6][1] - kp[12][1]) < 30:
#         labels.append("arched back")
#     if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
#         spine_angle = angle_between(kp[5], [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2], kp[11])
#         if spine_angle < 150:
#             labels.append("rounded upper back")
#     if 11 in kp and 12 in kp and abs(kp[11][1] - kp[12][1]) > 20:
#         labels.append("hip tilt")
#     if 13 in kp and 15 in kp and abs(kp[13][0] - kp[15][0]) > 40:
#         labels.append("bowed left leg")
#     if 14 in kp and 16 in kp and abs(kp[14][0] - kp[16][0]) > 40:
#         labels.append("bowed right leg")
#     if 11 in kp and 13 in kp and kp[13][1] < kp[11][1]:
#         labels.append("hyperextended left knee")
#     if 12 in kp and 14 in kp and kp[14][1] < kp[12][1]:
#         labels.append("hyperextended right knee")
#     if 0 in kp and 1 in kp and 2 in kp:
#         eye_diff = abs(kp[1][1] - kp[2][1])
#         if eye_diff > 10:
#             labels.append("head tilt")
#     if 0 in kp and 5 in kp and 6 in kp:
#         head_mid = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
#         neck_angle = angle_between(kp[0], head_mid, [(kp[11][0]+kp[12][0])/2, (kp[11][1]+kp[12][1])/2])
#         if neck_angle < 160:
#             labels.append("forward head posture")
#     return labels if labels else ["good posture"]



def angle_between(p1, p2, p3):
    a = torch.tensor(p1)
    b = torch.tensor(p2)
    c = torch.tensor(p3)
    ba = a - b
    bc = c - b
    cos_angle = F.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).item() * (180 / math.pi)

def infer_posture_label(keypoints):
    kp = {
        i: keypoints[i]
        for i in range(len(keypoints))
        if keypoints[i] is not None and not (keypoints[i][0] == 0 and keypoints[i][1] == 0)
    }

    labels = []

    if 5 in kp and 7 in kp and (kp[7][1] - kp[5][1] < 10):
        labels.append("slouching shoulders")
    if 6 in kp and 8 in kp and (kp[8][1] - kp[6][1] < 10):
        labels.append("drooping right shoulder")
    if 5 in kp and 6 in kp and (abs(kp[5][1] - kp[6][1]) > 20):
        labels.append("shoulder imbalance")

    if 5 in kp and 11 in kp and (abs(kp[5][1] - kp[11][1]) < 30):
        labels.append("arched back")
    if 6 in kp and 12 in kp and (abs(kp[6][1] - kp[12][1]) < 30):
        labels.append("arched back")
    if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        spine_angle = angle_between(
            kp[5],
            [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2],
            kp[11]
        )
        if spine_angle < 150:
            labels.append("rounded upper back")

    if 11 in kp and 12 in kp and abs(kp[11][1] - kp[12][1]) > 20:
        labels.append("hip tilt")

    if 13 in kp and 15 in kp and abs(kp[13][0] - kp[15][0]) > 40:
        labels.append("bowed left leg")
    if 14 in kp and 16 in kp and abs(kp[14][0] - kp[16][0]) > 40:
        labels.append("bowed right leg")
    if 11 in kp and 13 in kp and kp[13][1] < kp[11][1]:
        labels.append("hyperextended left knee")
    if 12 in kp and 14 in kp and kp[14][1] < kp[12][1]:
        labels.append("hyperextended right knee")

    if 0 in kp and 1 in kp and 2 in kp:
        if abs(kp[1][1] - kp[2][1]) > 10:
            labels.append("head tilt")
    if 0 in kp and 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        head_mid = [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2]
        torso_mid = [(kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2]
        neck_angle = angle_between(kp[0], head_mid, torso_mid)
        if neck_angle < 160:
            labels.append("forward head posture")

    return labels if labels else ["good posture"]


import random

def augment_keypoints(keypoints, img_width, img_height, n_augment=3):
    augmented = []

    for _ in range(n_augment):
        kps = keypoints.copy()

        # Horizontal flip
        if random.random() < 0.5:
            kps[:, 0] = img_width - kps[:, 0]
            # Flip left/right joints (swap 5 ↔ 6, 7 ↔ 8, etc.)
            flip_map = {
                5: 6, 6: 5,
                7: 8, 8: 7,
                11: 12, 12: 11,
                13: 14, 14: 13,
                15: 16, 16: 15
            }
            for a, b in flip_map.items():
                kps[a], kps[b] = kps[b].copy(), kps[a].copy()

        # Jitter
        noise = np.random.normal(0, 2.0, kps.shape)  # mild pixel jitter
        kps += noise

        # Scaling
        scale = random.uniform(0.9, 1.1)
        center = np.array([img_width / 2, img_height / 2])
        kps = (kps - center) * scale + center

        # Affine transform (translation)
        dx = random.uniform(-5, 5)
        dy = random.uniform(-5, 5)
        kps += np.array([dx, dy])

        # Clip to image bounds
        kps[:, 0] = np.clip(kps[:, 0], 0, img_width - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, img_height - 1)

        augmented.append(kps)

    return augmented


def estimate_severity(label, keypoints):
    if label == "slouching shoulders" and 5 in keypoints and 7 in keypoints:
        diff = abs(keypoints[7][1] - keypoints[5][1])
        return min(1.0, diff / 40)
    elif label == "shoulder imbalance" and 5 in keypoints and 6 in keypoints:
        diff = abs(keypoints[5][1] - keypoints[6][1])
        return min(1.0, diff / 40)
    return 0.5

def severity_descriptor(score):
    if score < 0.3:
        return "mild"
    elif score < 0.7:
        return "moderate"
    else:
        return "severe"

# def generate_feedback(label, severity_level, image_path):
#     with open(image_path, "rb") as img_file:
#         base64_image = base64.b64encode(img_file.read()).decode("utf-8")
#     return f"{label} correction at {severity_level} severity. (Simulated feedback)"


# def generate_feedback(label, severity_level, image_path):
#     with open(image_path, "rb") as img_file:
#         image_data = img_file.read()

#     prompt = (
#         f"You are a posture correction coach.\n"
#         f"A person is exhibiting a posture issue labeled: '{label}' with {severity_level} severity.\n"
#         f"Use the attached image to guide feedback.\n"
#         f"Generate specific, constructive advice (max 100 tokens), and do not repeat the issue name."
#     )

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You're a posture correction coach."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=100,
#         temperature=0.7,
#         tools=[
#             {
#                 "type": "image",
#                 "image": image_data,
#                 "mime_type": "image/png"  # or "image/jpeg" depending on file
#             }
#         ]
#     )
#     return response.choices[0].message.content.strip()


from PIL import Image
import io

def generate_feedback(label, severity_level, image_path):
    image = Image.open(image_path).convert("RGB")
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    prompt = (
        f"You are a posture correction coach.\n"
        f"A person is exhibiting a posture issue labeled: '{label}' with {severity_level} severity.\n"
        f"Generate specific, constructive advice (max 100 tokens), and do not repeat the issue name.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}" }}
                ]
            }
        ],
        max_tokens=100,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()



# ============================
# 4. Generate Training Triplets Using Faster R-CNN Keypoints
# ============================
data = []
for ann in tqdm(annotations):
    img_path = os.path.join(coco_image_dir, id_to_file[ann['image_id']])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    with torch.no_grad():
        prediction = fasterrcnn([img_tensor])[0]
        print(prediction)

    if "keypoints" not in prediction or len(prediction["keypoints"]) == 0:
        continue

    keypoints = prediction["keypoints"][0, :, :2].cpu().numpy()  # shape (17, 2)
    labels = infer_posture_label(keypoints)
    img_width, img_height = img.size
    augmented_keypoints = augment_keypoints(keypoints, img_width, img_height)

    for label in labels:
        severity = estimate_severity(label, keypoints)
        severity_label = severity_descriptor(severity)
        feedback = generate_feedback(label, severity_label, img_path)
        flat_pose = keypoints.flatten().tolist()
        data.append({
            "pose": flat_pose,
            "label": label,
            "severity_score": severity,
            "severity_label": severity_label,
            "feedback": feedback
        })

        # Add augmented versions
        for aug_kp in augmented_keypoints:
            aug_severity = estimate_severity(label, aug_kp)
            aug_severity_label = severity_descriptor(aug_severity)
            data.append({
                "pose": aug_kp.flatten().tolist(),
                "label": label,
                "severity_score": aug_severity,
                "severity_label": aug_severity_label,
                "feedback": feedback  # reuse original feedback for now
            })

# ============================
# 5. Save Dataset to JSON
# ============================
with open("pose_coco_feedback_dataset_fasterrcnn.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved {len(data)} examples with pose, label, severity, and feedback using Faster R-CNN.")
