import mediapipe as mp
import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import base64
import torch
import torch.nn.functional as Fnn
import math
from openai import OpenAI

client = OpenAI(api_key="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")

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
# 2. MediaPipe Pose Setup
# ============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# ============================
# 3. Helper Functions
# ============================
def angle_between(p1, p2, p3):
    a = torch.tensor(p1)
    b = torch.tensor(p2)
    c = torch.tensor(p3)
    ba = a - b
    bc = c - b
    cos_angle = Fnn.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).item() * (180 / math.pi)


def infer_posture_label(keypoints):
    kp = {i: keypoints[i] for i in range(len(keypoints)) if keypoints[i] is not None}
    labels = []
    if 5 in kp and 7 in kp and kp[7][1] - kp[5][1] < 10:
        labels.append("slouching shoulders")
    if 6 in kp and 8 in kp and kp[8][1] - kp[6][1] < 10:
        labels.append("drooping right shoulder")
    if 5 in kp and 6 in kp and abs(kp[5][1] - kp[6][1]) > 20:
        labels.append("shoulder imbalance")
    if 5 in kp and 11 in kp and abs(kp[5][1] - kp[11][1]) < 30:
        labels.append("arched back")
    if 6 in kp and 12 in kp and abs(kp[6][1] - kp[12][1]) < 30:
        labels.append("arched back")
    if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        spine_angle = angle_between(kp[5], [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2], kp[11])
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
        eye_diff = abs(kp[1][1] - kp[2][1])
        if eye_diff > 10:
            labels.append("head tilt")
    if 0 in kp and 5 in kp and 6 in kp:
        head_mid = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
        neck_angle = angle_between(kp[0], head_mid, [(kp[11][0]+kp[12][0])/2, (kp[11][1]+kp[12][1])/2])
        if neck_angle < 160:
            labels.append("forward head posture")
    return labels if labels else ["good posture"]



def estimate_severity(label, keypoints):
    if label == "slouching shoulders" and 11 in keypoints and 13 in keypoints:
        diff = abs(keypoints[13][1] - keypoints[11][1])
        return min(1.0, diff / 40)
    return 0.5

def severity_descriptor(score):
    if score < 0.3:
        return "mild"
    elif score < 0.7:
        return "moderate"
    else:
        return "severe"

def generate_feedback(label, severity_level, image_path):
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    prompt = (
        f"You are a posture correction coach.\n"
        f"A person is exhibiting a posture issue labeled: '{label}' with {severity_level} severity.\n"
        f"Use the attached image to guide feedback.\n"
        f"Generate specific, constructive advice (max 100 tokens), and do not repeat the issue name."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a posture correction coach."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7,
        tools=[
            {
                "type": "image",
                "image": image_data,
                "mime_type": "image/png"
            }
        ]
    )
    return response.choices[0].message.content.strip()

# ============================
# 4. Generate Training Triplets Using MediaPipe Keypoints
# ============================
data = []
for ann in tqdm(annotations[:50]):
    img_path = os.path.join(coco_image_dir, id_to_file[ann['image_id']])
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = pose.process(image_rgb)
    if not result.pose_landmarks:
        continue

    keypoints = []
    for lm in result.pose_landmarks.landmark:
        keypoints.append([lm.x * image.shape[1], lm.y * image.shape[0]])

    labels = infer_posture_label(keypoints)

    for label in labels:
        severity = estimate_severity(label, keypoints)
        severity_label = severity_descriptor(severity)
        feedback = generate_feedback(label, severity_label, img_path)
        flat_pose = np.array(keypoints).flatten().tolist()
        data.append({
            "pose": flat_pose,
            "label": label,
            "severity_score": severity,
            "severity_label": severity_label,
            "feedback": feedback
        })

# ============================
# 5. Save Dataset to JSON
# ============================
with open("pose_feedback_dataset_mediapipe.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved {len(data)} examples with pose, label, severity, and feedback using MediaPipe.")
